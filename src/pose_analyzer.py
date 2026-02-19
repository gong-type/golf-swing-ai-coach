from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import Body, PoseTracker


@dataclass
class FrameAnalysis:
    keypoints: np.ndarray | None
    scores: np.ndarray | None
    angles: dict[str, float]
    phase: str
    phase_confidence: float
    device: str
    coach_tip: str | None = None
    coach_level: str = "info"


class GolfPoseAnalyzer:
    """Analyze pose frames and estimate basic golf coaching signals."""

    PHASE_CN = {
        "Address": "准备站位",
        "Backswing": "上杆",
        "Top": "顶点",
        "Downswing": "下杆",
        "Impact": "击球",
        "Follow-through": "送杆",
        "Finish": "收杆",
    }

    KEYPOINTS = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(
        self,
        mode: str = "performance",
        device: str = "auto",
        det_frequency: int = 8,
        score_thr: float = 0.35,
        ema_alpha: float = 0.55,
        max_missing_frames: int = 8,
        inference_scale: float = 1.0,
        tracking: bool = False,
    ) -> None:
        self.score_thr = score_thr
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 0.95))
        self.max_missing_frames = max_missing_frames
        self.inference_scale = float(np.clip(inference_scale, 0.5, 1.0))
        self.device = self._resolve_device(device)
        self.pose_tracker = self._create_pose_tracker(
            mode=mode,
            det_frequency=max(1, int(det_frequency)),
            tracking=tracking,
        )

        self._wrist_height_history: Deque[float] = deque(maxlen=10)
        self._phase = "Address"
        self._candidate_phase = self._phase
        self._candidate_count = 0
        self._smoothed_keypoints: np.ndarray | None = None
        self._smoothed_scores: np.ndarray | None = None
        self._smoothed_angles: dict[str, float] = {}
        self._missing_frames = 0
        self._active_tip: str | None = None
        self._active_tip_level = "info"
        self._tip_frames_left = 0
        self._tip_cooldown = 0

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        return "cpu"

    def _create_pose_tracker(
        self, mode: str, det_frequency: int, tracking: bool
    ) -> PoseTracker:
        kwargs = {
            "mode": mode,
            "det_frequency": det_frequency,
            "backend": "onnxruntime",
            "device": self.device,
        }
        # Newer rtmlib versions accept tracking flag and can run faster with tracking disabled.
        try:
            return PoseTracker(Body, tracking=tracking, **kwargs)
        except TypeError:
            return PoseTracker(Body, **kwargs)

    def reset(self) -> None:
        self._wrist_height_history.clear()
        self._phase = "Address"
        self._candidate_phase = self._phase
        self._candidate_count = 0
        self._smoothed_keypoints = None
        self._smoothed_scores = None
        self._smoothed_angles.clear()
        self._missing_frames = 0
        self._active_tip = None
        self._active_tip_level = "info"
        self._tip_frames_left = 0
        self._tip_cooldown = 0

    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysis:
        self._tick_coach_tip()

        infer_frame = frame
        scale_x = 1.0
        scale_y = 1.0
        if self.inference_scale < 0.999:
            h, w = frame.shape[:2]
            infer_w = max(320, int(w * self.inference_scale))
            infer_h = max(256, int(h * self.inference_scale))
            infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            scale_x = w / float(infer_w)
            scale_y = h / float(infer_h)

        keypoints, scores = self.pose_tracker(infer_frame)
        person = self._pick_primary_person(keypoints, scores)

        if person is None:
            self._missing_frames += 1
            self._issue_coach_tip("请完整入镜，双手和髋部尽量都可见", "info", force=True)
            if (
                self._smoothed_keypoints is not None
                and self._smoothed_scores is not None
                and self._missing_frames <= self.max_missing_frames
            ):
                faded_scores = np.clip(self._smoothed_scores - 0.05, 0.0, 1.0)
                return FrameAnalysis(
                    keypoints=self._smoothed_keypoints.copy(),
                    scores=faded_scores,
                    angles=dict(self._smoothed_angles),
                    phase=self.PHASE_CN.get(self._phase, self._phase),
                    phase_confidence=0.15,
                    device=self.device,
                    coach_tip=self._active_tip,
                    coach_level=self._active_tip_level,
                )

            return FrameAnalysis(
                keypoints=None,
                scores=None,
                angles={},
                phase=self.PHASE_CN.get(self._phase, self._phase),
                phase_confidence=0.0,
                device=self.device,
                coach_tip=self._active_tip,
                coach_level=self._active_tip_level,
            )

        self._missing_frames = 0
        kps, scs = person
        if scale_x != 1.0 or scale_y != 1.0:
            kps = np.asarray(kps, dtype=np.float32).copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            scs = np.asarray(scs, dtype=np.float32).copy()

        kps, scs = self._smooth_pose(kps, scs)
        angles = self._calculate_angles(kps, scs)
        angles = self._smooth_angles(angles)
        phase, confidence = self._estimate_phase(kps, scs)
        coach = self._generate_coach_tip(phase, confidence, angles, kps, scs)
        if coach is not None:
            tip_text, tip_level = coach
            self._issue_coach_tip(tip_text, tip_level)

        return FrameAnalysis(
            keypoints=kps,
            scores=scs,
            angles=angles,
            phase=self.PHASE_CN.get(phase, phase),
            phase_confidence=confidence,
            device=self.device,
            coach_tip=self._active_tip,
            coach_level=self._active_tip_level,
        )

    def _tick_coach_tip(self) -> None:
        if self._tip_frames_left > 0:
            self._tip_frames_left -= 1
        if self._tip_cooldown > 0:
            self._tip_cooldown -= 1
        if self._tip_frames_left <= 0:
            self._active_tip = None
            self._active_tip_level = "info"

    def _issue_coach_tip(
        self,
        text: str,
        level: str,
        duration: int = 42,
        cooldown: int = 18,
        force: bool = False,
    ) -> None:
        if self._active_tip == text:
            self._tip_frames_left = max(self._tip_frames_left, duration // 2)
            return

        can_switch = force or self._tip_cooldown <= 0 or self._active_tip is None
        if not can_switch:
            return

        self._active_tip = text
        self._active_tip_level = level
        self._tip_frames_left = duration
        self._tip_cooldown = cooldown

    def _generate_coach_tip(
        self,
        phase: str,
        phase_confidence: float,
        angles: dict[str, float],
        kps: np.ndarray,
        scs: np.ndarray,
    ) -> tuple[str, str] | None:
        right_wrist = self._safe_point(kps, scs, self.KEYPOINTS["right_wrist"])
        left_wrist = self._safe_point(kps, scs, self.KEYPOINTS["left_wrist"])
        right_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["right_shoulder"])
        left_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["left_shoulder"])
        right_hip = self._safe_point(kps, scs, self.KEYPOINTS["right_hip"])
        left_hip = self._safe_point(kps, scs, self.KEYPOINTS["left_hip"])

        spine_tilt = angles.get("spine_tilt")
        shoulder_rotation = angles.get("shoulder_rotation")
        left_knee = angles.get("left_knee_flex")
        right_knee = angles.get("right_knee_flex")

        if spine_tilt is not None:
            if spine_tilt < 16:
                return "身体略微弯曲，再前倾一点更稳", "warn"
            if spine_tilt > 42:
                return "上身前倾有点多，稍微立起来一点", "warn"

        if phase in {"Backswing", "Top"}:
            if (
                right_wrist is not None
                and left_shoulder is not None
                and right_shoulder is not None
                and right_hip is not None
                and left_hip is not None
            ):
                mid_shoulder = (left_shoulder + right_shoulder) / 2.0
                mid_hip = (left_hip + right_hip) / 2.0
                torso_len = max(float(np.linalg.norm(mid_shoulder - mid_hip)), 1.0)
                # Wrist should generally move above shoulder line near top.
                if right_wrist[1] > mid_shoulder[1] - 0.06 * torso_len:
                    return "手部需要更高一点，上杆再完整些", "warn"

        if left_knee is not None and right_knee is not None:
            if abs(left_knee - right_knee) > 42:
                return "双膝角度差有点大，重心再稳定一些", "warn"

        if (
            phase == "Impact"
            and phase_confidence > 0.40
            and spine_tilt is not None
            and shoulder_rotation is not None
            and 20 <= spine_tilt <= 35
            and 155 <= shoulder_rotation <= 190
        ):
            return "完美！击球姿态很稳，继续保持", "good"

        if phase in {"Finish", "Follow-through"} and phase_confidence > 0.45:
            return "收杆不错，平衡控制得很好", "good"

        return None

    def _smooth_pose(self, kps: np.ndarray, scs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kps_arr = np.asarray(kps, dtype=np.float32)
        scs_arr = np.asarray(scs, dtype=np.float32)
        if self._smoothed_keypoints is None or self._smoothed_scores is None:
            self._smoothed_keypoints = kps_arr.copy()
            self._smoothed_scores = scs_arr.copy()
            return self._smoothed_keypoints.copy(), self._smoothed_scores.copy()

        alpha = self.ema_alpha
        self._smoothed_keypoints = alpha * kps_arr + (1.0 - alpha) * self._smoothed_keypoints
        self._smoothed_scores = alpha * scs_arr + (1.0 - alpha) * self._smoothed_scores
        self._smoothed_scores = np.clip(self._smoothed_scores, 0.0, 1.0)
        return self._smoothed_keypoints.copy(), self._smoothed_scores.copy()

    def _smooth_angles(self, angles: dict[str, float]) -> dict[str, float]:
        if not angles:
            return dict(self._smoothed_angles)

        alpha = 0.45
        for key, value in angles.items():
            if key in self._smoothed_angles:
                self._smoothed_angles[key] = (
                    alpha * float(value) + (1.0 - alpha) * self._smoothed_angles[key]
                )
            else:
                self._smoothed_angles[key] = float(value)
        return dict(self._smoothed_angles)

    def _pick_primary_person(
        self, keypoints: np.ndarray | None, scores: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if keypoints is None or scores is None:
            return None

        kps = np.asarray(keypoints)
        scs = np.asarray(scores)

        if kps.size == 0 or scs.size == 0:
            return None

        if kps.ndim == 2:
            return kps, scs

        person_idx = int(np.argmax(np.mean(scs, axis=1)))
        return kps[person_idx], scs[person_idx]

    def _calculate_angles(self, kps: np.ndarray, scs: np.ndarray) -> dict[str, float]:
        angles: dict[str, float] = {}

        left_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["left_shoulder"])
        right_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["right_shoulder"])
        left_hip = self._safe_point(kps, scs, self.KEYPOINTS["left_hip"])
        right_hip = self._safe_point(kps, scs, self.KEYPOINTS["right_hip"])
        if (
            left_shoulder is not None
            and right_shoulder is not None
            and left_hip is not None
            and right_hip is not None
        ):
            mid_shoulder = (left_shoulder + right_shoulder) / 2.0
            mid_hip = (left_hip + right_hip) / 2.0
            vec = mid_shoulder - mid_hip
            # Spine tilt relative to vertical axis.
            spine_tilt = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))
            angles["spine_tilt"] = float(spine_tilt)

            shoulder_line = right_shoulder - left_shoulder
            shoulder_rotation = np.degrees(
                np.arctan2(shoulder_line[1], shoulder_line[0] + 1e-6)
            )
            angles["shoulder_rotation"] = float(abs(shoulder_rotation))

        left_knee = self._angle_at_joint(
            kps,
            scs,
            self.KEYPOINTS["left_hip"],
            self.KEYPOINTS["left_knee"],
            self.KEYPOINTS["left_ankle"],
        )
        if left_knee is not None:
            angles["left_knee_flex"] = left_knee

        right_knee = self._angle_at_joint(
            kps,
            scs,
            self.KEYPOINTS["right_hip"],
            self.KEYPOINTS["right_knee"],
            self.KEYPOINTS["right_ankle"],
        )
        if right_knee is not None:
            angles["right_knee_flex"] = right_knee

        return angles

    def _angle_at_joint(
        self,
        kps: np.ndarray,
        scs: np.ndarray,
        p1: int,
        p2: int,
        p3: int,
    ) -> float | None:
        a = self._safe_point(kps, scs, p1)
        b = self._safe_point(kps, scs, p2)
        c = self._safe_point(kps, scs, p3)
        if a is None or b is None or c is None:
            return None

        v1 = a - b
        v2 = c - b
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_angle = np.dot(v1, v2) / denom
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return float(angle)

    def _safe_point(
        self, kps: np.ndarray, scs: np.ndarray, index: int
    ) -> np.ndarray | None:
        if index >= len(kps) or index >= len(scs):
            return None
        if float(scs[index]) < self.score_thr:
            return None
        return np.asarray(kps[index], dtype=np.float32)

    def _estimate_phase(self, kps: np.ndarray, scs: np.ndarray) -> tuple[str, float]:
        right_wrist = self._safe_point(kps, scs, self.KEYPOINTS["right_wrist"])
        left_wrist = self._safe_point(kps, scs, self.KEYPOINTS["left_wrist"])
        right_hip = self._safe_point(kps, scs, self.KEYPOINTS["right_hip"])
        left_hip = self._safe_point(kps, scs, self.KEYPOINTS["left_hip"])
        right_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["right_shoulder"])
        left_shoulder = self._safe_point(kps, scs, self.KEYPOINTS["left_shoulder"])

        if (
            right_hip is None
            or left_hip is None
            or right_shoulder is None
            or left_shoulder is None
            or (right_wrist is None and left_wrist is None)
        ):
            return self._phase, 0.1

        wrist = right_wrist if right_wrist is not None else left_wrist
        wrist_score = scs[self.KEYPOINTS["right_wrist"]]
        if right_wrist is None:
            wrist_score = scs[self.KEYPOINTS["left_wrist"]]

        mid_hip = (left_hip + right_hip) / 2.0
        torso_length = np.linalg.norm(((left_shoulder + right_shoulder) / 2.0) - mid_hip)
        torso_length = max(float(torso_length), 1.0)

        # Positive value means the wrist moved upward relative to hip.
        wrist_height = float((mid_hip[1] - wrist[1]) / torso_length)
        self._wrist_height_history.append(wrist_height)
        if len(self._wrist_height_history) < 5:
            return self._phase, 0.2

        velocity_window = np.diff(np.array(self._wrist_height_history, dtype=np.float32)[-4:])
        velocity = float(np.median(velocity_window))
        phase = self._classify_phase(wrist_height, velocity)
        phase = self._stabilize_phase(phase)

        confidence = min(1.0, max(0.1, float(wrist_score) + min(abs(velocity) * 2.0, 0.35)))
        return phase, confidence

    def _classify_phase(self, wrist_height: float, velocity: float) -> str:
        if wrist_height < 0.15 and abs(velocity) < 0.02:
            return "Address"
        if wrist_height < 0.85 and velocity > 0.025:
            return "Backswing"
        if wrist_height >= 0.85 and abs(velocity) <= 0.02:
            return "Top"
        if wrist_height > 0.25 and velocity < -0.03:
            return "Downswing"
        if wrist_height <= 0.25 and velocity < -0.01:
            return "Impact"
        if wrist_height > 0.3 and velocity > 0.02:
            return "Follow-through"
        return "Finish"

    def _stabilize_phase(self, phase: str) -> str:
        if phase == self._phase:
            self._candidate_phase = phase
            self._candidate_count = 0
            return self._phase

        if phase == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = phase
            self._candidate_count = 1

        if self._candidate_count >= 3:
            self._phase = phase
            self._candidate_count = 0

        return self._phase
