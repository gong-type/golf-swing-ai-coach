from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pose_analyzer import FrameAnalysis


class GolfVisualizer:
    """Render coaching overlays on top of webcam frames."""

    SKELETON = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    UPPER_BODY_SEGMENTS = {(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12)}

    COLORS = {
        "upper": (255, 170, 30),
        "lower": (60, 210, 100),
        "spine": (60, 60, 255),
        "joint": (245, 245, 245),
        "hud_bg": (12, 20, 28),
        "hud_text": (230, 240, 255),
        "trajectory": (255, 230, 80),
    }

    def __init__(
        self,
        score_thr: float = 0.35,
        trail_size: int = 25,
        hud_refresh_interval: int = 2,
        hud_opacity: float = 0.55,
    ) -> None:
        self.score_thr = score_thr
        self.right_wrist_trail: Deque[tuple[int, int]] = deque(maxlen=trail_size)
        self.hud_refresh_interval = max(1, int(hud_refresh_interval))
        self.hud_opacity = float(np.clip(hud_opacity, 0.2, 0.95))
        self.hud_visible = True
        self._font_cache: dict[int, ImageFont.ImageFont] = {}
        self._hud_tick = 0
        self._cached_panel: np.ndarray | None = None
        self._hud_rect: tuple[int, int, int, int] | None = None
        self._font_path = self._find_font_path()

    def reset(self) -> None:
        self.right_wrist_trail.clear()
        self._hud_tick = 0
        self._cached_panel = None
        self._hud_rect = None

    def toggle_hud(self) -> bool:
        self.hud_visible = not self.hud_visible
        return self.hud_visible

    def draw(self, frame: np.ndarray, analysis: FrameAnalysis, fps: float) -> np.ndarray:
        canvas = frame.copy()

        if analysis.keypoints is not None and analysis.scores is not None:
            self._draw_skeleton(canvas, analysis.keypoints, analysis.scores)
            self._draw_trajectory(canvas, analysis.keypoints, analysis.scores)

        if self.hud_visible:
            self._draw_hud(canvas, analysis, fps)
        else:
            self._hud_rect = None
        self._draw_coach_tip(canvas, analysis)
        return canvas

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, scs: np.ndarray) -> None:
        for a, b in self.SKELETON:
            if float(scs[a]) < self.score_thr or float(scs[b]) < self.score_thr:
                continue
            p1 = tuple(np.int32(kps[a]))
            p2 = tuple(np.int32(kps[b]))
            segment = (a, b)
            color = self.COLORS["upper"] if segment in self.UPPER_BODY_SEGMENTS else self.COLORS["lower"]
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        for i, pt in enumerate(kps):
            if float(scs[i]) < self.score_thr:
                continue
            p = tuple(np.int32(pt))
            cv2.circle(frame, p, 4, self.COLORS["joint"], -1, cv2.LINE_AA)

        if all(float(scs[idx]) >= self.score_thr for idx in (5, 6, 11, 12)):
            mid_shoulder = np.mean(kps[[5, 6]], axis=0).astype(np.int32)
            mid_hip = np.mean(kps[[11, 12]], axis=0).astype(np.int32)
            cv2.line(
                frame,
                tuple(mid_shoulder),
                tuple(mid_hip),
                self.COLORS["spine"],
                3,
                cv2.LINE_AA,
            )

    def _draw_trajectory(self, frame: np.ndarray, kps: np.ndarray, scs: np.ndarray) -> None:
        right_wrist_idx = 10
        if float(scs[right_wrist_idx]) >= self.score_thr:
            wrist = tuple(np.int32(kps[right_wrist_idx]))
            self.right_wrist_trail.append(wrist)

        trail_points = list(self.right_wrist_trail)
        if len(trail_points) < 2:
            return

        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)
            color = (
                int(self.COLORS["trajectory"][0] * alpha),
                int(self.COLORS["trajectory"][1] * alpha),
                int(self.COLORS["trajectory"][2] * alpha),
            )
            cv2.line(frame, trail_points[i - 1], trail_points[i], color, 2, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, analysis: FrameAnalysis, fps: float) -> None:
        frame_h, frame_w = frame.shape[:2]
        font_size = int(np.clip(frame_h * 0.024, 13, 18))
        line_height = font_size + 5
        panel_w = int(np.clip(frame_w * 0.25, 220, 320))
        self._hud_tick += 1
        should_refresh = (
            self._cached_panel is None or self._hud_tick % self.hud_refresh_interval == 0
        )

        if should_refresh:
            device_text = "GPU" if analysis.device.lower() == "cuda" else "CPU"
            lines: list[str] = [
                f"阶段: {analysis.phase}",
                f"阶段置信度: {analysis.phase_confidence:.2f}",
                f"推理设备: {device_text}",
                f"实时帧率: {fps:.1f} FPS",
            ]

            labels = {
                "spine_tilt": "脊柱前倾角",
                "shoulder_rotation": "肩线旋转角",
                "left_knee_flex": "左膝角",
                "right_knee_flex": "右膝角",
            }
            for key in ("spine_tilt", "shoulder_rotation", "left_knee_flex", "right_knee_flex"):
                if key in analysis.angles:
                    lines.append(f"{labels[key]}: {analysis.angles[key]:.1f}°")

            hints = self._build_hints(analysis, fps)
            lines.extend(hints)
            panel_h = int(np.clip(14 + line_height * len(lines), 120, frame_h * 0.55))
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = self.COLORS["hud_bg"]
            cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (90, 120, 150), 1)
            self._draw_multiline_text_on_panel(
                panel,
                lines,
                x=10,
                y=10,
                line_height=line_height,
                font_size=font_size,
            )
            self._cached_panel = panel

        panel_h, panel_w = self._cached_panel.shape[:2]
        x = max(8, frame_w - panel_w - 12)
        y = 12
        self._hud_rect = (x, y, panel_w, panel_h)
        overlay = frame.copy()
        overlay[y : y + panel_h, x : x + panel_w] = self._cached_panel
        cv2.addWeighted(
            overlay,
            self.hud_opacity,
            frame,
            1.0 - self.hud_opacity,
            0,
            dst=frame,
        )

    @staticmethod
    def _build_hints(analysis: FrameAnalysis, fps: float) -> list[str]:
        hints: list[str] = []
        if analysis.keypoints is None:
            hints.append("提示: 请完整入镜并确保环境光线充足")
        elif analysis.phase_confidence < 0.25:
            hints.append("提示: 请让手腕与髋部保持可见")

        if fps < 15:
            hints.append("提示: 可使用 --mode lightweight 提升流畅度")
        return hints

    def _draw_coach_tip(self, frame: np.ndarray, analysis: FrameAnalysis) -> None:
        if not analysis.coach_tip:
            return

        frame_h, frame_w = frame.shape[:2]
        font_size = int(np.clip(frame_h * 0.032, 16, 28))
        font = self._get_font(font_size)

        # Measure text box with PIL for accurate Chinese layout.
        probe = Image.new("RGB", (10, 10))
        probe_draw = ImageDraw.Draw(probe)
        bbox = probe_draw.textbbox((0, 0), analysis.coach_tip, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])

        panel_w = min(int(frame_w * 0.82), text_w + 44)
        panel_h = text_h + 20
        x = max(0, (frame_w - panel_w) // 2)
        y = max(8, int(frame_h * 0.06))

        if self._hud_rect is not None:
            hud_x, hud_y, hud_w, hud_h = self._hud_rect
            if self._rect_overlaps((x, y, panel_w, panel_h), (hud_x, hud_y, hud_w, hud_h)):
                y = min(frame_h - panel_h - 8, hud_y + hud_h + 10)
                if self._rect_overlaps((x, y, panel_w, panel_h), (hud_x, hud_y, hud_w, hud_h)):
                    x = max(8, hud_x - panel_w - 12)
                    x = min(x, frame_w - panel_w - 8)

        roi = frame[y : y + panel_h, x : x + panel_w]
        if roi.size == 0:
            return

        bg_color = (80, 145, 70)  # good
        if analysis.coach_level == "warn":
            bg_color = (42, 155, 235)
        elif analysis.coach_level == "info":
            bg_color = (160, 120, 40)

        overlay = roi.copy()
        overlay[:] = bg_color
        cv2.addWeighted(overlay, 0.42, roi, 0.58, 0.0, dst=roi)
        cv2.rectangle(roi, (0, 0), (panel_w - 1, panel_h - 1), bg_color, 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        text_x = (panel_w - text_w) // 2
        text_y = max(2, (panel_h - text_h) // 2 - 1)
        draw.text((text_x, text_y), analysis.coach_tip, fill=(255, 255, 255), font=font)
        roi[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _rect_overlaps(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int]
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    def _draw_multiline_text_on_panel(
        self,
        panel: np.ndarray,
        lines: list[str],
        x: int,
        y: int,
        line_height: int,
        font_size: int,
    ) -> None:
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        font = self._get_font(font_size)
        for idx, line in enumerate(lines):
            draw.text(
                (x, y + idx * line_height),
                line,
                fill=(230, 240, 255),
                font=font,
            )
        panel[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _find_font_path() -> Path | None:
        font_candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        for font_path in font_candidates:
            if font_path.exists():
                return font_path
        return None

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        size = int(np.clip(size, 12, 24))
        if size in self._font_cache:
            return self._font_cache[size]

        if self._font_path is not None:
            font = ImageFont.truetype(str(self._font_path), size=size)
        else:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font
