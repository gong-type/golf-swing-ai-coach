from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort

from src import GolfPoseAnalyzer, GolfVisualizer


def configure_windows_dll_paths() -> list[str]:
    if os.name != "nt":
        return []

    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if not venv_path:
        return []
    base = pathlib.Path(venv_path)

    site_packages = base / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return []

    added: list[str] = []
    for bin_dir in site_packages.glob("*/bin"):
        if not bin_dir.is_dir():
            continue
        path_str = str(bin_dir.resolve())
        try:
            os.add_dll_directory(path_str)
        except (AttributeError, FileNotFoundError, OSError):
            continue
        added.append(path_str)

    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
    return added


def probe_cuda_runtime(providers: set[str]) -> tuple[bool, str]:
    if "CUDAExecutionProvider" not in providers:
        return False, "未检测到 CUDAExecutionProvider"

    if os.name == "nt":
        try:
            ctypes.WinDLL("cudnn64_9.dll")
        except OSError:
            return False, "缺少 cudnn64_9.dll（需安装 cuDNN 9）"
    return True, ""


def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高尔夫挥杆摄像头实时 Demo")
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号，默认 0")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "lightweight", "balanced", "performance"],
        help="模型模式，auto 会按设备自动选择",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--mirror", action="store_true", help="镜像显示")
    parser.add_argument(
        "--infer-scale",
        type=float,
        default=0.0,
        help="推理分辨率缩放，范围 0.5~1.0，0 表示自动",
    )
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=0,
        help="检测频率，0 表示自动",
    )
    parser.add_argument(
        "--infer-interval",
        type=int,
        default=0,
        help="每 N 帧做一次推理，0 表示自动",
    )
    parser.add_argument("--score-thr", type=float, default=0.30, help="关键点阈值")
    parser.add_argument("--tracking", action="store_true", help="开启跟踪（默认关闭更快）")
    parser.add_argument("--record-dir", default="recordings", help="录制视频输出目录")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="自动退出帧数，0 表示手动退出",
    )
    return parser


def draw_record_button(
    frame: np.ndarray,
    recording: bool,
    elapsed_sec: float,
) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    btn_w = int(np.clip(w * 0.14, 110, 150))
    btn_h = int(np.clip(h * 0.07, 34, 46))
    x = 14
    y = max(10, h - btn_h - 14)

    bg = frame.copy()
    color = (32, 55, 180) if recording else (48, 140, 70)
    cv2.rectangle(bg, (x, y), (x + btn_w, y + btn_h), color, -1, cv2.LINE_AA)
    cv2.addWeighted(bg, 0.46, frame, 0.54, 0.0, dst=frame)
    cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), (230, 230, 230), 1, cv2.LINE_AA)

    label = "STOP" if recording else "REC"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    tx = x + (btn_w - tw) // 2
    ty = y + (btn_h + th) // 2 - 2
    cv2.putText(
        frame,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if recording:
        total = int(max(0, elapsed_sec))
        mm = total // 60
        ss = total % 60
        tip = f"REC {mm:02d}:{ss:02d}"
        info_x = x + btn_w + 12
        info_y = y + btn_h // 2 + 6
        cv2.circle(frame, (info_x, y + btn_h // 2), 5, (40, 40, 235), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            tip,
            (info_x + 12, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return x, y, btn_w, btn_h


def main() -> None:
    args = build_parser().parse_args()
    window_name = "高尔夫挥杆AI教练 - 实时Demo"

    cv2.setUseOptimized(True)
    added_dll_dirs = configure_windows_dll_paths()
    providers = set(ort.get_available_providers())
    cuda_ready, cuda_reason = probe_cuda_runtime(providers)

    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "cuda" if cuda_ready else "cpu"
    elif resolved_device == "cuda" and not cuda_ready:
        print(f"警告: CUDA 当前不可用（{cuda_reason}），已自动回退到 CPU。")
        resolved_device = "cpu"

    resolved_mode = args.mode
    if resolved_mode == "auto":
        resolved_mode = "balanced" if resolved_device == "cuda" else "lightweight"

    infer_scale = args.infer_scale
    if infer_scale <= 0:
        infer_scale = 0.85 if resolved_device == "cuda" else 0.60

    det_frequency = args.det_frequency
    if det_frequency <= 0:
        det_frequency = 5 if resolved_device == "cuda" else 10

    infer_interval = args.infer_interval
    if infer_interval <= 0:
        infer_interval = 1 if resolved_device == "cuda" else 2

    analyzer = GolfPoseAnalyzer(
        mode=resolved_mode,
        device=resolved_device,
        det_frequency=det_frequency,
        score_thr=args.score_thr,
        inference_scale=infer_scale,
        tracking=args.tracking,
    )
    visualizer = GolfVisualizer()

    cap = open_camera(args.camera)
    if cap is None:
        raise RuntimeError(
            f"无法打开摄像头 {args.camera}。请尝试 --camera 1，"
            "并检查 Windows 的相机权限设置。"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("摄像头 Demo 已启动")
    print("按键: q/ESC 退出, r 重置状态, h 显示/隐藏信息面板, v 开始/停止录制")
    print(
        f"Provider: {', '.join(sorted(providers))}"
    )
    if added_dll_dirs:
        print("已加载本地 CUDA/cuDNN 运行时目录。")
    if not cuda_ready:
        print(f"CUDA状态: 不可用（{cuda_reason}）")
    print(
        f"推理设备: {analyzer.device}, 模式: {resolved_mode}, "
        f"推理缩放: {infer_scale:.2f}, 检测频率: {det_frequency}, "
        f"推理间隔: {infer_interval}, 跟踪: {'开' if args.tracking else '关'}"
    )

    prev_time = time.perf_counter()
    fps_history = deque(maxlen=20)
    frame_count = 0
    last_analysis = None
    recording = False
    record_path: pathlib.Path | None = None
    recorder: cv2.VideoWriter | None = None
    record_started_at = 0.0
    mouse_state = {"toggle_requested": False, "button_rect": (0, 0, 0, 0)}

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx, by, bw, bh = mouse_state["button_rect"]
        if bw <= 0 or bh <= 0:
            return
        if bx <= x <= bx + bw and by <= y <= by + bh:
            mouse_state["toggle_requested"] = True

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头帧失败，程序结束。")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        if last_analysis is None or frame_count % infer_interval == 0:
            analysis = analyzer.analyze_frame(frame)
            last_analysis = analysis
        else:
            analysis = last_analysis
        frame_count += 1

        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps_history.append(1.0 / dt)
        fps = float(np.mean(fps_history))

        rendered = visualizer.draw(frame, analysis, fps=fps)
        if recording and recorder is not None:
            recorder.write(rendered)

        display = rendered
        try:
            _x, _y, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0 and (win_w != rendered.shape[1] or win_h != rendered.shape[0]):
                display = cv2.resize(rendered, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            pass

        elapsed = time.perf_counter() - record_started_at if recording else 0.0
        mouse_state["button_rect"] = draw_record_button(display, recording, elapsed)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if recording and recorder is not None:
                recorder.release()
                recorder = None
                recording = False
                if record_path is not None:
                    print(f"录制已保存: {record_path}")
            break
        if key == ord("r"):
            analyzer.reset()
            visualizer.reset()
        if key == ord("h"):
            is_visible = visualizer.toggle_hud()
            print(f"信息面板: {'显示' if is_visible else '隐藏'}")
        toggle_req = mouse_state["toggle_requested"] or key == ord("v")
        if mouse_state["toggle_requested"]:
            mouse_state["toggle_requested"] = False
        if toggle_req:
            if not recording:
                out_dir = pathlib.Path(args.record_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_path = out_dir / f"golf_demo_{stamp}.mp4"
                h, w = rendered.shape[:2]
                fps_out = float(np.clip(fps if fps > 1 else 30.0, 15.0, 60.0))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(record_path), fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    print("录制启动失败：无法创建视频文件。")
                else:
                    recorder = writer
                    recording = True
                    record_started_at = time.perf_counter()
                    print(f"开始录制: {record_path}")
            else:
                if recorder is not None:
                    recorder.release()
                recorder = None
                recording = False
                if record_path is not None:
                    print(f"录制已保存: {record_path}")
        if args.max_frames > 0 and frame_count >= args.max_frames:
            if recording and recorder is not None:
                recorder.release()
                recorder = None
                recording = False
                if record_path is not None:
                    print(f"录制已保存: {record_path}")
            break

    cap.release()
    if recorder is not None:
        recorder.release()
        if record_path is not None:
            print(f"录制已保存: {record_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
