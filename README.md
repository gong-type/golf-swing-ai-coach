# 🏌️ 高尔夫挥杆动作 AI 智能分析系统

> 基于姿态估计的人体骨骼关键点检测、骨骼连线可视化与挥杆阶段智能分析

## 📖 项目简介

本项目为高尔夫模拟器开发一套 **AI 教练** 功能模块，实现以下核心能力：

- **人体骨骼关键点检测**：自动识别球员身体的关键关节点（肩膀、手肘、手腕、脊柱、髋、膝盖、脚踝等）
- **骨骼连线绘制**：将检测到的关节点按人体结构正确连接，形成完整的骨骼姿态线
- **击球回放视频叠加**：将关节点标记和连线实时叠加在击球回放视频上
- **挥杆阶段检测**：自动识别 8 个挥杆关键阶段（准备 → 上杆 → 顶杆 → 下杆 → 击球 → 送杆 → 收杆）
- **教学分析指标**：计算脊柱倾斜角、髋部旋转角、手臂挥杆平面等生物力学指标

## 🎯 应用场景

学员在模拟器中打完球后，观看回放时画面上直接显示：
1. 身体骨骼线和动作轨迹
2. 各挥杆阶段自动标注
3. 关键角度数值
4. 与标准动作的对比偏差

一眼就能看出动作哪里不标准。

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 姿态估计引擎 | [rtmlib](https://github.com/Tau-J/rtmlib) | 超轻量封装，无 mmcv 依赖，支持 RTMPose / ViTPose++ / RTMW |
| 骨骼模型 | RTMPose-l (17点) / RTMW (133点) | 17点用于实时回放，133点用于精细握杆分析 |
| 3D 姿态 | RTMW3D | 3D全身关键点估计，可旋转查看挥杆平面 |
| 挥杆阶段检测 | [GolfDB](https://github.com/wmcnally/golfdb) SwingNet | 8 阶段自动识别，CVPR Workshop 论文 |
| 人体检测器 | YOLOX-m | 目标检测，定位画面中的球员 |
| 视频处理 | OpenCV | 视频读取、帧处理、骨骼叠加绘制 |
| 推理后端 | ONNXRuntime / TensorRT | ONNX 通用推理 / TensorRT GPU 加速 |
| GUI 展示 | Gradio / PyQt | Demo 演示界面 |

## 📊 性能指标

| 模型 | 关键点数 | 精度 (AP) | GPU 速度 | 适用场景 |
|------|---------|----------|---------|---------|
| RTMPose-m | 17 | 75.8 | 430+ FPS | 实时回放叠加 |
| RTMPose-l | 17 | 78.3 | 160+ FPS | 高精度分析 |
| RTMW-l | 133 | 70.1 | — | 全身+手指分析 |
| ViTPose++-l | 17 | 78.6 | — | 最高精度离线分析 |

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- NVIDIA GPU（推荐，CPU 也可运行）

### 安装（推荐 uv）

```bash
# 克隆仓库
git clone https://github.com/gong-type/golf-swing-ai-coach.git
cd golf-swing-ai-coach

# 创建虚拟环境并安装依赖
uv venv
uv sync
```

默认依赖已包含 `onnxruntime-gpu`。

### Windows 摄像头实时 Demo

```bash
# 推荐（自动选设备/模式，适合大多数机器）
uv run python demo_webcam.py --camera 0 --mode auto --device auto --mirror

# 强制用 GPU（需 onnxruntime-gpu + NVIDIA CUDA 环境可用）
uv run python demo_webcam.py --camera 0 --mode balanced --device cuda --mirror --infer-scale 0.85 --infer-interval 1

# 低配 CPU 流畅参数（优先不卡）
uv run python demo_webcam.py --camera 0 --mode lightweight --device cpu --infer-scale 0.60 --infer-interval 2 --det-frequency 10

# 快速自检（跑 120 帧后自动退出）
uv run python demo_webcam.py --camera 0 --mode auto --max-frames 120

# 查看 ONNX Runtime provider（排查 GPU）
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 自定义录制输出目录
uv run python demo_webcam.py --camera 0 --mode auto --record-dir outputs/records
```

按键说明：
- `q` 或 `ESC`：退出
- `r`：重置阶段和轨迹状态
- `h`：显示/隐藏 HUD 信息面板
- `v`：开始/停止录制视频（也可点击左下角 `REC/STOP` 按钮）

界面说明：
- HUD 信息和提示已改为中文（默认右上角半透明）
- 当入镜不完整或帧率偏低时，会显示中文提醒
- 新增 AI 教练提示条：根据实时动作在画面上方居中提示（如“手部需要更高”“完美！”）
- 窗口放大后画面会自动铺满窗口，避免只显示在左上角
- 左下角提供录制按钮，录制文件默认保存到 `recordings/`

性能说明：
- 首次运行会下载模型，速度会明显慢，下载完成后会恢复正常
- 启动日志会打印 `Provider` 列表，包含 `CUDAExecutionProvider` 才表示已启用 GPU 推理
- 若出现 `CUDA状态: 不可用（缺少 cudnn64_9.dll）`，请安装 cuDNN 9 并把 DLL 加入系统 `PATH`
- 低配机器优先使用 `--mode lightweight --infer-interval 2 --infer-scale 0.60`
- 默认分辨率为 `960x540`（更流畅）；需要更清晰可手动加 `--width 1280 --height 720`

### 基础使用

```python
import cv2
from rtmlib import PoseTracker, Body, draw_skeleton

# 初始化姿态追踪器
pose_tracker = PoseTracker(
    Body,
    mode='performance',
    det_frequency=5,
    backend='onnxruntime',
    device='cuda'  # 或 'cpu'
)

# 处理挥杆视频
cap = cv2.VideoCapture('golf_swing.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, scores = pose_tracker(frame)
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

    cv2.imshow('Golf Swing Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 全身 133 关键点分析（含手指/握杆）

```python
from rtmlib import Wholebody, draw_skeleton

wholebody = Wholebody(
    mode='performance',
    backend='onnxruntime',
    device='cuda'
)

keypoints, scores = wholebody(frame)
frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
```

## 📁 项目结构

```
golf-swing-ai-coach/
├── README.md                   # 项目说明
├── PLAN.md                     # 项目计划书（实施方案）
├── requirements.txt            # Python 依赖
├── pyproject.toml              # uv 项目配置
├── uv.lock                     # uv 锁文件
├── demo_webcam.py              # Windows 摄像头实时 Demo
├── src/
│   ├── __init__.py
│   ├── pose_analyzer.py        # 姿态分析 + 角度计算 + 简化阶段识别
│   └── visualizer.py           # 高尔夫风格骨骼与 HUD 绘制
└── .venv/                      # 本地虚拟环境（运行后生成）
```

## 🔗 参考资源

- [RTMPose 论文](https://arxiv.org/abs/2303.07399) — 实时多人姿态估计
- [GolfDB 论文](https://arxiv.org/abs/1903.06528) — 高尔夫挥杆视频数据库
- [Sapiens](https://github.com/facebookresearch/sapiens) — Meta 人体视觉基础模型 (ECCV 2024)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) — Vision Transformer 姿态估计
- [rtmlib](https://github.com/Tau-J/rtmlib) — 超轻量姿态估计推理库

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。
