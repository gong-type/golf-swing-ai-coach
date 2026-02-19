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

### 安装

```bash
# 克隆仓库
git clone https://github.com/gong-type/golf-swing-ai-coach.git
cd golf-swing-ai-coach

# 安装依赖
pip install rtmlib onnxruntime-gpu opencv-python
# CPU 用户替换为: pip install rtmlib onnxruntime opencv-python
```

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
├── src/
│   ├── pose_analyzer.py        # 姿态分析核心模块
│   ├── swing_detector.py       # 挥杆阶段检测
│   ├── angle_calculator.py     # 关节角度计算
│   ├── video_processor.py      # 视频处理与骨骼叠加
│   ├── visualizer.py           # 自定义可视化（高尔夫配色/标注）
│   └── utils.py                # 工具函数
├── gui/
│   ├── app.py                  # Gradio Web 界面
│   └── qt_app.py               # PyQt 桌面界面
├── models/                     # 模型权重文件
├── configs/                    # 配置文件
├── data/
│   └── sample_videos/          # 示例挥杆视频
├── docs/                       # 文档
│   └── images/                 # 效果截图
└── tests/                      # 单元测试
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
