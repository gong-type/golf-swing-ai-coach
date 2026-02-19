# 📋 项目计划书：高尔夫挥杆动作 AI 智能分析系统

## 一、项目概述

### 1.1 项目背景

公司开发高尔夫模拟游戏，上司要求在模拟器中加入 **AI 教练** 功能：
- 在击球回放视频上自动标记人体关节点并连线
- 绘制完整的骨骼姿态线，展示挥杆时的身体角度和姿势
- 关节点标记并连接正确，才有教学意义

### 1.2 项目目标

开发一套可集成到现有模拟器中的挥杆分析模块，实现：

1. **骨骼检测与绘制**：在回放视频上实时叠加人体骨骼线
2. **挥杆阶段识别**：自动识别 8 个挥杆阶段并标注
3. **角度分析**：计算教学关键指标（脊柱角度、髋部旋转、手臂平面等）
4. **可视化输出**：生成带标注的分析视频，支持多视角

### 1.3 核心技术选型

**rtmlib** + **GolfDB** + **OpenCV** 方案

选择理由：
- rtmlib 仅依赖 numpy + opencv + onnxruntime，**安装零门槛**
- 内置 RTMPose / ViTPose++ / RTMW / RTMW3D 全系列模型
- RTMPose-m 在 GPU 上可达 **430+ FPS**，远超实时需求
- 133 关键点模型覆盖手指关节，可分析握杆姿势
- Windows 完美兼容，与模拟器开发环境一致

---

## 二、实施计划（4 周）

### 第一周：基础骨骼检测与视频叠加

#### 第 1 天：环境搭建与验证

**任务清单：**
- [ ] 创建 Python 虚拟环境（Python 3.10+）
- [ ] 安装核心依赖：`pip install rtmlib onnxruntime-gpu opencv-python`
- [ ] 使用 rtmlib 官方 demo 跑通单图姿态检测
- [ ] 使用 PoseTracker 跑通视频姿态追踪
- [ ] 确认 GPU 推理正常（CUDA 可用）

**验收标准：** 能对任意人物视频输出带骨骼线的视频

#### 第 2-3 天：高尔夫专用骨骼可视化

**任务清单：**
- [ ] 定义高尔夫教学关注的关键骨骼连接
  - 躯干线：左肩-右肩、左髋-右髋、肩中点-髋中点（脊柱线）
  - 双臂线：肩-肘-腕
  - 双腿线：髋-膝-踝
- [ ] 自定义绘制样式
  - 关节点：圆圈标记，不同部位不同颜色
  - 骨骼线：加粗线条，教学配色方案
  - 脊柱线：特殊高亮显示
- [ ] 实现轨迹叠加功能
  - 手腕运动轨迹线（最近 N 帧连线）
  - 髋部中心轨迹线
- [ ] 添加关键角度实时显示
  - 脊柱前倾角
  - 双膝弯曲角
  - 手臂伸展角

**验收标准：** 输入高尔夫挥杆视频，输出专业教学风格的骨骼分析视频

**代码实现要点：**

```python
# src/pose_analyzer.py
import cv2
import numpy as np
from rtmlib import PoseTracker, Body, Wholebody, draw_skeleton

class GolfPoseAnalyzer:
    """高尔夫姿态分析器"""

    # COCO 17 关键点索引
    KEYPOINTS = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2,
        'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }

    # 高尔夫教学重点骨骼连接
    GOLF_SKELETON = [
        # 躯干
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        # 左臂
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        # 右臂
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        # 左腿
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        # 右腿
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        # 躯干连接
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
    ]

    def __init__(self, mode='performance', device='cuda'):
        self.pose_tracker = PoseTracker(
            Body,
            mode=mode,
            det_frequency=5,
            backend='onnxruntime',
            device=device
        )
        self.trajectory = {'left_wrist': [], 'right_wrist': [], 'hip_center': []}

    def analyze_frame(self, frame):
        """分析单帧，返回关键点和角度"""
        keypoints, scores = self.pose_tracker(frame)
        angles = self._calculate_angles(keypoints, scores)
        return keypoints, scores, angles

    def _calculate_angles(self, keypoints, scores):
        """计算教学关键角度"""
        angles = {}
        if keypoints is None or len(keypoints) == 0:
            return angles

        kps = keypoints[0]  # 取第一个人
        
        # 脊柱前倾角
        mid_shoulder = (kps[5] + kps[6]) / 2
        mid_hip = (kps[11] + kps[12]) / 2
        spine_angle = np.degrees(np.arctan2(
            mid_shoulder[0] - mid_hip[0],
            mid_hip[1] - mid_shoulder[1]
        ))
        angles['spine_tilt'] = abs(spine_angle)

        # 左膝弯曲角
        angles['left_knee'] = self._angle_between(kps[11], kps[13], kps[15])
        # 右膝弯曲角
        angles['right_knee'] = self._angle_between(kps[12], kps[14], kps[16])

        return angles

    @staticmethod
    def _angle_between(p1, p2, p3):
        """计算三点之间的角度"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
```

#### 第 4-5 天：侧面/正面双视角支持

**任务清单：**
- [ ] 实现 Down the Line（侧面）视角分析
  - 挥杆平面可视化（手腕轨迹拟合平面）
  - 脊柱角度测量
  - 头部稳定性检测
- [ ] 实现 Face On（正面）视角分析
  - 髋部旋转幅度
  - 重心横向偏移
  - 双肩旋转角度

### 第二周：挥杆阶段检测与分析

#### 第 6-7 天：集成 GolfDB SwingNet

**任务清单：**
- [ ] 下载 GolfDB 预训练模型 swingnet_1800.pth.tar
- [ ] 适配 SwingNet 推理接口
- [ ] 实现 8 阶段自动检测：
  1. Address（准备站位）
  2. Toe-up（脚尖朝上 / 上杆初期）
  3. Mid-backswing（上杆中段）
  4. Top（顶点）
  5. Mid-downswing（下杆中段）
  6. Impact（击球瞬间）
  7. Mid-follow-through（送杆中段）
  8. Finish（收杆完成）
- [ ] 在视频上标注当前阶段名称

**代码实现要点：**

```python
# src/swing_detector.py
import torch
from model import EventDetector  # GolfDB 模型

class SwingPhaseDetector:
    """挥杆阶段检测器"""

    PHASES = [
        'Address', 'Toe-up', 'Mid-backswing', 'Top',
        'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish'
    ]

    PHASES_CN = [
        '准备站位', '上杆初期', '上杆中段', '顶杆',
        '下杆中段', '击球', '送杆中段', '收杆'
    ]

    def __init__(self, model_path='models/swingnet_1800.pth.tar', device='cuda'):
        self.device = device
        self.model = EventDetector(
            pretrain=True,
            width_mult=1.,
            lstm_layers=1,
            lstm_hidden=256,
            bidirectional=True,
            dropout=False
        )
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def detect_phases(self, video_frames):
        """输入视频帧序列，返回各阶段对应的帧索引"""
        # 预处理 + 推理逻辑
        pass
```

#### 第 8-10 天：教学分析指标系统

**任务清单：**
- [ ] 基于挥杆阶段的角度变化分析
  - Address 阶段：记录初始脊柱角度、膝盖弯曲度
  - Top 阶段：测量肩部旋转角（应达到 90°）、髋部旋转角（应达到 45°）
  - Impact 阶段：检测头部是否保持稳定、重心转移
  - Finish 阶段：检测重心是否完全转移到前脚
- [ ] 实现各阶段角度对比功能
- [ ] 生成分析报告数据结构

```python
# src/angle_calculator.py

class SwingAnalysisReport:
    """挥杆分析报告"""

    # 标准值参考（职业球员平均值）
    STANDARDS = {
        'address_spine_tilt': 35,       # 准备时脊柱前倾角 (度)
        'address_knee_flex': 155,        # 准备时膝盖弯曲度 (度)
        'top_shoulder_rotation': 90,     # 顶杆肩部旋转 (度)
        'top_hip_rotation': 45,          # 顶杆髋部旋转 (度)
        'impact_spine_tilt': 38,         # 击球时脊柱前倾角 (度)
        'finish_weight_front': 0.85,     # 收杆前脚承重比例
    }

    def __init__(self):
        self.phase_angles = {}  # 各阶段角度数据
        self.deviations = {}    # 与标准值的偏差

    def add_phase_data(self, phase_name, angles):
        """记录某个阶段的角度数据"""
        self.phase_angles[phase_name] = angles

    def analyze(self):
        """生成分析结论"""
        results = []
        if 'top' in self.phase_angles:
            top = self.phase_angles['top']
            if top.get('shoulder_rotation', 0) < 80:
                results.append({
                    'level': 'warning',
                    'message': '上杆不充分：肩部旋转不足，建议加大上杆幅度'
                })
        return results
```

### 第三周：可视化与 GUI

#### 第 11-12 天：专业级可视化渲染

**任务清单：**
- [ ] 高尔夫教学配色方案
  - 上半身骨骼：蓝色系
  - 下半身骨骼：绿色系
  - 脊柱线：红色高亮
  - 关节点：白色圆圈 + 彩色边框
- [ ] 信息面板 HUD
  - 左上角：当前挥杆阶段
  - 右上角：关键角度数值
  - 底部：挥杆进度条（8 阶段时间轴）
- [ ] 轨迹淡出效果
  - 手腕轨迹线透明度随时间衰减
  - 最近 30 帧轨迹线渐变色

```python
# src/visualizer.py

class GolfVisualizer:
    """高尔夫分析可视化器"""

    # 教学配色
    COLORS = {
        'skeleton_upper': (255, 165, 0),    # 上半身 - 橙色
        'skeleton_lower': (0, 200, 100),    # 下半身 - 绿色
        'spine': (0, 0, 255),               # 脊柱 - 红色
        'trajectory': (255, 255, 0),        # 轨迹 - 黄色
        'joint': (255, 255, 255),           # 关节 - 白色
        'text_bg': (0, 0, 0),              # 文字背景 - 黑色
    }

    def draw_golf_skeleton(self, frame, keypoints, scores, angles=None, phase=None):
        """绘制高尔夫教学骨骼图"""
        # 绘制骨骼连接线
        # 绘制关节点
        # 绘制角度标注
        # 绘制阶段信息
        # 绘制轨迹线
        pass

    def draw_hud(self, frame, phase, angles, progress):
        """绘制信息面板"""
        pass

    def draw_trajectory(self, frame, trajectory_points, max_length=30):
        """绘制运动轨迹（带淡出）"""
        pass
```

#### 第 13-15 天：GUI Demo 界面

**任务清单：**
- [ ] Gradio Web 界面（快速 Demo）
  - 视频上传
  - 实时分析预览
  - 分析报告展示
  - 导出带标注视频
- [ ] 参数调节面板
  - 模型选择（17点 / 133点）
  - 检测阈值
  - 可视化选项开关
  - 显示/隐藏轨迹线

```python
# gui/app.py
import gradio as gr
from src.pose_analyzer import GolfPoseAnalyzer
from src.swing_detector import SwingPhaseDetector
from src.visualizer import GolfVisualizer

def analyze_video(video_path, model_type, show_trajectory, show_angles):
    """分析上传的挥杆视频"""
    analyzer = GolfPoseAnalyzer(mode=model_type)
    # 处理视频...
    return output_video_path, report_text

demo = gr.Interface(
    fn=analyze_video,
    inputs=[
        gr.Video(label="上传挥杆视频"),
        gr.Radio(["balanced", "performance"], label="模型精度", value="balanced"),
        gr.Checkbox(label="显示运动轨迹", value=True),
        gr.Checkbox(label="显示角度数值", value=True),
    ],
    outputs=[
        gr.Video(label="分析结果"),
        gr.Textbox(label="分析报告", lines=10),
    ],
    title="🏌️ 高尔夫挥杆 AI 分析",
    description="上传挥杆视频，AI 自动分析骨骼姿态和动作阶段",
)
```

### 第四周：优化与集成

#### 第 16-17 天：性能优化

**任务清单：**
- [ ] TensorRT 加速部署（如有 NVIDIA GPU）
- [ ] 模型裁剪：评估 RTMPose-s 是否满足精度需求（更快）
- [ ] 视频处理管线优化
  - 多线程解码
  - 批量推理
  - 异步写入

#### 第 18-19 天：133关键点握杆分析（进阶）

**任务清单：**
- [ ] 使用 Wholebody 检测手部 21 个关键点
- [ ] 分析握杆位置和手型
- [ ] 检测握杆压力分布（基于手指弯曲角度推断）

#### 第 20 天：模拟器集成方案

**任务清单：**
- [ ] 定义与模拟器的数据接口
  - 输入：回放视频帧流 / 视频文件路径
  - 输出：带标注的视频帧流 / JSON 分析数据
- [ ] 编写集成文档
- [ ] 整理 API 接口供模拟器调用

```python
# 集成接口示例
class GolfAnalysisAPI:
    """供模拟器调用的分析接口"""

    def __init__(self, config=None):
        self.analyzer = GolfPoseAnalyzer()
        self.swing_detector = SwingPhaseDetector()
        self.visualizer = GolfVisualizer()

    def process_frame(self, frame):
        """处理单帧，返回标注后的帧和分析数据"""
        keypoints, scores, angles = self.analyzer.analyze_frame(frame)
        annotated = self.visualizer.draw_golf_skeleton(
            frame, keypoints, scores, angles
        )
        return annotated, {
            'keypoints': keypoints.tolist() if keypoints is not None else None,
            'angles': angles,
        }

    def process_video(self, video_path, output_path=None):
        """处理完整视频，返回分析报告"""
        pass
```

---

## 三、里程碑与交付物

| 里程碑 | 时间 | 交付物 | 演示重点 |
|--------|------|--------|---------|
| M1：基础 Demo | 第 1 周末 | 带骨骼线的分析视频 | "能检测身体、连线正确" |
| M2：阶段检测 | 第 2 周末 | 挥杆 8 阶段自动标注 | "知道球员在做什么" |
| M3：GUI Demo | 第 3 周末 | 可交互的 Web 演示界面 | "给上司看的完整 Demo" |
| M4：集成就绪 | 第 4 周末 | API 接口和集成文档 | "可以接入模拟器了" |

---

## 四、风险评估与应对

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| GPU 环境不可用 | 推理速度慢 | 低 | rtmlib 支持纯 CPU 推理，RTMPose-m CPU 可达 90 FPS |
| 挥杆视频角度特殊导致检测不准 | 关节定位偏差 | 中 | 使用 performance 模式（最大模型），必要时用 133 点全身模型 |
| GolfDB 模型与实际视频不匹配 | 阶段检测不准 | 中 | 先只做骨骼检测（M1），阶段检测作为增强功能 |
| 球杆检测需求 | 现有模型不检测球杆 | 高 | 第一阶段先不做杆头追踪，后期参考 3dGolfPoseEstimation 项目扩展 |
| 模拟器集成接口不清晰 | 对接困难 | 中 | 先做独立工具，定义清晰的 API 边界，降低耦合 |

---

## 五、技术储备（后续迭代方向）

### 5.1 球杆追踪

当前方案暂不包含球杆检测。后续可参考：
- [3dGolfPoseEstimation](https://github.com/Molten-Ice/3dGolfPoseEstimation)：自定义了 grip 和 club head 关键点
- 基于目标检测训练球杆专用检测模型

### 5.2 3D 姿态分析

rtmlib 已内置 RTMW3D 模型，可输出 3D 关键点坐标：
- 3D 挥杆平面可视化
- 旋转角度精确计算
- 多角度自由旋转查看

### 5.3 高尔夫专用模型微调

使用 [Sapiens](https://github.com/facebookresearch/sapiens)（Meta, ECCV 2024）的预训练权重，在高尔夫数据上微调：
- 提升遮挡场景（手臂交叉）的检测精度
- 增加球杆关键点
- 适配高尔夫特有姿势

### 5.4 动作对比功能

- 学员动作 vs 职业球员标准动作叠加对比
- 关键帧自动对齐
- 偏差热力图生成

---

## 六、依赖清单

```
# requirements.txt
rtmlib>=0.0.15
onnxruntime-gpu>=1.16.0    # GPU 推理（CPU 用户用 onnxruntime）
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0               # GolfDB SwingNet 依赖
gradio>=4.0.0              # Web GUI
```

---

## 七、参考资料

1. **RTMPose** — Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose", arXiv 2303.07399
2. **GolfDB** — McNally et al., "GolfDB: A Video Database for Golf Swing Sequencing", CVPR Workshop 2019
3. **ViTPose** — Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation", NeurIPS 2022
4. **Sapiens** — Khirodkar et al., "Sapiens: Foundation for Human Vision Models", ECCV 2024
5. **rtmlib** — https://github.com/Tau-J/rtmlib — 超轻量 RTMPose 推理库
