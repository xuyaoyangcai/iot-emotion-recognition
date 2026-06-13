# 🏫 课堂状态分析系统

物联网课程期末大作业 — 基于表情识别 + 头姿态估计的课堂状态实时分析系统。

## 功能

- **多输入源**：图片 / 视频 / 实时摄像头（WebRTC 流畅推流）
- **双模式 UI**：仪表盘模式（统计分析）+ 魔镜模式（emoji + 滤镜互动）
- **7 种基础表情**：Happy, Neutral, Sad, Angry, Surprise, Fear, Disgust（GPU 批量推理）
- **6 种课堂复合情绪**：Focused, Distracted, Engaged, Confused, Thinking, Tired（表情 + yaw/pitch 融合）
- **头姿态估计**：YuNet 5 点关键点 + solvePnP → pitch/yaw/roll，含抬头率统计
- **课堂状态分类**：良好 / 平稳 / 较低落 / 注意力波动 / 未检测到学生（含低头率阈值）
- **1fps 视频采样**：按视频帧率自适应采样，避免重复处理
- **5 帧滑动窗口**：窗口内表情均值/方差统计，状态突变检测
- **三级预警追踪**：Green / Yellow / Red，连续帧状态累积触发
- **Plotly 时序图**：表情占比折线 + 抬头率曲线 + 预警区域标记
- **CSV 导出**：课堂状态 CSV + 时序分析 CSV

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

首次运行自动加载 HuggingFace 模型（~300MB）和 YuNet ONNX 模型。国内用户：

```bash
export HF_ENDPOINT=https://hf-mirror.com
streamlit run app.py
```

## 项目结构

```
app.py                    # Streamlit 主界面
face_detector.py          # 人脸检测（YuNet ONNX）+ 头姿态估计（solvePnP）
expression_recognizer.py  # 表情识别（HuggingFace ViT，GPU 批量 TTA）
gaze_emotion.py           # 课堂复合情绪（表情 + yaw/pitch 融合推导）
classroom_state.py        # 课堂状态分类 + 滑动窗口 + 预警追踪
analyzer.py               # 统计分析 + CSV 导出
utils.py                  # 画框 / 滤镜 / emoji / 颜色映射
tests/                    # 单元测试
test_images/              # 测试图片
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 人脸检测 | OpenCV YuNet DNN (ONNX) |
| 关键点 & 头姿态 | YuNet 5点 + solvePnP (EPNP) |
| 表情识别 | HuggingFace ViT (dima806/facial_emotions_image_detection) |
| GPU 推理 | PyTorch 2.6 + CUDA 12.4 (RTX 4060) |
| 摄像头 | streamlit-webrtc (WebRTC 异步推流) |
| UI | Streamlit 1.58 + @st.fragment |
| 图表 | Plotly |
| 图像处理 | OpenCV + Pillow |

## 运行测试

```bash
python -m pytest tests/ -v
```
