# 🏫 课堂状态分析系统

物联网课程期末大作业 — 基于多人人脸检测与表情识别的课堂状态分析系统。

## 功能

- **三输入源**：图片 / 视频 / 实时摄像头（WebRTC 流畅推流）
- **8 类基础表情**：Happy, Neutral, Sad, Angry, Surprise, Fear, Disgust, Contempt（三模型集成 + TTA）
- **8 类学生状态**：Focused, Engaged, Thinking, Tired, Bored, Distracted, Confused, Anxious（基础表情 × 头姿态融合推导）
- **头姿态估计**：YuNet 5 点关键点 + solvePnP → pitch/yaw → 抬头/正常/低头，含抬头率统计
- **课堂状态分类**：良好 / 平稳 / 较低落 / 注意力波动（按优先级规则匹配）
- **视频流畅播放**：每帧轻量人脸检测保证流畅，定时完整表情分析，非分析帧复用缓存结果
- **5 帧滑动窗口**：窗口内表情均值/方差统计，状态突变检测
- **双层预警**：滑动窗口预警 + 连续帧追踪预警（Green / Yellow / Red / Normal）
- **Plotly 时序图**：表情占比折线 + 抬头率曲线 + 预警区域标记
- **CSV 导出**：课堂状态 CSV + 时序分析 CSV
- **系统简介面板**：侧边栏开关，右侧展示表情/状态/规则/预警完整说明

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

首次运行自动加载 HuggingFace 模型（~300MB，已缓存本地）和 YuNet ONNX 模型。国内用户：

```bash
set HF_ENDPOINT=https://hf-mirror.com
streamlit run app.py
```

## 项目结构

```
app.py                    # Streamlit 主界面（图片/视频/摄像头三种模式）
face_detector.py          # 人脸检测（YuNet ONNX）+ 头姿态估计（solvePnP）
expression_recognizer.py  # 表情识别（三模型集成：dima806 + mo-thecreator + HardlyHumans）
gaze_emotion.py           # 课堂复合状态推导（基础表情 × 头姿态 → 8类学生状态）
classroom_state.py        # 课堂状态分类 + 滑动窗口 + 预警追踪器
analyzer.py               # 统计分析 + CSV 导出
utils.py                  # 画框 / 滤镜 / emoji / 颜色映射
tests/                    # 单元测试
test_images/              # 测试图片
test_videos/              # 测试视频
```

## 核心模型

| 组件 | 技术 |
|------|------|
| 人脸检测 + 关键点 | OpenCV YuNet DNN (ONNX) |
| 头姿态估计 | solvePnP (EPNP) + 3D 标准人脸模型 |
| 表情识别 | HuggingFace × 3: dima806 (CNN) + mo-thecreator (ViT) + HardlyHumans (ViT) |
| 推理加速 | PyTorch GPU + TTA + 本地缓存 |
| 实时摄像头 | streamlit-webrtc (WebRTC 异步推流) |
| UI | Streamlit 1.58 + @st.fragment |
| 图表 | Plotly |
| 图像处理 | OpenCV + Pillow |

## 运行测试

```bash
python -m pytest tests/ -v
```
