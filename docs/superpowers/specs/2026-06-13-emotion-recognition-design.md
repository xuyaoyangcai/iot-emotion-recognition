# 人脸表情识别系统 — 设计文档

## 概述

物联网课程期末大作业。基于 Python 的人脸表情识别系统，支持图片/视频上传和实时摄像头采集，提供双模式 UI（魔镜互动 + IoT 仪表盘），使用预训练模型进行人脸检测和表情分类。

## 技术选型

| 组件 | 技术 | 原因 |
|------|------|------|
| 人脸检测 | MediaPipe Face Detection | 快速、鲁棒、CPU 友好 |
| 表情识别 | HuggingFace ViT 预训练模型 (FER2013) | 准确率 65-72%，开箱即用 |
| UI 框架 | Streamlit | 仪表盘布局强，纯 Python |
| 图表 | Plotly / Altair | Streamlit 原生兼容 |
| 图像处理 | OpenCV + PIL | 标准工具链 |

## 项目结构

```
app.py                    # Streamlit 主界面，双模式切换
face_detector.py          # 人脸检测 (MediaPipe)
expression_recognizer.py  # 表情识别 (HuggingFace 预训练模型)
analyzer.py               # 统计分析器
utils.py                  # 辅助函数
test_images/              # 测试图片
test_videos/              # 测试视频
results/                  # 输出结果
records.csv               # 检测记录
requirements.txt          # 依赖
```

## 模块设计

### face_detector.py
- `detect_faces(image) -> list[Face]` — 检测所有人脸
- `extract_face_roi(image, face) -> np.ndarray` — 裁剪人脸区域到 224x224
- `Face = namedtuple('Face', ['bbox', 'confidence'])`

### expression_recognizer.py
- `EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]`
- `ExpressionRecognizer(model_name)` — 加载 HuggingFace 模型
- `recognize(face_img) -> dict[str, float]` — 返回 7 种表情概率
- `top_emotion(probs) -> str` — 最高概率表情

### analyzer.py
- `ResultAnalyzer` — 收集检测结果
- `add_record(ts, img_name, person_id, emotion, conf)` — 记录一条
- `get_summary() -> dict` — 总人数、各表情计数、比例、主导表情
- `export_csv(filepath)` — 导出 CSV
- `get_timeline() -> list` — 时间序列

### utils.py
- `draw_face_boxes(image, faces, emotions)` — 画框+标签
- `apply_emoji_overlay(image, face, emotion)` — 叠加 emoji
- `apply_mood_filter(image, emotion)` — 表情滤镜
- `read_csv/write_csv` — CSV 读写

### app.py — Streamlit UI

**侧边栏：**
- 模式切换：魔镜模式 / 仪表盘模式
- 输入源：上传图片 / 上传视频 / 实时摄像头
- 置信度阈值调整
- 重置统计 / 导出 CSV

**主区域 — 魔镜模式：**
- emoji 贴纸实时跟随人脸
- 表情对应色调滤镜
- "拍照"按钮保存带水印照片
- 表情弹幕文字

**主区域 — 仪表盘模式：**
- IoT 监控中心深色主题
- 实时检测画面（带框+标签）
- 统计卡片（总人数、主导表情、各表情计数）
- Plotly 环形图 / 柱状图
- 时间序列折线图
- CSV 记录表格

## 数据流

```
输入 (图片/视频帧/摄像头帧)
  → face_detector.detect_faces() → [Face, ...]
  → expression_recognizer.recognize() per face → [{emotion: prob}, ...]
  → analyzer.add_record() → 内存统计
  → utils.draw_face_boxes() / apply_emoji_overlay() → 输出画面
  → analyzer.export_csv() → records.csv
```

## 双模式设计

### 魔镜模式 (Mirror Mode)
互动趣味展示：emoji 贴纸跟随人脸实时渲染，根据检测到的表情应用不同的色调滤镜（开心=暖黄、悲伤=冷蓝、生气=红色、惊讶=紫色），支持一键拍照保存带日期水印的趣味照片。

### 仪表盘模式 (Dashboard Mode)
专业 IoT 监控风格：深色主题界面，仿监控中心布局。实时统计面板显示所有检测结果，支持环形图（表情占比）、柱状图（各表情计数）、时间序列折线图（情绪趋势），以及完整的 CSV 记录表格。

## 待定细节
- HuggingFace 具体模型名称在实施阶段确定（测试 2-3 个选择最佳）
- emoji 贴纸使用 Pillow 叠加 PNG 素材，素材文件放在 `assets/` 目录
