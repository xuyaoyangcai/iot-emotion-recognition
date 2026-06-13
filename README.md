# 🎭 智能表情识别系统

物联网课程期末大作业 — 基于深度学习的实时人脸表情识别系统。

## 功能

- **双模式UI**：魔镜模式（emoji贴纸+弹幕+滤镜互动）+ 仪表盘模式（IoT监控风统计）
- **多输入源**：图片上传 / 视频上传 / 实时摄像头拍照
- **7种表情识别**：Happy, Neutral, Sad, Angry, Surprise, Fear, Disgust
- **实时统计**：表情计数、占比、主导表情、时间序列
- **CSV导出**：完整检测记录导出

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

首次运行会自动下载 HuggingFace 预训练模型（约 300MB）。国内用户可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
streamlit run app.py
```

## 项目结构

```
app.py                    # Streamlit 主界面（双模式）
face_detector.py          # 人脸检测（OpenCV Haar Cascade）
expression_recognizer.py  # 表情识别（HuggingFace ViT）
analyzer.py               # 统计分析器
utils.py                  # 辅助函数（画框/滤镜/emoji）
tests/                    # 单元测试
test_images/              # 测试图片
test_videos/              # 测试视频
results/                  # 输出结果
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 人脸检测 | OpenCV Haar Cascade |
| 表情识别 | HuggingFace ViT (dima806/facial_emotions_image_detection) |
| UI 框架 | Streamlit |
| 图表 | Plotly |
| 图像处理 | OpenCV + Pillow |

## 使用说明

1. 启动应用后，在侧边栏选择模式（魔镜/仪表盘）和输入源（图片/视频/摄像头）
2. **魔镜模式**：检测人脸后叠加 emoji 贴纸、表情滤镜和弹幕文字
3. **仪表盘模式**：显示检测框、标签、统计图表和 CSV 记录
4. 侧边栏可调整检测阈值、重置统计、导出 CSV

## 运行测试

```bash
python -m pytest tests/ -v
```
