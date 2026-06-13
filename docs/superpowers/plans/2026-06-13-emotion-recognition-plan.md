# 人脸表情识别系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 Streamlit 的双模式人脸表情识别系统，支持图片/视频/摄像头输入，实时检测+统计+CSV导出。

**Architecture:** 5个Python模块组成流水线：MediaPipe人脸检测 → HuggingFace预训练表情识别 → 统计分析 → CSV导出，Streamlit前端分魔镜模式（emoji贴纸+滤镜互动）和仪表盘模式（IoT监控风统计面板）。

**Tech Stack:** Python 3.10+, MediaPipe, HuggingFace Transformers, PyTorch, Streamlit, Plotly, OpenCV, Pillow

---

### Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `test_images/` (空目录)
- Create: `test_videos/` (空目录)
- Create: `results/` (空目录)
- Create: `assets/` (空目录)

- [ ] **Step 1: 创建 .gitignore**

```bash
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
.env
venv/
.venv/
.superpowers/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
GITIGNORE
git add .gitignore
```

- [ ] **Step 2: 创建 requirements.txt**

```bash
cat > requirements.txt << 'EOF'
streamlit>=1.28.0
mediapipe>=0.10.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
transformers>=4.35.0
torch>=2.0.0
EOF
git add requirements.txt
```

- [ ] **Step 3: 创建目录结构并提交**

```bash
mkdir -p test_images test_videos results assets
touch test_images/.gitkeep test_videos/.gitkeep results/.gitkeep assets/.gitkeep
git add test_images/.gitkeep test_videos/.gitkeep results/.gitkeep assets/.gitkeep .gitignore requirements.txt
git commit -m "chore: 项目初始化 — 依赖和目录结构"
```

---

### Task 2: 人脸检测模块

**Files:**
- Create: `face_detector.py`
- Create: `tests/test_face_detector.py`

- [ ] **Step 1: 编写人脸检测的测试**

创建 `tests/test_face_detector.py`:

```python
import numpy as np
from face_detector import detect_faces, extract_face_roi, Face


def test_face_namedtuple():
    face = Face(bbox=(0, 0, 100, 100), confidence=0.95)
    assert face.bbox == (0, 0, 100, 100)
    assert face.confidence == 0.95


def test_detect_faces_returns_list():
    # 创建一个简单的测试图像 (640x480, 3通道, 全黑)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detect_faces(img)
    assert isinstance(results, list)
    # 黑色图像应该检测不到人脸
    assert len(results) == 0


def test_extract_face_roi_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    face = Face(bbox=(100, 100, 200, 200), confidence=0.9)
    roi = extract_face_roi(img, face)
    assert roi.shape == (224, 224, 3)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_face_detector.py -v
```
预期: `ModuleNotFoundError: No module named 'face_detector'`

- [ ] **Step 3: 实现 face_detector.py**

```python
import mediapipe as mp
import numpy as np
from collections import namedtuple

Face = namedtuple("Face", ["bbox", "confidence"])

_face_detection = None


def _get_detector():
    global _face_detection
    if _face_detection is None:
        _face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    return _face_detection


def detect_faces(image: np.ndarray) -> list[Face]:
    """检测图像中的所有人脸，返回 Face 列表"""
    h, w = image.shape[:2]
    detector = _get_detector()
    results = detector.process(image)

    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            # 确保坐标在图像范围内
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            face = Face(
                bbox=(x, y, x + bw, y + bh),
                confidence=det.score[0]
            )
            faces.append(face)
    return faces


def extract_face_roi(image: np.ndarray, face: Face) -> np.ndarray:
    """裁剪人脸区域并缩放到 224x224"""
    x1, y1, x2, y2 = face.bbox
    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))
    return roi


import cv2  # noqa: E402 (import at end for resize usage)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_face_detector.py -v
```
预期: 3 passed

- [ ] **Step 5: 提交**

```bash
git add face_detector.py tests/test_face_detector.py
git commit -m "feat: 人脸检测模块 (MediaPipe Face Detection)"
```

---

### Task 3: 表情识别模块

**Files:**
- Create: `expression_recognizer.py`
- Create: `tests/test_expression_recognizer.py`

- [ ] **Step 1: 编写表情识别测试**

创建 `tests/test_expression_recognizer.py`:

```python
import numpy as np
from expression_recognizer import ExpressionRecognizer, EMOTIONS


def test_emotions_list():
    assert len(EMOTIONS) == 7
    assert "Happy" in EMOTIONS
    assert "Neutral" in EMOTIONS
    assert "Sad" in EMOTIONS


def test_recognizer_init():
    rec = ExpressionRecognizer()
    assert rec is not None
    assert hasattr(rec, "recognize")


def test_recognize_format():
    rec = ExpressionRecognizer()
    # 随机输入图像 (224x224)
    face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = rec.recognize(face_img)
    assert isinstance(result, dict)
    for emotion in EMOTIONS:
        assert emotion in result
        assert 0.0 <= result[emotion] <= 1.0
    # 概率和应接近 1.0
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_expression_recognizer.py -v
```
预期: import error

- [ ] **Step 3: 实现 expression_recognizer.py**

```python
import numpy as np
from PIL import Image
from transformers import pipeline

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# FER2013 标签映射 (HuggingFace 模型输出 → 标准标签)
_LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


class ExpressionRecognizer:
    def __init__(self, model_name: str = "dima806/facial_emotions_image_detection"):
        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            device=-1  # CPU
        )

    def recognize(self, face_img: np.ndarray) -> dict[str, float]:
        """识别单张人脸的表情，返回7种情绪概率"""
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)

        predictions = self._pipe(face_img)

        result = {emotion: 0.0 for emotion in EMOTIONS}
        for pred in predictions:
            label = pred["label"].lower()
            score = pred["score"]
            if label in _LABEL_MAP:
                result[_LABEL_MAP[label]] = score
        return result

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_expression_recognizer.py -v
```
预期: 3 passed (注意：首次运行会下载模型 ~300MB)

- [ ] **Step 5: 提交**

```bash
git add expression_recognizer.py tests/test_expression_recognizer.py
git commit -m "feat: 表情识别模块 (HuggingFace ViT)"
```

---

### Task 4: 统计分析模块

**Files:**
- Create: `analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: 编写分析器测试**

创建 `tests/test_analyzer.py`:

```python
import os
import tempfile
from analyzer import ResultAnalyzer, EMOTIONS


def test_add_and_summary():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 2, "Neutral", 0.72)
    analyzer.add_record("2026-06-13 10:20:01", "test.jpg", 1, "Happy", 0.90)
    analyzer.add_record("2026-06-13 10:20:01", "test.jpg", 2, "Sad", 0.65)

    summary = analyzer.get_summary()
    assert summary["total_people"] == 2
    assert summary["expression_count"]["Happy"] == 2
    assert summary["expression_count"]["Neutral"] == 1
    assert summary["expression_count"]["Sad"] == 1
    assert summary["expression_count"]["Angry"] == 0
    assert summary["expression_ratio"]["Happy"] == 50.0
    assert summary["expression_ratio"]["Neutral"] == 25.0
    assert summary["expression_ratio"]["Sad"] == 25.0
    assert summary["main_expression"] == "Happy"


def test_export_csv():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    try:
        analyzer.export_csv(tmp)
        with open(tmp, "r") as f:
            content = f.read()
        assert "Timestamp" in content
        assert "Happy" in content
        assert "0.85" in content
    finally:
        os.unlink(tmp)


def test_clear():
    analyzer = ResultAnalyzer()
    analyzer.add_record("2026-06-13 10:20:00", "test.jpg", 1, "Happy", 0.85)
    analyzer.clear()
    summary = analyzer.get_summary()
    assert summary["total_people"] == 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_analyzer.py -v
```
预期: import error

- [ ] **Step 3: 实现 analyzer.py**

```python
import csv
from datetime import datetime
from expression_recognizer import EMOTIONS


class ResultAnalyzer:
    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, timestamp: str, image_name: str,
                   person_id: int, emotion: str, confidence: float):
        self.records.append({
            "timestamp": timestamp,
            "image_name": image_name,
            "person_id": person_id,
            "emotion": emotion,
            "confidence": round(confidence, 4),
        })

    def get_summary(self) -> dict:
        if not self.records:
            return {
                "total_people": 0,
                "expression_count": {e: 0 for e in EMOTIONS},
                "expression_ratio": {e: 0.0 for e in EMOTIONS},
                "main_expression": "N/A",
            }
        total = len(self.records)
        counts = {e: 0 for e in EMOTIONS}
        for r in self.records:
            em = r["emotion"]
            if em in counts:
                counts[em] += 1

        ratios = {e: round(counts[e] / total * 100, 1) for e in EMOTIONS}
        main = max(counts, key=counts.get)
        return {
            "total_people": total,
            "expression_count": counts,
            "expression_ratio": ratios,
            "main_expression": main,
        }

    def export_csv(self, filepath: str):
        fieldnames = ["Timestamp", "Image", "Person_ID",
                      "Happy", "Neutral", "Sad", "Angry",
                      "Surprise", "Fear", "Disgust", "Dominant"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # 按时间分组聚合
            groups = {}
            for r in self.records:
                key = (r["timestamp"], r["image_name"])
                if key not in groups:
                    groups[key] = {"counts": {e: 0 for e in EMOTIONS}, "total": 0}
                groups[key][r["emotion"]] = groups[key]["counts"].get(r["emotion"], 0) + 1
                groups[key]["total"] += 1

            for (ts, img), data in groups.items():
                row = {"Timestamp": ts, "Image": img,
                       "Person_ID": data["total"]}
                for e in EMOTIONS:
                    row[e] = data["counts"][e]
                dominant = max(data["counts"], key=data["counts"].get)
                row["Dominant"] = dominant if data["counts"][dominant] > 0 else "N/A"
                writer.writerow(row)

    def clear(self):
        self.records.clear()

    def get_timeline(self) -> list[dict]:
        timeline = []
        for r in self.records:
            timeline.append({
                "timestamp": r["timestamp"],
                "emotion": r["emotion"],
            })
        return timeline
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_analyzer.py -v
```
预期: 3 passed

- [ ] **Step 5: 提交**

```bash
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: 统计分析模块 (计数/比例/CSV导出)"
```

---

### Task 5: 工具函数模块

**Files:**
- Create: `utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: 编写工具函数测试**

创建 `tests/test_utils.py`:

```python
import numpy as np
from face_detector import Face
from utils import draw_face_boxes, apply_mood_filter, make_emoji_overlay


def test_draw_face_boxes():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = [Face(bbox=(100, 100, 200, 200), confidence=0.9)]
    emotions = [{"Happy": 0.85, "Neutral": 0.10, "Sad": 0.02,
                  "Angry": 0.01, "Surprise": 0.01, "Fear": 0.005, "Disgust": 0.005}]
    result = draw_face_boxes(img, faces, emotions)
    assert result.shape == (480, 640, 3)
    # 确保有绘制内容 (不全黑)
    assert result.sum() > 0


def test_apply_mood_filter():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = apply_mood_filter(img, "Happy")
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_make_emoji_overlay():
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    face = Face(bbox=(100, 100, 200, 200), confidence=0.9)
    result = make_emoji_overlay(img, face, "Happy")
    assert result.shape == (480, 640, 3)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_utils.py -v
```
预期: import error

- [ ] **Step 3: 实现 utils.py**

```python
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_detector import Face
from expression_recognizer import EMOTIONS

# 表情 → 颜色映射 (BGR)
EMOTION_COLORS = {
    "Happy": (0, 215, 255),     # 金色
    "Neutral": (200, 200, 200),  # 灰色
    "Sad": (150, 100, 50),       # 蓝色
    "Angry": (0, 0, 255),        # 红色
    "Surprise": (255, 120, 0),   # 橙色
    "Fear": (100, 0, 100),       # 紫色
    "Disgust": (0, 100, 50),     # 深绿
}

# 表情 → Emoji 字符
EMOTION_EMOJI = {
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Angry": "😡",
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
}

# 表情 → 滤镜色调 (RGB 偏色)
MOOD_FILTERS = {
    "Happy": (1.1, 1.0, 0.8),     # 暖黄
    "Neutral": (1.0, 1.0, 1.0),    # 无色
    "Sad": (0.8, 0.9, 1.1),        # 冷蓝
    "Angry": (1.1, 0.7, 0.7),      # 偏红
    "Surprise": (1.0, 0.8, 1.1),   # 淡紫
    "Fear": (0.8, 0.8, 1.0),       # 暗蓝
    "Disgust": (0.8, 1.1, 0.7),    # 偏绿
}


def draw_face_boxes(image: np.ndarray, faces: list[Face],
                    emotions: list[dict]) -> np.ndarray:
    """在人脸区域画框和表情标签"""
    img = image.copy()
    for face, probs in zip(faces, emotions):
        x1, y1, x2, y2 = face.bbox
        top_emotion = max(probs, key=probs.get)
        color = EMOTION_COLORS.get(top_emotion, (0, 255, 0))

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 标签文字
        label = f"{top_emotion} ({face.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def apply_mood_filter(image: np.ndarray, emotion: str) -> np.ndarray:
    """根据情绪应用色调滤镜"""
    if emotion not in MOOD_FILTERS:
        return image
    r, g, b = MOOD_FILTERS[emotion]
    img = image.copy().astype(np.float32)
    img[:, :, 2] *= r  # R
    img[:, :, 1] *= g  # G
    img[:, :, 0] *= b  # B
    return np.clip(img, 0, 255).astype(np.uint8)


def make_emoji_overlay(image: np.ndarray, face: Face, emotion: str) -> np.ndarray:
    """在人脸上方叠加 emoji 文字"""
    img = image.copy()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    emoji = EMOTION_EMOJI.get(emotion, "😐")
    x1, y1, x2, y2 = face.bbox
    emoji_x = x1 + (x2 - x1) // 2 - 24
    emoji_y = y1 - 40

    try:
        font = ImageFont.truetype("seguiemj.ttf", 48)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((emoji_x, emoji_y), emoji, font=font, embedded_color=True)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_utils.py -v
```
预期: 3 passed

- [ ] **Step 5: 提交**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: 工具函数模块 (画框/滤镜/emoji叠加)"
```

---

### Task 6: Streamlit 主界面 — 基础框架

**Files:**
- Create: `app.py`

- [ ] **Step 1: 创建双模式 Streamlit 应用骨架**

创建 `app.py`:

```python
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from PIL import Image

from face_detector import detect_faces, extract_face_roi
from expression_recognizer import ExpressionRecognizer, EMOTIONS
from analyzer import ResultAnalyzer
from utils import (
    draw_face_boxes, apply_mood_filter,
    make_emoji_overlay, EMOTION_COLORS
)

# ── 页面配置 ──────────────────────────────────
st.set_page_config(
    page_title="表情识别系统",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局状态初始化 ────────────────────────────
if "analyzer" not in st.session_state:
    st.session_state.analyzer = ResultAnalyzer()
if "recognizer" not in st.session_state:
    st.session_state.recognizer = ExpressionRecognizer()
if "captured_photo" not in st.session_state:
    st.session_state.captured_photo = None

analyzer = st.session_state.analyzer
recognizer = st.session_state.recognizer

# ── 侧边栏 ─────────────────────────────────────
with st.sidebar:
    st.title("🎭 表情识别系统")
    st.markdown("---")

    mode = st.radio(
        "📌 选择模式",
        ["🪞 魔镜模式", "📊 仪表盘模式"],
    )

    st.markdown("---")
    input_type = st.radio(
        "📥 输入来源",
        ["📷 上传图片", "🎬 上传视频", "🎥 实时摄像头"],
    )

    confidence_threshold = st.slider(
        "🎚️ 检测阈值",
        min_value=0.3, max_value=0.9, value=0.5, step=0.05,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置统计", use_container_width=True):
            analyzer.clear()
            st.rerun()
    with col2:
        if analyzer.records:
            buf = StringIO()
            analyzer.export_csv(buf)
            st.download_button(
                "📥 导出CSV",
                data=buf.getvalue(),
                file_name=f"emotion_records_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ── 处理函数 ──────────────────────────────────
@st.cache_resource
def get_recognizer():
    return ExpressionRecognizer()


def process_frame(image: np.ndarray) -> tuple[np.ndarray, list, list]:
    """处理单帧：检测人脸 → 识别表情"""
    faces = detect_faces(image)
    emotions = []
    for face in faces:
        if face.confidence < confidence_threshold:
            continue
        roi = extract_face_roi(image, face)
        probs = recognizer.recognize(roi)
        emotions.append(probs)
    return faces, emotions


def draw_results_mirror(image, faces, emotions):
    """魔镜模式渲染"""
    img = image.copy()
    for face, probs in zip(faces, emotions):
        top_emo = max(probs, key=probs.get)
        img = apply_mood_filter(img, top_emo)
        img = make_emoji_overlay(img, face, top_emo)
    return img


def draw_results_dashboard(image, faces, emotions):
    """仪表盘模式渲染"""
    return draw_face_boxes(image, faces, emotions)


def save_records(faces, emotions, source_name):
    """保存检测记录到分析器"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (face, probs) in enumerate(zip(faces, emotions)):
        top_emo = max(probs, key=probs.get)
        top_conf = probs[top_emo]
        analyzer.add_record(ts, source_name, i + 1, top_emo, top_conf)


def render_stats_panel():
    """渲染统计面板"""
    summary = analyzer.get_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 检测人次", summary["total_people"])
    with col2:
        st.metric("🎯 主导表情", summary["main_expression"])
    with col3:
        active = sum(1 for v in summary["expression_count"].values() if v > 0)
        st.metric("🌈 表情种类", active)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        # 环形图
        data = {e: summary["expression_count"][e] for e in EMOTIONS
                if summary["expression_count"][e] > 0}
        if data:
            fig = go.Figure(data=[go.Pie(
                labels=list(data.keys()),
                values=list(data.values()),
                hole=0.5,
                marker_colors=[f"rgb{tuple(int(c*0.8) for c in EMOTION_COLORS.get(e, (128,128,128)))[::-1]}"
                               for e in data.keys()],
            )])
            fig.update_layout(
                title="情绪占比",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        # 柱状图
        if data:
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                title="各表情计数",
                labels={"x": "表情", "y": "次数"},
            )
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # CSV 记录表格
    if analyzer.records:
        st.markdown("---")
        st.subheader("📋 检测记录")
        df = pd.DataFrame(analyzer.records)
        st.dataframe(df, use_container_width=True, height=200)


# ── 主界面 ─────────────────────────────────────
st.title("🎭 智能表情识别系统")
st.caption("人脸表情识别 — IoT 综合大作业")

if input_type == "📷 上传图片":
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)

        st.markdown("---")
        col_img, col_stats = st.columns([2, 1])

        with col_img:
            if "🪞" in mode:
                result = draw_results_mirror(image, faces, emotions)
            else:
                result = draw_results_dashboard(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸",
                     use_container_width=True)

        with col_stats:
            save_records(faces, emotions, uploaded.name)
            render_stats_panel()

elif input_type == "🎬 上传视频":
    uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded:
        # 保存到临时文件
        tmp_path = f"_tmp_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"视频共 {frame_count} 帧，正在处理...")

        progress = st.progress(0)
        frame_placeholder = st.empty()
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            if processed % 5 != 0:  # 每5帧处理一次
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, emotions = process_frame(rgb)

            if "🪞" in mode:
                result = draw_results_mirror(rgb, faces, emotions)
            else:
                result = draw_results_dashboard(rgb, faces, emotions)

            frame_placeholder.image(result, use_container_width=True)
            save_records(faces, emotions, uploaded.name)
            progress.progress(min(processed / frame_count, 1.0))

        cap.release()
        progress.empty()
        st.success("视频处理完成!")

elif input_type == "🎥 实时摄像头":
    # 使用 Streamlit 内置 camera_input，返回单帧照片
    camera_photo = st.camera_input("📸 点击拍照", key="emotion_camera")

    if camera_photo:
        pil_img = Image.open(camera_photo).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, emotions = process_frame(image)

        st.markdown("---")
        col_img, col_stats = st.columns([2, 1])

        with col_img:
            if "🪞" in mode:
                result = draw_results_mirror(image, faces, emotions)
            else:
                result = draw_results_dashboard(image, faces, emotions)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption=f"检测到 {len(faces)} 张人脸",
                     use_container_width=True)

        with col_stats:
            save_records(faces, emotions, "camera_capture")
            render_stats_panel()

# ── 底部统计 ──────────────────────────────────
with st.sidebar:
    st.markdown("---")
    if analyzer.records:
        summary = analyzer.get_summary()
        st.subheader("📊 实时统计")
        st.write(f"检测人次: **{summary['total_people']}**")
        st.write(f"主导表情: **{summary['main_expression']}**")
        for e in EMOTIONS:
            if summary["expression_count"][e] > 0:
                st.write(f"{e}: {summary['expression_count'][e]} "
                         f"({summary['expression_ratio'][e]}%)")
```

- [ ] **Step 2: 启动测试确认能运行**

```bash
streamlit run app.py --server.port 8501
```
预期: 应用在 http://localhost:8501 启动，无 ImportError

- [ ] **Step 3: 提交**

```bash
git add app.py
git commit -m "feat: Streamlit 主界面 (魔镜+仪表盘 双模式)"
```

---

### Task 7: 创意功能增强

**Files:**
- Modify: `app.py`
- Create: `assets/emojis/` (创意emoji素材目录)

- [ ] **Step 1: 添加弹幕文字效果到魔镜模式**

在 `app.py` 中添加弹幕效果（修改 `draw_results_mirror` 函数）：

```python
def draw_results_mirror(image, faces, emotions):
    """魔镜模式渲染 — 加弹幕文字"""
    img = image.copy()
    for face, probs in zip(faces, emotions):
        top_emo = max(probs, key=probs.get)
        img = apply_mood_filter(img, top_emo)
        img = make_emoji_overlay(img, face, top_emo)
        # 弹幕文字
        danmaku = get_danmaku(top_emo)
        x1, y1, x2, y2 = face.bbox
        cv2.putText(img, danmaku, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    EMOTION_COLORS.get(top_emo, (255, 255, 255)), 2)
    return img


def get_danmaku(emotion):
    """返回表情对应的趣味文字"""
    danmaku_map = {
        "Happy": "今天很开心! ✨",
        "Sad": "有点emo... 😔",
        "Angry": "别惹我! 💢",
        "Surprise": "真的假的!? 😲",
        "Fear": "好可怕! 🙀",
        "Disgust": "咦~受不了 🤢",
        "Neutral": "淡定... 😐",
    }
    return danmaku_map.get(emotion, "")
```

- [ ] **Step 2: 仪表盘模式添加深色IoT主题**

在 `app.py` 页面配置处，通过 CSS 注入深色主题效果（仪表盘模式时使用暗色 metric 卡片样式）：

```python
# 在 set_page_config 之后添加
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
    }
    /* IoT 仪表盘风格装饰 */
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)
```

- [ ] **Step 3: 提交**

```bash
git add app.py assets/
git commit -m "feat: 创意增强 — 弹幕文字 + IoT深色主题"
```

---

### Task 8: 端到端验证与修复

**Files:**
- Modify: `app.py` (潜在修复)
- Create: `results/records.csv` (首次运行后)

- [ ] **Step 1: 运行完整流程测试**

```bash
python -m pytest tests/ -v
```
预期: 所有测试通过 (~11 tests)

- [ ] **Step 2: 准备测试图片**

找一张多人合照放入 `test_images/` 目录，例如命名为 `test_group.jpg`。

- [ ] **Step 3: Streamlit 全功能验证**

```bash
streamlit run app.py
```

验证清单:
- [ ] 上传图片 → 检测人脸 → 显示框+标签
- [ ] 切换魔镜模式 → emoji + 滤镜 + 弹幕生效
- [ ] 切换仪表盘模式 → 饼图/柱状图/表格正确
- [ ] 导出 CSV → 内容格式正确
- [ ] 重置统计 → 数据清零
- [ ] 实时摄像头 → 画面流畅 (非阻塞式)
- [ ] 阈值滑块 → 影响检测结果

- [ ] **Step 4: 修复发现的问题并提交**

```bash
git add -A
git commit -m "fix: 端到端验证修复"
```

---

### Task 9: README 文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 编写 README**

```markdown
# 🎭 智能表情识别系统

物联网课程期末大作业 — 基于深度学习的实时人脸表情识别系统。

## 功能

- **双模式UI**：魔镜模式（emoji贴纸+滤镜互动）+ 仪表盘模式（IoT监控风统计）
- **多输入源**：图片上传 / 视频上传 / 实时摄像头
- **7种表情识别**：Happy, Neutral, Sad, Angry, Surprise, Fear, Disgust
- **实时统计**：表情计数、占比、主导表情、时间序列
- **CSV导出**：完整检测记录导出

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 项目结构

```
app.py                    # Streamlit 主界面
face_detector.py          # MediaPipe 人脸检测
expression_recognizer.py  # HuggingFace 表情识别
analyzer.py               # 统计分析器
utils.py                  # 辅助函数
tests/                    # 单元测试
test_images/              # 测试图片
test_videos/              # 测试视频
results/                  # 输出结果
```

## 技术栈

- MediaPipe Face Detection
- HuggingFace Transformers (ViT)
- Streamlit
- OpenCV + Plotly
```

- [ ] **Step 2: 提交并推送**

```bash
git add README.md
git commit -m "docs: 添加 README"
git push origin master
```
```

After the README task, the project is complete.
