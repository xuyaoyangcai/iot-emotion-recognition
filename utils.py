import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_detector import Face
from expression_recognizer import EMOTIONS

# 中文字体（Windows 默认）
try:
    _CN_FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    _CN_FONT_SM = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
except (OSError, IOError):
    _CN_FONT = ImageFont.load_default()
    _CN_FONT_SM = ImageFont.load_default()


def _pil_draw_text(img_bgr, text, x, y, font, text_color, bg_color):
    """用 PIL 在 OpenCV BGR 图像上绘制文字（支持中文），返回文字宽高"""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # 背景
    draw.rectangle([x, y, x + tw + 6, y + th + 4], fill=bg_color[::-1])  # BGR→RGB
    draw.text((x + 3, y + 1), text, fill=text_color[::-1], font=font)
    rgb = np.array(pil_img)
    img_bgr[:, :, :] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return tw + 6, th + 4

# 表情 → 颜色映射 (BGR)
EMOTION_COLORS = {
    "Happy": (0, 215, 255),     # 金色
    "Neutral": (200, 200, 200),  # 灰色
    "Sad": (150, 100, 50),       # 蓝色
    "Angry": (0, 0, 255),        # 红色
    "Surprise": (255, 120, 0),   # 橙色
    "Fear": (100, 0, 100),       # 紫色
    "Disgust": (0, 100, 50),     # 深绿
    "Contempt": (80, 200, 200),  # 青色
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
    "Contempt": "😏",
}

# 课堂状态 → 颜色映射
CLASSROOM_STATE_COLORS = {
    "课堂状态良好": "#4CAF50",
    "课堂状态平稳": "#FF9800",
    "课堂状态较低落或需要关注": "#F44336",
    "课堂注意力波动较大": "#FF5722",
    "未检测到学生": "#9E9E9E",
}

# 课堂复合情绪 → 颜色映射 (BGR)
COMPOSITE_EMOTION_COLORS = {
    "Focused":     (100, 200, 0),     # green
    "Distracted":  (255, 100, 0),     # orange-red
    "Engaged":     (255, 215, 0),     # gold
    "Confused":    (0, 120, 255),     # orange
    "Thinking":    (50, 150, 200),    # blue-gray
    "Tired":       (150, 50, 100),    # dark blue
    "Bored":       (180, 180, 100),   # teal-gray
    "Anxious":     (200, 50, 200),    # magenta
}

COMPOSITE_EMOTION_EMOJI = {
    "Focused":     "🎯",
    "Distracted":  "👀",
    "Engaged":     "🙋",
    "Confused":    "🤔",
    "Thinking":    "💭",
    "Tired":       "😴",
    "Bored":       "🥱",
    "Anxious":     "😰",
}

# 表情 → 滤镜色调 (RGB 偏色系数)
MOOD_FILTERS = {
    "Happy": (1.1, 1.0, 0.8),     # 暖黄
    "Neutral": (1.0, 1.0, 1.0),    # 无色
    "Sad": (0.8, 0.9, 1.1),        # 冷蓝
    "Angry": (1.1, 0.7, 0.7),      # 偏红
    "Surprise": (1.0, 0.8, 1.1),   # 淡紫
    "Fear": (0.8, 0.8, 1.0),       # 暗蓝
    "Disgust": (0.8, 1.1, 0.7),    # 偏绿
    "Contempt": (0.9, 1.0, 0.8),   # 冷淡
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
    img[:, :, 2] *= r  # R channel
    img[:, :, 1] *= g  # G channel
    img[:, :, 0] *= b  # B channel
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
