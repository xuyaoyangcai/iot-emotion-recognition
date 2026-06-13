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
