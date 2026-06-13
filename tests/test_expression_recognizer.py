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
