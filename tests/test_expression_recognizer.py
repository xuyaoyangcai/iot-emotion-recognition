import numpy as np
import pytest
from expression_recognizer import ExpressionRecognizer, EMOTIONS


@pytest.fixture(scope="session")
def recognizer():
    """Session-scoped fixture: load model once for all tests."""
    return ExpressionRecognizer()


def test_emotions_list():
    assert len(EMOTIONS) == 7
    assert "Happy" in EMOTIONS
    assert "Neutral" in EMOTIONS
    assert "Sad" in EMOTIONS


def test_recognizer_init(recognizer):
    rec = recognizer
    assert rec is not None
    assert hasattr(rec, "recognize")


def test_recognize_format(recognizer):
    rec = recognizer
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
