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
            device=-1,  # CPU
            local_files_only=True,  # 纯本地，不联网
        )

    def recognize(self, face_img: np.ndarray) -> dict[str, float]:
        """识别单张人脸的表情，返回7种情绪概率"""
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)

        predictions = self._pipe(face_img, top_k=7)

        result = {emotion: 0.0 for emotion in EMOTIONS}
        for pred in predictions:
            label = pred["label"].lower()
            score = pred["score"]
            if label in _LABEL_MAP:
                result[_LABEL_MAP[label]] = score
        return result

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
