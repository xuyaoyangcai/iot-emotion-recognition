import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from transformers import pipeline

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


def _augment_variants(face_img: np.ndarray) -> list[Image.Image]:
    """生成多个人脸变体用于测试时增广 (TTA)"""
    h, w = face_img.shape[:2]
    variants = []

    # 转 PIL
    pil_img = Image.fromarray(face_img)
    variants.append(pil_img)  # 原始

    # 水平翻转
    variants.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))

    # 轻微旋转 ±8°
    for angle in [-8, 8]:
        rotated = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        variants.append(rotated)

    # 亮度变化
    for factor in [0.85, 1.15]:
        enhancer = ImageEnhance.Brightness(pil_img)
        variants.append(enhancer.enhance(factor))

    # 中心裁剪 90%
    crop = int(w * 0.05)
    cropped = pil_img.crop((crop, crop, w - crop, h - crop)).resize((w, h), Image.BILINEAR)
    variants.append(cropped)

    return variants


class ExpressionRecognizer:
    def __init__(self, model_name: str = "dima806/facial_emotions_image_detection"):
        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            device=-1,
        )

    def recognize(self, face_img: np.ndarray) -> dict[str, float]:
        """TTA 增强识别：生成7个变体，综合预测"""
        variants = _augment_variants(face_img)

        accumulated = {emotion: 0.0 for emotion in EMOTIONS}
        for variant in variants:
            predictions = self._pipe(variant, top_k=7)
            for pred in predictions:
                label = pred["label"].lower()
                if label in _LABEL_MAP:
                    accumulated[_LABEL_MAP[label]] += pred["score"]

        # 归一化
        total = sum(accumulated.values())
        if total > 0:
            return {k: v / total for k, v in accumulated.items()}
        return accumulated

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
