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
    def __init__(self, model_name: str = "dima806/facial_emotions_image_detection", device: int = 0):
        import torch
        _device = device if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            device=_device,
        )

    def recognize(self, face_img: np.ndarray, head_status: str = None) -> dict[str, float]:
        """TTA 增强识别：7个变体 + 温度缩放 + 强化 Neutral 压制 + 头姿态上下文"""
        variants = _augment_variants(face_img)

        accumulated = {emotion: 0.0 for emotion in EMOTIONS}
        for variant in variants:
            predictions = self._pipe(variant, top_k=7)
            for pred in predictions:
                label = pred["label"].lower()
                if label in _LABEL_MAP:
                    accumulated[_LABEL_MAP[label]] += pred["score"]

        total = sum(accumulated.values())
        if total == 0:
            return accumulated
        probs = {k: v / total for k, v in accumulated.items()}

        # 温度缩放 (T=0.35)：更"果断"，拉开类别差距
        t = 0.35
        probs = {k: v ** (1.0 / t) for k, v in probs.items()}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # 强化 Neutral 压制（始终生效，不只是 Neutral 排第一时）
        top = max(probs, key=probs.get)
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        neutral_val = probs.get("Neutral", 0)
        runner_up = sorted_items[1][0] if sorted_items[0][0] == "Neutral" else sorted_items[0][0]
        runner_up_val = probs.get(runner_up, 0)

        if top == "Neutral":
            gap = sorted_items[0][1] - sorted_items[1][1]
            if gap < 0.10:
                probs["Neutral"] *= 0.3
                probs[runner_up] *= 1.4
            elif neutral_val < 0.40:
                probs["Neutral"] *= 0.4
                probs[runner_up] *= 1.3
            elif neutral_val < 0.50:
                probs["Neutral"] *= 0.5
                probs[runner_up] *= 1.2
        else:
            # Neutral 不是第一但也偏高 (>25%)，压制
            if neutral_val > 0.25:
                probs["Neutral"] *= 0.7

        # 头姿态上下文修正: 低头→Sad/Angry倾向，抬头→Surprise/Happy倾向
        if head_status == "低头":
            probs["Sad"] = probs.get("Sad", 0) * 1.3
            probs["Angry"] = probs.get("Angry", 0) * 1.2
            probs["Happy"] = probs.get("Happy", 0) * 0.7
        elif head_status == "抬头":
            probs["Happy"] = probs.get("Happy", 0) * 1.2
            probs["Surprise"] = probs.get("Surprise", 0) * 1.2

        # 重新归一化
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
