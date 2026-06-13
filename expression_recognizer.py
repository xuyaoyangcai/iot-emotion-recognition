import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor, AutoModelForImageClassification

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_LABEL_MAP = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear",
    "happy": "Happy", "sad": "Sad", "surprise": "Surprise", "neutral": "Neutral",
}


def _augment_variants(face_img: np.ndarray) -> list[Image.Image]:
    """生成 TTA 变体 (5个：原图+翻转+旋转+亮度)"""
    h, w = face_img.shape[:2]
    pil_img = Image.fromarray(face_img)
    variants = [pil_img]
    variants.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    for angle in [-8, 8]:
        variants.append(pil_img.rotate(angle, resample=Image.BILINEAR,
                                        fillcolor=(128, 128, 128)))
    enhancer = ImageEnhance.Brightness(pil_img)
    variants.append(enhancer.enhance(0.85))
    return variants


class ExpressionRecognizer:
    def __init__(self, model_name: str = "dima806/facial_emotions_image_detection",
                 device: int = 0):
        _device = device if torch.cuda.is_available() else -1
        self._device = torch.device(f"cuda:{_device}" if _device >= 0 else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        self._id2label = self._model.config.id2label

    def recognize(self, face_img: np.ndarray, head_status: str = None) -> dict[str, float]:
        """TTA 批处理：所有变体一次性送入 GPU，充分打满显卡"""
        variants = _augment_variants(face_img)

        # 批量预处理 → GPU tensor
        inputs = self._processor(images=variants, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (N, num_classes)
            probs = torch.softmax(logits, dim=-1)

        # 聚合所有变体的预测
        avg_probs = probs.mean(dim=0)  # (num_classes,)
        accumulated = {self._id2label[i].capitalize(): avg_probs[i].item()
                       for i in range(len(avg_probs))}
        # 映射标签
        probs = {}
        for label, score in accumulated.items():
            key = _LABEL_MAP.get(label.lower(), label)
            probs[key] = score

        total = sum(probs.values())
        if total == 0:
            return {e: 0.0 for e in EMOTIONS}
        probs = {k: v / total for k, v in probs.items()}

        # 温度缩放
        t = 0.35
        probs = {k: v ** (1.0 / t) for k, v in probs.items()}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Neutral 压制
        top = max(probs, key=probs.get)
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        neutral_val = probs.get("Neutral", 0)
        runner_up = sorted_items[1][0] if sorted_items[0][0] == "Neutral" else sorted_items[0][0]

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
        elif neutral_val > 0.25:
            probs["Neutral"] *= 0.7

        # 头姿态上下文
        if head_status == "低头":
            probs["Sad"] = probs.get("Sad", 0) * 1.3
            probs["Angry"] = probs.get("Angry", 0) * 1.2
            probs["Happy"] = probs.get("Happy", 0) * 0.7
        elif head_status == "抬头":
            probs["Happy"] = probs.get("Happy", 0) * 1.2
            probs["Surprise"] = probs.get("Surprise", 0) * 1.2

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
