import os
# 默认允许在线下载；国内用户设置 HF_ENDPOINT=https://hf-mirror.com 加速
# 离线使用：设置环境变量 HF_HUB_OFFLINE=1

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          ViTImageProcessor)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]

_LABEL_MAPS = [
    {  # dima806
        "angry": "Angry", "anger": "Angry", "disgust": "Disgust", "fear": "Fear",
        "happy": "Happy", "sad": "Sad", "surprise": "Surprise", "neutral": "Neutral",
    },
    {  # mo-thecreator
        "angry": "Angry", "anger": "Angry", "disgust": "Disgust", "fear": "Fear",
        "happy": "Happy", "sad": "Sad", "surprise": "Surprise", "neutral": "Neutral",
    },
    {  # HardlyHumans (8-class, includes contempt)
        "angry": "Angry", "anger": "Angry", "contempt": "Contempt",
        "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
        "neutral": "Neutral", "sad": "Sad", "surprise": "Surprise",
    },
]


def _augment_variants(face_img: np.ndarray) -> list[Image.Image]:
    """TTA 变体 (5个：原图+翻转+旋转+亮度)"""
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
    """三模型集成：dima806 + mo-thecreator + HardlyHumans ViT，平均概率互相纠偏"""

    # (model_name, use_vit_processor)
    _MODEL_SPECS = [
        ("dima806/facial_emotions_image_detection", False),
        ("mo-thecreator/vit-Facial-Expression-Recognition", False),
        ("HardlyHumans/Facial-expression-detection", True),
    ]

    def __init__(self, device: int = 0):
        _device = device if torch.cuda.is_available() else -1
        self._device = torch.device(f"cuda:{_device}" if _device >= 0 else "cpu")
        self._models = []
        self._processors = []
        self._id2labels = []

        for name, use_vit in self._MODEL_SPECS:
            if use_vit:
                proc = ViTImageProcessor.from_pretrained(name, local_files_only=True)
            else:
                proc = AutoImageProcessor.from_pretrained(name, local_files_only=True)
            model = AutoModelForImageClassification.from_pretrained(name, local_files_only=True)
            model.to(self._device)
            model.eval()
            self._processors.append(proc)
            self._models.append(model)
            self._id2labels.append(model.config.id2label)

    def recognize(self, face_img: np.ndarray, head_status: str = None) -> dict[str, float]:
        """三模型集成推理：各自TTA，平均所有概率"""
        variants = _augment_variants(face_img)
        all_probs = []

        for proc, model, id2label, label_map in zip(
            self._processors, self._models, self._id2labels, _LABEL_MAPS
        ):
            inputs = proc(images=variants, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs_tensor = torch.softmax(logits, dim=-1)

            avg_probs = probs_tensor.mean(dim=0)
            accumulated = {id2label[i].capitalize(): avg_probs[i].item()
                          for i in range(len(avg_probs))}

            model_probs = {}
            for label, score in accumulated.items():
                key = label_map.get(label.lower(), label)
                model_probs[key] = score

            total = sum(model_probs.values())
            if total > 0:
                model_probs = {k: v / total for k, v in model_probs.items()}
            all_probs.append(model_probs)

        # 三模型平均
        probs = {}
        all_keys = set()
        for mp in all_probs:
            all_keys.update(mp.keys())
        for k in all_keys:
            vals = [mp.get(k, 0.0) for mp in all_probs]
            probs[k] = sum(vals) / len(vals)

        total = sum(probs.values())
        if total == 0:
            return {e: 0.0 for e in EMOTIONS}
        probs = {k: v / total for k, v in probs.items()}

        # 温度缩放
        t = 0.65
        probs = {k: v ** (1.0 / t) for k, v in probs.items()}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Neutral 压制
        top = max(probs, key=probs.get)
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        neutral_val = probs.get("Neutral", 0)
        if top == "Neutral":
            gap = sorted_items[0][1] - sorted_items[1][1]
            if gap < 0.08:
                probs["Neutral"] *= 0.50
                runner_up = sorted_items[1][0]
                probs[runner_up] *= 1.10
            elif neutral_val < 0.45:
                probs["Neutral"] *= 0.60
                runner_up = sorted_items[1][0]
                probs[runner_up] *= 1.08
        elif neutral_val > 0.30:
            probs["Neutral"] *= 0.80

        # Sad 偏置修正
        sad_val = probs.get("Sad", 0)
        if top == "Sad" and sad_val > 0.50:
            gap = sorted_items[0][1] - sorted_items[1][1]
            if gap > 0.35:
                transfer = sad_val * 0.50
                probs["Sad"] -= transfer
                probs["Neutral"] = probs.get("Neutral", 0) + transfer * 0.6
                probs["Happy"] = probs.get("Happy", 0) + transfer * 0.4
            else:
                probs["Sad"] *= 0.55
        elif top == "Sad":
            probs["Sad"] *= 0.65
        elif sad_val > 0.15:
            probs["Sad"] *= 0.75

        # 头姿态上下文
        if head_status == "抬头":
            probs["Happy"] = probs.get("Happy", 0) * 1.05
            probs["Surprise"] = probs.get("Surprise", 0) * 1.05

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def top_emotion(self, probs: dict[str, float]) -> str:
        return max(probs, key=probs.get)
