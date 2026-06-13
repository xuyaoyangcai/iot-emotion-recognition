import cv2
import numpy as np
from collections import namedtuple

Face = namedtuple("Face", ["bbox", "confidence"])

_face_detector = None


def _get_detector():
    global _face_detector
    if _face_detector is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_detector = cv2.CascadeClassifier(cascade_path)
    return _face_detector


def detect_faces(image: np.ndarray) -> list[Face]:
    """检测图像中的所有人脸，返回 Face 列表"""
    detector = _get_detector()

    # 转为灰度图用于 Haar 级联检测
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # detectMultiScale3 返回 (rects, rejectLevels, levelWeights)
    rects, _, weights = detector.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        outputRejectLevels=True
    )

    faces = []
    if rects is not None and len(rects) > 0:
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(weight)
            face = Face(
                bbox=(x, y, x + w, y + h),
                confidence=confidence
            )
            faces.append(face)
    return faces


def extract_face_roi(image: np.ndarray, face: Face) -> np.ndarray:
    """裁剪人脸区域并缩放到 224x224"""
    x1, y1, x2, y2 = face.bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    roi = cv2.resize(roi, (224, 224))
    return roi
