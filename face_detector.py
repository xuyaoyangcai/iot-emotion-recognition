import cv2
import numpy as np
from collections import namedtuple

Face = namedtuple("Face", ["bbox", "confidence"])

_frontal_face = None
_profile_face = None


def _get_frontal():
    global _frontal_face
    if _frontal_face is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _frontal_face = cv2.CascadeClassifier(path)
    return _frontal_face


def _get_profile():
    global _profile_face
    if _profile_face is None:
        path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        _profile_face = cv2.CascadeClassifier(path)
    return _profile_face


def _detect_with(cascade, gray, scale_factor, min_neighbors, min_size):
    rects, _, weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        outputRejectLevels=True,
    )
    results = []
    if rects is not None and len(rects) > 0:
        for (x, y, w, h), weight in zip(rects, weights):
            results.append((x, y, w, h, float(weight)))
    return results


def detect_faces(image: np.ndarray) -> list[Face]:
    """检测图像中的所有人脸"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 直方图均衡化改善光照
    gray = cv2.equalizeHist(gray)

    frontal = _get_frontal()
    profile = _get_profile()

    raw_faces = []

    # 多轮检测：不同参数组合
    for scale in [1.05, 1.1, 1.2]:
        for neigh in [4, 5, 6]:
            results = _detect_with(frontal, gray, scale, neigh, (40, 40))
            raw_faces.extend(results)

    # 侧脸检测
    profile_results = _detect_with(profile, gray, 1.1, 5, (40, 40))
    raw_faces.extend(profile_results)

    if not raw_faces:
        return []

    # 去重 (IoU > 0.5 视为重复)
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    raw_faces.sort(key=lambda r: r[4], reverse=True)
    kept = []
    for r in raw_faces:
        dup = False
        for k in kept:
            if iou(r, k) > 0.4:
                dup = True
                break
        if not dup:
            kept.append(r)

    faces = []
    for (x, y, w, h, conf) in kept:
        # 扩大框的范围，给表情识别更多上下文
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        faces.append(Face(bbox=(x1, y1, x2, y2), confidence=conf))

    return faces


def extract_face_roi(image: np.ndarray, face: Face) -> np.ndarray:
    """裁剪人脸区域并缩放到 224x224"""
    x1, y1, x2, y2 = face.bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    roi = cv2.resize(roi, (224, 224))
    return roi
