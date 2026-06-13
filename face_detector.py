import os
import cv2
import numpy as np
from collections import namedtuple

# landmarks: (5, 2) np.array — 右眼/左眼/鼻尖/右嘴角/左嘴角 (像素坐标)，无则为 None
Face = namedtuple("Face", ["bbox", "confidence", "landmarks"])

_detector = None

# 标准3D人脸模型 (用于 solvePnP 头部姿态估计)
_MODEL_POINTS = np.array([
    (-30.0, -30.0, -30.0),   # right eye
    (30.0, -30.0, -30.0),   # left eye
    (0.0, 0.0, 0.0),        # nose tip
    (-20.0, 20.0, -40.0),   # right mouth
    (20.0, 20.0, -40.0),    # left mouth
], dtype=np.float64)


def _get_detector():
    global _detector
    if _detector is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "opencv")
        model_path = os.path.join(cache_dir, "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet 模型未找到: {model_path}\n"
                "请下载: https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
                f"并放置到: {cache_dir}/"
            )
        _detector = cv2.FaceDetectorYN.create(
            model_path, "",
            (320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=500,
        )
    return _detector


def detect_faces(image: np.ndarray) -> list[Face]:
    """YuNet DNN 检测所有人脸 + 5个面部关键点"""
    h, w = image.shape[:2]

    if len(image.shape) == 3:
        bgr = image.copy()
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    detector = _get_detector()
    detector.setInputSize((w, h))

    _, results = detector.detect(bgr)

    faces = []
    if results is not None and len(results) > 0:
        for det in results:
            x1 = int(det[0])
            y1 = int(det[1])
            bw, bh = det[2], det[3]
            conf = float(det[14])

            if bw < 20 or bh < 20:
                continue

            # 提取5个关键点 (YuNet: 右眼 左眼 鼻尖 右嘴角 左嘴角)
            landmarks = np.array([
                [det[4], det[5]],    # right eye
                [det[6], det[7]],    # left eye
                [det[8], det[9]],    # nose tip
                [det[10], det[11]],  # right mouth
                [det[12], det[13]],  # left mouth
            ], dtype=np.float64)

            # Padding for expression recognition context
            pad_w = int(bw * 0.15)
            pad_h = int(bh * 0.15)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, int(det[0] + bw + pad_w))
            y2 = min(h, int(det[1] + bh + pad_h))

            faces.append(Face(bbox=(x1, y1, x2, y2), confidence=conf,
                            landmarks=landmarks))

    faces.sort(key=lambda f: f.confidence, reverse=True)
    return faces


def estimate_head_pose(face: Face, image_shape: tuple) -> tuple:
    """
    估算头部姿态，返回 (pitch, yaw, roll) 角度 (度)
    pitch > 0 = 低头, pitch < 0 = 抬头
    """
    if face.landmarks is None:
        return (0.0, 0.0, 0.0)

    h, w = image_shape[:2]

    # 相机内参：典型 webcam 焦距约 1.2× 图像宽度
    focal = w * 1.2
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, face.landmarks,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        return (0.0, 0.0, 0.0)

    rmat, _ = cv2.Rodrigues(rvec)

    # 提取欧拉角
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(-rmat[1, 2], rmat[1, 1])
        roll = 0.0

    return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))


def face_aspect_ratio(face: Face) -> float:
    """脸部宽高比 (height/width)，低头时比值变小"""
    x1, y1, x2, y2 = face.bbox
    w_box = x2 - x1
    h_box = y2 - y1
    if w_box <= 0:
        return 1.0
    return h_box / w_box


def classify_head_pose(pitch: float, threshold: float = 12.0,
                       face_ar: float = None) -> str:
    """
    综合俯仰角 + 脸部宽高比判断头部状态
    pitch > threshold  → "低头"
    pitch < -threshold → "抬头"
    如果 face_ar 明显偏小，直接判为低头
    """
    # 脸部宽高比辅助：正常脸约 1.2~1.4，低头时 < 1.05
    if face_ar is not None and face_ar < 1.05:
        return "低头"
    if face_ar is not None and face_ar < 1.10:
        threshold = threshold * 0.6

    if pitch > threshold:
        return "低头"
    elif pitch < -threshold:
        return "抬头"
    return "正常"


def extract_face_roi(image: np.ndarray, face: Face) -> np.ndarray:
    """裁剪人脸区域并缩放到 224x224，直方图均衡化改善光照"""
    x1, y1, x2, y2 = face.bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    roi = cv2.resize(roi, (224, 224))
    roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
    roi_yuv[:, :, 0] = cv2.equalizeHist(roi_yuv[:, :, 0])
    roi = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2BGR)
    return roi
