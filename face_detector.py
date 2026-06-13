import os
import cv2
import numpy as np
from collections import namedtuple

Face = namedtuple("Face", ["bbox", "confidence"])

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        # 避免中文路径问题，模型放 ~/.cache/opencv/
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
            (320, 320),     # 内部处理尺寸
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=500,       # 最多检测500张脸
        )
    return _detector


def detect_faces(image: np.ndarray) -> list[Face]:
    """YuNet DNN 检测所有人脸 — 比 Haar Cascade 精准，小脸也能抓"""
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
            x2 = int(det[0] + det[2])
            y2 = int(det[1] + det[3])
            conf = float(det[14])

            # 过滤太小的框（可能是误检）
            if det[2] < 20 or det[3] < 20:
                continue

            # Padding 给表情识别更多上下文
            bw, bh = det[2], det[3]
            pad_w = int(bw * 0.15)
            pad_h = int(bh * 0.15)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            faces.append(Face(bbox=(x1, y1, x2, y2), confidence=conf))

    # 按置信度排序
    faces.sort(key=lambda f: f.confidence, reverse=True)
    return faces


def extract_face_roi(image: np.ndarray, face: Face) -> np.ndarray:
    """裁剪人脸区域并缩放到 224x224，直方图均衡化改善光照"""
    x1, y1, x2, y2 = face.bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    roi = cv2.resize(roi, (224, 224))

    # YUV 直方图均衡化
    roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
    roi_yuv[:, :, 0] = cv2.equalizeHist(roi_yuv[:, :, 0])
    roi = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2BGR)

    return roi
