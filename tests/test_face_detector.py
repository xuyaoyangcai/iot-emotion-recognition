import numpy as np
from face_detector import detect_faces, extract_face_roi, Face


def test_face_namedtuple():
    face = Face(bbox=(0, 0, 100, 100), confidence=0.95, landmarks=None)
    assert face.bbox == (0, 0, 100, 100)
    assert face.confidence == 0.95
    assert face.landmarks is None


def test_detect_faces_returns_list():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detect_faces(img)
    assert isinstance(results, list)
    assert len(results) == 0


def test_extract_face_roi_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    face = Face(bbox=(100, 100, 200, 200), confidence=0.9, landmarks=None)
    roi = extract_face_roi(img, face)
    assert roi.shape == (224, 224, 3)
