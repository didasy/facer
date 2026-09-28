"""Canonical five-landmark face alignment."""
import cv2
import numpy as np

from .errors import InvalidImageError

REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a BGR image to 112x112 using five facial landmarks."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise InvalidImageError("Expected five finite facial landmarks (5x2)")
    tfm, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
    if tfm is None or not np.isfinite(tfm).all():
        raise InvalidImageError("Cannot align degenerate facial landmarks")
    return cv2.warpAffine(img, tfm, (112, 112))
