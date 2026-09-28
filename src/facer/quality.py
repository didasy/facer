"""Non-blocking face sharpness and whole-image luminance reporting."""
import cv2
import numpy as np

from .config import FacerConfig
from .results import Quality


def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness of a BGR crop: variance of the Laplacian on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_blur(raw: float, score_max: float = 250.0) -> float:
    """Map raw Laplacian variance [0, inf) to a 0-1 blurriness score."""
    return 1.0 - min(raw / score_max, 1.0) if score_max > 0 else 1.0


def normalize_luminance(mean_gray: float) -> float:
    """Map mean gray intensity 0-255 to a 0-1 brightness score."""
    return min(max(mean_gray / 255.0, 0.0), 1.0)


def evaluate_quality(img: np.ndarray, face_box: np.ndarray | None = None, config: FacerConfig | None = None) -> Quality:
    """Report padded face-ROI blur (whole-image fallback) and image brightness."""
    config = config if config is not None else FacerConfig()
    h, w = img.shape[:2]
    if face_box is not None:
        x1, y1, x2, y2 = (float(v) for v in face_box)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        x1 = max(0, int(x1 - bw * config.face_pad_ratio))
        y1 = max(0, int(y1 - bh * config.face_pad_ratio))
        x2 = min(w, int(x2 + bw * config.face_pad_ratio))
        y2 = min(h, int(y2 + bh * config.face_pad_ratio))
        roi = img[y1:y2, x1:x2]
        raw_blur = laplacian_variance(roi) if roi.size else laplacian_variance(img)
    else:
        raw_blur = laplacian_variance(img)
    blur_score = normalize_blur(raw_blur, config.blur_score_max)
    blur_ceiling = 1.0 - config.blur_threshold / config.blur_score_max if config.blur_score_max > 0 else 1.0
    luminance_score = normalize_luminance(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
    return Quality(
        blur_score=round(blur_score, 4),
        is_blurry=blur_score > blur_ceiling,
        luminance_score=round(luminance_score, 4),
        is_dark=luminance_score < config.luminance_dark_min or luminance_score > config.luminance_dark_max
    )
