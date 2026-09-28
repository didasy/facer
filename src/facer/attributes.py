"""FaceAttribNet independent binary heads on RGB [0, 1] input."""
import cv2
import numpy as np

from .config import FacerConfig
from .errors import CropOutOfBoundsError
from .results import Occlusions

_ATTRIB_INPUT_SIZE = (128, 128)


def _attrib_crop_face(img: np.ndarray, bbox: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Clip an optionally padded bounding box to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if margin:
        dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise CropOutOfBoundsError(f"Bounding box {bbox[:4]} does not overlap the {w}x{h} image")
    return img[y1:y2, x1:x2]


def _attrib_resize_pad(crop: np.ndarray, size: tuple[int, int] = _ATTRIB_INPUT_SIZE) -> np.ndarray:
    """Aspect-preserving resize with centered black padding."""
    dst_h, dst_w = size
    h, w = crop.shape[:2]
    scale = min(dst_h / h, dst_w / w)
    new_h, new_w = int(np.floor(h * scale)), int(np.floor(w * scale))
    if dst_h / h < dst_w / w:
        new_h = dst_h
    else:
        new_w = dst_w
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((dst_h, dst_w, crop.shape[2]), dtype=crop.dtype)
    top, left = (dst_h - new_h) // 2, (dst_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _attrib_blob(crop: np.ndarray) -> np.ndarray:
    """Prepare RGB NCHW input; mean/std normalization is baked into ONNX."""
    padded = _attrib_resize_pad(crop, _ATTRIB_INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def predict_face_attributes(img: np.ndarray, face_box: np.ndarray, session) -> tuple[float, float, float, float, float]:
    """Return left/right eye openness, eyeglasses, mask, sunglasses."""
    blob = _attrib_blob(_attrib_crop_face(img, face_box))
    out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: blob})[0]
    probs = np.squeeze(out).astype(float).tolist()
    return (float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]), float(probs[4]))


def build_occlusion_result(img: np.ndarray, face_box: np.ndarray, session, config: FacerConfig) -> Occlusions:
    """Threshold each head independently; sunglasses suppress closed-eye flags."""
    left_open, right_open, eyeglasses, mask, sunglasses = predict_face_attributes(img, face_box, session)
    eye_open_mean = (left_open + right_open) / 2.0
    sunglasses_detected = bool(sunglasses > config.attrib_threshold)
    return Occlusions(
        sunglasses_detected=sunglasses_detected,
        eyeglasses_detected=bool(eyeglasses > config.attrib_threshold),
        eye_closed_detected=False if sunglasses_detected else bool(eye_open_mean < config.eye_closed_threshold),
        mask_detected=bool(mask > config.attrib_threshold),
        hat_detected=None,
        veil_detected=None,
    )
