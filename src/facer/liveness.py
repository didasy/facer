"""MiniFASNetV2 passive liveness on raw BGR input."""
import cv2
import numpy as np

from .errors import CropOutOfBoundsError


def _minifas_crop_face(img: np.ndarray, bbox_xywh: list[int], scale: float, out_w: int, out_h: int) -> np.ndarray:
    """Expand and clip a face crop for MiniFASNetV2."""
    src_h, src_w = img.shape[:2]
    x, y, box_w, box_h = bbox_xywh
    box_w, box_h = max(1, box_w), max(1, box_h)
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    center_x, center_y = x + box_w / 2, y + box_h / 2
    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))
    cropped = img[y1:y2 + 1, x1:x2 + 1]
    if not cropped.size:
        raise CropOutOfBoundsError("Liveness crop does not overlap the image")
    return cv2.resize(cropped, (out_w, out_h))


def predict_liveness(img: np.ndarray, face_box: np.ndarray, session, scale: float = 2.7) -> float:
    """Anti-spoof score P(real) in [0, 1] from MiniFASNetV2 softmax logits."""
    x1, y1, x2, y2 = (float(v) for v in face_box[:4])
    bbox_xywh = [int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))]
    try:
        shape = session.get_inputs()[0].shape
        in_h = int(shape[2]) if isinstance(shape[2], int) else 80
        in_w = int(shape[3]) if isinstance(shape[3], int) else 80
    except (IndexError, TypeError, ValueError):
        in_h, in_w = 80, 80
    face = _minifas_crop_face(img, bbox_xywh, scale, in_w, in_h)
    blob = np.transpose(face.astype(np.float32), (2, 0, 1))[np.newaxis, ...]
    logits = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: blob})[0]
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
