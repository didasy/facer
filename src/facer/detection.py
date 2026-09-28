"""RetinaFace-R34 inference with NumPy priors and offset decoding (no torch)."""
import cv2
import numpy as np

from .config import FacerConfig

_RF_STEPS = (8, 16, 32)
_RF_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_RF_VARIANCE = (0.1, 0.2)
_RF_MEAN = (104.0, 117.0, 123.0)
_RF_PRE_NMS_TOPK = 5000
_RF_NMS_THRESH = 0.4
_RF_POST_NMS_TOPK = 750


def _retina_priors(h: int, w: int) -> np.ndarray:
    """Prior boxes (cx, cy, s_kx, s_ky), normalized for a h*w input.

    Mirrors upstream PriorBox.generate_anchors with clip=False.
    """
    import math
    from itertools import product
    anchors = []
    for k, step in enumerate(_RF_STEPS):
        map_h, map_w = math.ceil(h / step), math.ceil(w / step)
        for i, j in product(range(map_h), range(map_w)):
            for min_size in _RF_MIN_SIZES[k]:
                s_kx = min_size / w
                s_ky = min_size / h
                cx = (j + 0.5) * step / w
                cy = (i + 0.5) * step / h
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


def _decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Undo train-time offset encoding. Returns normalized xyxy boxes."""
    v0, v1 = _RF_VARIANCE
    cxcy = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
    wh = priors[:, 2:] * np.exp(loc[:, 2:] * v1)
    boxes = np.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2
    boxes[:, 2:] = cxcy + wh / 2
    return boxes


def _decode_landmarks(pred: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Decode five landmarks per prior: (N,10) to normalized (N,5,2)."""
    v0 = _RF_VARIANCE[0]
    pred = pred.reshape(pred.shape[0], 5, 2)
    lms = priors[:, None, :2] + pred * v0 * priors[:, None, 2:]
    return lms.reshape(pred.shape[0], 5, 2)


def detect_faces(img: np.ndarray, session, config: FacerConfig) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return [(confidence, box_xyxy, landmarks_5x2)] best-first."""
    h0, w0 = img.shape[:2]
    # Downscale-only resize: longest side capped, no upscale/stretch.
    scale = min(1.0, config.det_input_size / max(h0, w0))
    rh, rw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else img
    blob = (resized.astype(np.float32) - np.array(_RF_MEAN, dtype=np.float32)).transpose(2, 0, 1)[np.newaxis, ...]
    outs = session.run(None, {session.get_inputs()[0].name: blob})
    names = [o.name for o in session.get_outputs()]
    if len(outs) != 3 or not all(n in ("loc", "conf", "landmarks") for n in names):
        raise RuntimeError(f"Unexpected RetinaFace outputs: {names}")
    loc = np.asarray(outs[names.index("loc")]).squeeze(0)
    conf = np.asarray(outs[names.index("conf")]).squeeze(0)
    lms_pred = np.asarray(outs[names.index("landmarks")]).squeeze(0)
    priors = _retina_priors(rh, rw)
    boxes = _decode_boxes(loc, priors) * np.array([rw, rh, rw, rh], dtype=np.float32)
    landmarks = _decode_landmarks(lms_pred, priors) * np.array([rw, rh], dtype=np.float32)
    scores = conf[:, 1]
    keep = scores > config.det_threshold
    boxes, landmarks, scores = boxes[keep], landmarks[keep], scores[keep]
    if len(scores) == 0:
        return []
    order = np.argsort(-scores)[:_RF_PRE_NMS_TOPK]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
    # Preserve the existing server's NMS behavior during extraction.
    nms_idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), config.det_threshold, _RF_NMS_THRESH)
    if len(nms_idx) == 0:
        return []
    keep_idx = np.asarray(nms_idx).ravel()[:_RF_POST_NMS_TOPK]
    inv = 1.0 / scale
    faces = []
    for j in keep_idx:
        x1, y1, x2, y2 = boxes[j] * inv
        faces.append((
            float(scores[j]),
            np.array([x1, y1, x2, y2], dtype=np.float32),
            (landmarks[j] * inv).astype(np.float32),
        ))
    faces = [
        (c, np.array([max(0.0, min(b[0], w0)), max(0.0, min(b[1], h0)),
                      max(0.0, min(b[2], w0)), max(0.0, min(b[3], h0))], dtype=np.float32), l)
        for c, b, l in faces
    ]
    return sorted(faces, key=lambda f: -f[0])
