import asyncio
import os
import time
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Initialize FastAPI with OpenAPI metadata for Swagger UI
app = FastAPI(
    title="Biometric Verification API (SCRFD 10G + EdgeFace Base)",
    description="High-accuracy, commercially permissive face verification pipeline powered by ONNX Runtime.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
SCRFD_PATH = os.path.join(MODEL_DIR, "det_10g.onnx")
EDGEFACE_PATH = os.path.join(MODEL_DIR, "edgeface_base.onnx")
MINIFAS_PATH = os.path.join(MODEL_DIR, "minifasnetv2.onnx")
FACE_ATTRIB_PATH = os.path.join(MODEL_DIR, "face_attrib_net.onnx")

# --- Tunables (env-configurable, calibrated 2026-09-15 on Lena @512px: ------
# --- sharp ROI=553 / slight-blur ROI=53 / heavy-blur ROI=4-16) ---------------
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "25.0"))  # face-ROI Laplacian variance floor (raw scale)
BLUR_SCORE_MAX = float(os.getenv("BLUR_SCORE_MAX", "250.0"))  # raw variance mapping to blur_score 0.0 (sharp); raw 0 maps to 1.0 (max blur)
LUMINANCE_DARK_MIN = 45.0 / 255.0   # rescaled from 45.0 on the 0-255 gray scale
LUMINANCE_DARK_MAX = 225.0 / 255.0  # rescaled from 225.0 on the 0-255 gray scale
DET_THRESHOLD = float(os.getenv("DET_THRESHOLD", "0.5"))     # SCRFD face score threshold
DET_INPUT_SIZE = int(os.getenv("DET_INPUT_SIZE", "640"))     # SCRFD square input
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.40"))
FACE_PAD_RATIO = float(os.getenv("FACE_PAD_RATIO", "0.25"))   # padding around face box for blur ROI
ATTRIB_THRESHOLD = float(os.getenv("ATTRIB_THRESHOLD", "0.5"))  # FaceAttribNet per-head probability gate
EYE_CLOSED_THRESHOLD = float(os.getenv("EYE_CLOSED_THRESHOLD", "0.5"))  # mean(left_eye_open, right_eye_open) floor
MINIFAS_SCALE = float(os.getenv("MINIFAS_SCALE", "2.7"))  # MiniFASNetV2 crop expand factor


def _max_queued_requests() -> int:
    try:
        value = int(os.getenv("MAX_QUEUED_REQUESTS", "10"))
    except ValueError:
        return 10
    return value if value > 0 else 10


MAX_QUEUED_REQUESTS = _max_queued_requests()  # max in-flight POST /api/v1/verify per process
_verify_semaphore = asyncio.Semaphore(MAX_QUEUED_REQUESTS)


class VerifyConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != "/api/v1/verify":
            return await call_next(request)
        # Non-blocking acquire: wait_for(acquire(), timeout=0) always times out
        # (verified on 3.12: ensure_future needs a loop tick), so guard with
        # locked() instead. No await between check and acquire, so no race.
        if _verify_semaphore.locked():
            return JSONResponse(
                status_code=429,
                content={"detail": "Server busy: too many queued verification requests, retry shortly"},
                headers={"Retry-After": "1"},
            )
        await _verify_semaphore.acquire()
        try:
            return await call_next(request)
        finally:
            _verify_semaphore.release()


app.add_middleware(VerifyConcurrencyLimitMiddleware)

# Standard 112x112 canonical reference landmarks for similarity warping
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

sessions = {}

@app.on_event("startup")
def load_onnx_sessions():
    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CPUExecutionProvider"]

    if os.path.exists(SCRFD_PATH):
        sessions["detector"] = ort.InferenceSession(SCRFD_PATH, opts, providers=providers)
    if os.path.exists(EDGEFACE_PATH):
        sessions["recognizer"] = ort.InferenceSession(EDGEFACE_PATH, opts, providers=providers)
    if os.path.exists(MINIFAS_PATH):
        sessions["liveness"] = ort.InferenceSession(MINIFAS_PATH, opts, providers=providers)
    if os.path.exists(FACE_ATTRIB_PATH):
        sessions["attribute"] = ort.InferenceSession(FACE_ATTRIB_PATH, opts, providers=providers)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class QualityCheckResult(BaseModel):
    blur_score: float = Field(..., description="Normalized face-ROI blurriness 0-1 (1 - min(raw/BLUR_SCORE_MAX, 1.0); is_blurry above 1 - BLUR_THRESHOLD/BLUR_SCORE_MAX)")
    is_blurry: bool
    luminance_score: float = Field(..., description="Normalized mean gray intensity 0-1 (0.176 - 0.882 required)")
    is_dark: bool

class OcclusionResult(BaseModel):
    sunglasses_detected: bool
    eyeglasses_detected: bool
    eye_closed_detected: bool
    mask_detected: bool
    hat_detected: Optional[bool] = None
    veil_detected: Optional[bool] = None

class VerificationResponse(BaseModel):
    verified: bool = Field(..., description="True if target matches reference identity")
    similarity_score: float = Field(..., description="Cosine similarity clamped to 0-1 (EdgeFace Base threshold >= 0.40)")
    liveness_score: float = Field(..., description="Anti-spoofing score (threshold >= 0.90)")
    processing_time_ms: float = Field(..., description="End-to-end server processing time in milliseconds (detect → match)")
    quality: QualityCheckResult
    occlusions: OcclusionResult

# ---------------------------------------------------------------------------
# Preprocessing & Inference Helpers
# ---------------------------------------------------------------------------
def bytes_to_mat(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding")
    return img

def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness of a BGR crop: variance of the Laplacian on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_blur(raw: float) -> float:
    """Map raw Laplacian variance [0, inf) to a 0-1 blurriness score."""
    return 1.0 - min(raw / BLUR_SCORE_MAX, 1.0) if BLUR_SCORE_MAX > 0 else 1.0


def normalize_luminance(mean_gray: float) -> float:
    """Map mean gray intensity 0-255 to a 0-1 brightness score."""
    return min(max(mean_gray / 255.0, 0.0), 1.0)


def evaluate_quality(img: np.ndarray, face_box: np.ndarray | None = None) -> QualityCheckResult:
    """Non-blocking quality report. Blur is measured on the padded face ROI
    (falls back to the whole image when no face was detected). Both scores
    are normalized to 0-1."""
    h, w = img.shape[:2]
    if face_box is not None:
        x1, y1, x2, y2 = (float(v) for v in face_box)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        x1 = max(0, int(x1 - bw * FACE_PAD_RATIO))
        y1 = max(0, int(y1 - bh * FACE_PAD_RATIO))
        x2 = min(w, int(x2 + bw * FACE_PAD_RATIO))
        y2 = min(h, int(y2 + bh * FACE_PAD_RATIO))
        roi = img[y1:y2, x1:x2]
        raw_blur = laplacian_variance(roi) if roi.size else laplacian_variance(img)
    else:
        raw_blur = laplacian_variance(img)
    blur_score = normalize_blur(raw_blur)
    blur_ceiling = 1.0 - BLUR_THRESHOLD / BLUR_SCORE_MAX if BLUR_SCORE_MAX > 0 else 1.0
    luminance_score = normalize_luminance(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))

    return QualityCheckResult(
        blur_score=round(blur_score, 4),
        is_blurry=blur_score > blur_ceiling,
        luminance_score=round(luminance_score, 4),
        is_dark=luminance_score < LUMINANCE_DARK_MIN or luminance_score > LUMINANCE_DARK_MAX
    )


# ---------------------------------------------------------------------------
# SCRFD-10G Face Detection (insightface layout: 3 strides x {score, bbox, kps})
# ---------------------------------------------------------------------------
_SCRFD_STRIDES = (8, 16, 32)
_SCRFD_ANCHORS = 2


def _scrfd_anchors(size: int) -> list[np.ndarray]:
    grids = []
    for stride in _SCRFD_STRIDES:
        n = size // stride
        ax = np.stack(np.meshgrid(np.arange(n), np.arange(n)), axis=-1).astype(np.float32) * stride
        grids.append(np.tile(ax.reshape(-1, 2)[:, None, :], (1, _SCRFD_ANCHORS, 1)).reshape(-1, 2))
    return grids


def detect_faces(img: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Run SCRFD-10G. Returns [(confidence, box_xyxy, landmarks_5x2)] best-first."""
    session = sessions["detector"]
    h0, w0 = img.shape[:2]
    size = DET_INPUT_SIZE
    rgb = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis, ...]
    outs = session.run(None, {session.get_inputs()[0].name: blob})

    boxes_all, confs_all, kps_all = [], [], []
    for i, stride in enumerate(_SCRFD_STRIDES):
        anchors = _scrfd_anchors(size)[i]
        sc = outs[i].ravel()
        deltas = outs[i + 3].reshape(-1, 4) * stride
        kps = outs[i + 6].reshape(-1, 5, 2) * stride + anchors[:, None, :]
        keep = sc > DET_THRESHOLD
        anchors, sc, deltas, kps = anchors[keep], sc[keep], deltas[keep], kps[keep]
        boxes_all.append(np.column_stack([
            anchors[:, 0] - deltas[:, 0], anchors[:, 1] - deltas[:, 1],
            anchors[:, 0] + deltas[:, 2], anchors[:, 1] + deltas[:, 3],
        ]))
        confs_all.append(sc)
        kps_all.append(kps)
    if not confs_all or sum(len(c) for c in confs_all) == 0:
        return []
    boxes_all = np.vstack(boxes_all)
    confs_all = np.concatenate(confs_all)
    kps_all = np.vstack(kps_all)

    keep_idx = cv2.dnn.NMSBoxes(boxes_all.tolist(), confs_all.tolist(), DET_THRESHOLD, 0.4)
    sx, sy = w0 / size, h0 / size
    faces = []
    for j in np.asarray(keep_idx).ravel():
        x1, y1, x2, y2 = boxes_all[j]
        faces.append((
            float(confs_all[j]),
            np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32),
            (kps_all[j] * np.array([sx, sy], dtype=np.float32)).astype(np.float32),
        ))
    return sorted(faces, key=lambda f: -f[0])

def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warps and crops face to 112x112 using 5 facial landmarks."""
    tfm, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
    return cv2.warpAffine(img, tfm, (112, 112))

def extract_edgeface_embedding(aligned_img: np.ndarray) -> np.ndarray:
    """Runs EdgeFace Base ONNX inference to output a normalized 512D vector."""
    rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5  # Normalize [-1, 1]
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    session = sessions["recognizer"]
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: blob})[0]
    
    # L2 Normalization
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(max(0.0, min(1.0, np.dot(emb1.flatten(), emb2.flatten()))))

# ---------------------------------------------------------------------------
# FaceAttribNet Attributes (yakhyo/face-attribute, FaceAttribNet @128x128)
# Output order: left_eye_open, right_eye_open, eyeglasses, mask, sunglasses.
# Five independent binary heads: threshold each, never argmax. Mean/std norm
# is baked into the graph, so the blob stays RGB [0, 1] with no 2nd normalize.
# ---------------------------------------------------------------------------
_ATTRIB_INPUT_SIZE = (128, 128)  # (h, w)


def _attrib_crop_face(img: np.ndarray, bbox: np.ndarray, margin: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if margin:
        dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bounding box {bbox[:4]} does not overlap the {w}x{h} image")
    return img[y1:y2, x1:x2]


def _attrib_resize_pad(crop: np.ndarray, size: tuple[int, int] = _ATTRIB_INPUT_SIZE) -> np.ndarray:
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
    padded = _attrib_resize_pad(crop, _ATTRIB_INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def predict_face_attributes(img: np.ndarray, face_box: np.ndarray) -> tuple[float, float, float, float, float]:
    session = sessions["attribute"]
    blob = _attrib_blob(_attrib_crop_face(img, face_box))
    out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: blob})[0]
    probs = np.squeeze(out).astype(float).tolist()
    return (float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]), float(probs[4]))


def build_occlusion_result(img: np.ndarray, face_box: np.ndarray) -> OcclusionResult:
    left_open, right_open, eyeglasses, mask, sunglasses = predict_face_attributes(img, face_box)
    eye_open_mean = (left_open + right_open) / 2.0
    sunglasses_detected = bool(sunglasses > ATTRIB_THRESHOLD)
    return OcclusionResult(
        sunglasses_detected=sunglasses_detected,
        eyeglasses_detected=bool(eyeglasses > ATTRIB_THRESHOLD),
        eye_closed_detected=False if sunglasses_detected else bool(eye_open_mean < EYE_CLOSED_THRESHOLD),
        mask_detected=bool(mask > ATTRIB_THRESHOLD),
        hat_detected=None,
        veil_detected=None,
    )

# ---------------------------------------------------------------------------
# MiniFASNetV2 Liveness (yakhyo/face-anti-spoofing, scale 2.7, BGR raw input)
# ---------------------------------------------------------------------------


def _minifas_crop_face(img: np.ndarray, bbox_xywh: list[int], scale: float, out_w: int, out_h: int) -> np.ndarray:
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
    return cv2.resize(cropped, (out_w, out_h))


def predict_liveness(img: np.ndarray, face_box: np.ndarray) -> float:
    """Anti-spoof score P(real) in [0, 1] from MiniFASNetV2 softmax logits."""
    session = sessions["liveness"]
    x1, y1, x2, y2 = (float(v) for v in face_box[:4])
    bbox_xywh = [int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))]
    try:
        shape = session.get_inputs()[0].shape
        in_h = int(shape[2]) if isinstance(shape[2], int) else 80
        in_w = int(shape[3]) if isinstance(shape[3], int) else 80
    except (IndexError, TypeError, ValueError):
        in_h, in_w = 80, 80
    face = _minifas_crop_face(img, bbox_xywh, MINIFAS_SCALE, in_w, in_h)
    blob = np.transpose(face.astype(np.float32), (2, 0, 1))[np.newaxis, ...]
    logits = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: blob})[0]
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])

# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    models = {
        "detector": "detector" in sessions,
        "recognizer": "recognizer" in sessions,
        "liveness": "liveness" in sessions,
        "attribute": "attribute" in sessions,
    }
    loaded = all(models.values())
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded, "models": models}

@app.post(
    "/api/v1/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="1:1 Verification via SCRFD 10G & EdgeFace Base",
    tags=["Biometric Engine"],
    responses={
        429: {
            "description": "Too Many Requests — verify concurrency limit exceeded; retry after 1s",
            "headers": {"Retry-After": {"schema": {"type": "string"}}},
            "content": {
                "application/json": {
                    "example": {"detail": "Server busy: too many queued verification requests, retry shortly"}
                }
            },
        }
    },
)
async def verify_identity(
    target_image: UploadFile = File(..., description="Live camera frame"),
    reference_image: UploadFile = File(..., description="Reference photo")
):
    start = time.perf_counter()
    if "detector" not in sessions or "recognizer" not in sessions:
        raise HTTPException(status_code=503, detail="ONNX models not loaded")

    img_target = bytes_to_mat(await target_image.read())
    img_ref = bytes_to_mat(await reference_image.read())

    # 1. Detect faces FIRST so quality is measured on the face, not background
    target_faces = detect_faces(img_target)
    ref_faces = detect_faces(img_ref)
    if not target_faces:
        raise HTTPException(status_code=422, detail="No face detected in target image")
    if not ref_faces:
        raise HTTPException(status_code=422, detail="No face detected in reference image")
    _, target_box, target_kps = target_faces[0]
    _, ref_box, ref_kps = ref_faces[0]

    # 2. Quality report (non-blocking: reported in response, never 422)
    quality = evaluate_quality(img_target, target_box)

    # 3. Occlusion attributes + passive liveness (fail closed when weights missing)
    missing = [name for name in ("attribute", "liveness") if name not in sessions]
    if missing:
        raise HTTPException(status_code=503, detail=f"ONNX models not loaded: {', '.join(missing)}")
    try:
        occlusions = build_occlusion_result(img_target, target_box)
        liveness_score = predict_liveness(img_target, target_box)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 5. Face Alignment & EdgeFace Base Embedding Extraction
    aligned_target = align_face(img_target, target_kps)
    aligned_ref = align_face(img_ref, ref_kps)

    emb_target = extract_edgeface_embedding(aligned_target)
    emb_ref = extract_edgeface_embedding(aligned_ref)

    similarity = calculate_cosine_similarity(emb_target, emb_ref)

    # Threshold for EdgeFace Base 512D Cosine Match
    verified = bool(similarity >= VERIFY_THRESHOLD)

    return VerificationResponse(
        verified=verified,
        similarity_score=round(similarity, 4),
        liveness_score=round(liveness_score, 4),
        processing_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
        quality=quality,
        occlusions=occlusions
    )
