import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from facer import Facer, InvalidImageError, NoFaceError, CropOutOfBoundsError, ModelNotLoadedError

facer = Facer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    facer.load_models()
    try:
        yield
    finally:
        facer.close()


# Initialize FastAPI with OpenAPI metadata for Swagger UI
app = FastAPI(
    title="Biometric Verification API (RetinaFace-R34 + EdgeFace Base)",
    description="High-accuracy face verification pipeline with a fully permissive model stack, powered by ONNX Runtime.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

MAX_QUEUED_REQUESTS = facer.config.max_queued_requests  # max in-flight POST /api/v1/verify per process
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


@app.get("/health", tags=["System"])
async def health_check():
    models = facer.models_loaded
    loaded = all(models.values())
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded, "models": models}


@app.post(
    "/api/v1/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="1:1 Verification via RetinaFace-R34 & EdgeFace Base",
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
    try:
        result = facer.verify(await target_image.read(), await reference_image.read())
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (NoFaceError, CropOutOfBoundsError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VerificationResponse(**asdict(result))
