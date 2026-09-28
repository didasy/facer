"""Plain result dataclasses shared by library callers and the HTTP wrapper."""
from dataclasses import dataclass


@dataclass
class Quality:
    """Non-blocking quality report: blur and luminance scores are in [0, 1]."""

    blur_score: float
    is_blurry: bool
    luminance_score: float
    is_dark: bool


@dataclass
class Occlusions:
    """Independent FaceAttribNet flags; unsupported heads remain None."""

    sunglasses_detected: bool
    eyeglasses_detected: bool
    eye_closed_detected: bool
    mask_detected: bool
    hat_detected: bool | None = None
    veil_detected: bool | None = None


@dataclass
class VerificationResult:
    """1:1 identity match. Quality and liveness do not gate verified."""

    verified: bool
    similarity_score: float
    liveness_score: float
    processing_time_ms: float
    quality: Quality
    occlusions: Occlusions
