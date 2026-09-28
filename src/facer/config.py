"""Environment-compatible configuration for facer."""
from dataclasses import dataclass, fields
import os


@dataclass
class FacerConfig:
    """Pipeline tunables. Use from_env() to read the server's environment names."""

    model_dir: str = "./models"
    blur_threshold: float = 25.0
    blur_score_max: float = 250.0
    luminance_dark_min: float = 45.0 / 255.0
    luminance_dark_max: float = 225.0 / 255.0
    det_threshold: float = 0.5
    det_input_size: int = 640
    verify_threshold: float = 0.40
    face_pad_ratio: float = 0.25
    attrib_threshold: float = 0.5
    eye_closed_threshold: float = 0.5
    minifas_scale: float = 2.7
    max_queued_requests: int = 10

    @classmethod
    def from_env(cls) -> "FacerConfig":
        """Read uppercase field names; preserve the server queue-limit fallback.

        Luminance limits were constants in the server, not environment tunables.
        Environment files (.env) are not loaded automatically.
        """
        config = cls()
        for field in fields(config):
            if field.name in ("luminance_dark_min", "luminance_dark_max", "max_queued_requests"):
                continue
            value = os.getenv(field.name.upper())
            if value is not None:
                setattr(config, field.name, type(getattr(config, field.name))(value))
        try:
            queued = int(os.getenv("MAX_QUEUED_REQUESTS", "10"))
        except ValueError:
            queued = 10
        config.max_queued_requests = queued if queued > 0 else 10
        return config
