"""facer: CPU face verification with explicit weights and no server dependency."""
from .config import FacerConfig
from .errors import CropOutOfBoundsError, InvalidImageError, ModelNotLoadedError, NoFaceError
from .results import Occlusions, Quality, VerificationResult
from .verification import Facer, verify

__version__ = "2.1.0"
__all__ = [
    "Facer", "FacerConfig", "verify", "InvalidImageError", "NoFaceError",
    "CropOutOfBoundsError", "ModelNotLoadedError", "VerificationResult", "Quality", "Occlusions",
]
