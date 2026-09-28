"""Public face-verification pipeline and per-instance lazy ONNX sessions."""
from os import PathLike
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

from . import alignment, attributes, detection, liveness, quality, recognition
from .config import FacerConfig
from .errors import InvalidImageError, ModelNotLoadedError, NoFaceError
from .results import Occlusions, Quality, VerificationResult
from .weights import MODEL_FILES

ImageInput = str | PathLike[str] | bytes | np.ndarray


def _read_image(image: ImageInput) -> np.ndarray:
    """Read encoded images or validate uint8 BGR arrays without changing channels."""
    if isinstance(image, (str, PathLike)):
        try:
            image = Path(image).read_bytes()
        except OSError as exc:
            raise InvalidImageError("Cannot read image path") from exc
    if isinstance(image, bytes):
        if not image:
            raise InvalidImageError("Invalid image encoding")
        try:
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise InvalidImageError("Invalid image encoding") from exc
        if image is None:
            raise InvalidImageError("Invalid image encoding")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise InvalidImageError("Expected a nonempty uint8 BGR image (HxWx3)")
    return image


class Facer:
    """CPU-only 1:1 face verification with reusable, lazily loaded model sessions.

    Inputs accept paths, encoded bytes, or uint8 BGR arrays. Construction does
    not load or download weights. close() releases cached sessions; later calls
    may lazily load them again. Do not close while inference is in progress.
    """

    def __init__(self, config: FacerConfig | None = None):
        self.config = config if config is not None else FacerConfig.from_env()
        self._sessions = {}
        self._load_lock = threading.Lock()

    @property
    def models_loaded(self) -> dict[str, bool]:
        """Snapshot of session load state without triggering inference or loading."""
        with self._load_lock:
            return {name: name in self._sessions for name in MODEL_FILES}

    def _session(self, name: str):
        with self._load_lock:
            if name not in self._sessions:
                path = Path(self.config.model_dir) / MODEL_FILES[name]
                if not path.is_file():
                    raise ModelNotLoadedError(f"ONNX models not loaded: {name}")
                opts = ort.SessionOptions()
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                try:
                    self._sessions[name] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                except Exception as exc:
                    raise ModelNotLoadedError(f"ONNX models not loaded: {name}") from exc
            return self._sessions[name]

    def load_models(self) -> None:
        """Load available models for server startup; missing weights stay unloaded."""
        for name, filename in MODEL_FILES.items():
            if (Path(self.config.model_dir) / filename).is_file():
                self._session(name)

    def close(self) -> None:
        """Release this instance's cached ONNX sessions. Safe to call repeatedly."""
        with self._load_lock:
            self._sessions.clear()

    def __enter__(self) -> "Facer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def detect_faces(self, image: ImageInput) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Detect faces, returning (score, xyxy box, 5x2 landmarks) best-first."""
        img = _read_image(image)
        return detection.detect_faces(img, self._session("detector"), self.config)

    def align_face(self, image: ImageInput, landmarks: np.ndarray) -> np.ndarray:
        """Align a face to a 112x112 BGR image using five landmarks."""
        return alignment.align_face(_read_image(image), landmarks)

    def embed(self, image: ImageInput) -> np.ndarray:
        """Embed an already-aligned 112x112 BGR face as a normalized 512D vector."""
        img = _read_image(image)
        if img.shape[:2] != (112, 112):
            raise InvalidImageError("Expected an aligned 112x112 face")
        return recognition.embed(img, self._session("recognizer"))

    def quality(self, image: ImageInput, box: np.ndarray | None = None) -> Quality:
        """Report face-ROI blur and whole-image luminance; never gate verification."""
        return quality.evaluate_quality(_read_image(image), box, self.config)

    def attributes(self, image: ImageInput, box: np.ndarray) -> Occlusions:
        """Report independently thresholded FaceAttribNet occlusion heads."""
        img = _read_image(image)
        return attributes.build_occlusion_result(img, box, self._session("attribute"), self.config)

    def liveness(self, image: ImageInput, box: np.ndarray) -> float:
        """Return MiniFASNetV2 P(real), without applying a pass/fail threshold."""
        img = _read_image(image)
        return liveness.predict_liveness(img, box, self._session("liveness"), self.config.minifas_scale)

    def verify(self, target: ImageInput, reference: ImageInput) -> VerificationResult:
        """Compare the best face in each input. Only similarity gates verified."""
        start = time.perf_counter()
        try:
            self._session("detector")
            self._session("recognizer")
        except ModelNotLoadedError as exc:
            raise ModelNotLoadedError("ONNX models not loaded") from exc
        img_target, img_ref = _read_image(target), _read_image(reference)
        target_faces = self.detect_faces(img_target)
        ref_faces = self.detect_faces(img_ref)
        if not target_faces:
            raise NoFaceError("No face detected in target image")
        if not ref_faces:
            raise NoFaceError("No face detected in reference image")
        _, target_box, target_kps = target_faces[0]
        _, _, ref_kps = ref_faces[0]
        report = self.quality(img_target, target_box)
        missing = []
        for name in ("attribute", "liveness"):
            try:
                self._session(name)
            except ModelNotLoadedError:
                missing.append(name)
        if missing:
            raise ModelNotLoadedError(f"ONNX models not loaded: {', '.join(missing)}")
        occlusions = self.attributes(img_target, target_box)
        liveness_score = self.liveness(img_target, target_box)
        aligned_target = self.align_face(img_target, target_kps)
        aligned_ref = self.align_face(img_ref, ref_kps)
        similarity = recognition.cosine_similarity(self.embed(aligned_target), self.embed(aligned_ref))
        return VerificationResult(
            verified=bool(similarity >= self.config.verify_threshold),
            similarity_score=round(similarity, 4),
            liveness_score=round(liveness_score, 4),
            processing_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            quality=report,
            occlusions=occlusions,
        )


def verify(target: ImageInput, reference: ImageInput, config: FacerConfig | None = None) -> VerificationResult:
    """One-shot verification; use a reusable Facer instance for repeated calls."""
    with Facer(config) as facer:
        return facer.verify(target, reference)
