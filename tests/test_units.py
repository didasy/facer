"""Weight-free checks for the public library and preserved model math."""
import importlib.util


def test_library_exists():
    assert importlib.util.find_spec("facer") is not None, "src/facer library is missing"


def test_priors_and_decode():
    import numpy as np
    from facer.detection import _retina_priors, _decode_boxes, _decode_landmarks
    priors = _retina_priors(640, 640)
    assert priors.shape == (16800, 4)
    boxes = _decode_boxes(np.zeros_like(priors), priors)
    np.testing.assert_allclose((boxes[:, :2] + boxes[:, 2:]) / 2, priors[:, :2], atol=1e-7)
    np.testing.assert_allclose(boxes[:, 2:] - boxes[:, :2], priors[:, 2:], atol=1e-7)
    landmarks = _decode_landmarks(np.zeros((16800, 10), dtype=np.float32), priors)
    np.testing.assert_allclose(landmarks, np.repeat(priors[:, None, :2], 5, axis=1))


def test_quality_mapping():
    import numpy as np
    from facer import Facer, FacerConfig, Quality
    from facer.quality import normalize_blur, normalize_luminance
    assert normalize_blur(0) == 1
    assert normalize_blur(125) == 0.5
    assert normalize_blur(500) == 0
    assert normalize_blur(10, 0) == 1
    assert normalize_luminance(-10) == 0
    assert normalize_luminance(255) == 1
    assert normalize_luminance(300) == 1
    f = Facer(FacerConfig())
    result = f.quality(np.full((20, 20, 3), 128, dtype=np.uint8))
    assert isinstance(result, Quality)
    assert result.is_blurry and not result.is_dark
    assert result.luminance_score == 0.502
    assert not any(f.models_loaded.values())


def test_cosine_clamp():
    import numpy as np
    from facer.recognition import cosine_similarity
    assert cosine_similarity(np.array([1.00001]), np.array([1.0])) == 1
    assert cosine_similarity(np.array([-1.0]), np.array([1.0])) == 0
    assert cosine_similarity(np.array([0.5]), np.array([1.0])) == 0.5


def test_config_from_env(monkeypatch):
    from facer import FacerConfig
    monkeypatch.setenv("MODEL_DIR", "/tmp/facer-test-weights")
    monkeypatch.setenv("VERIFY_THRESHOLD", "0.73")
    monkeypatch.setenv("DET_INPUT_SIZE", "320")
    monkeypatch.setenv("BLUR_SCORE_MAX", "500")
    monkeypatch.setenv("MAX_QUEUED_REQUESTS", "invalid")
    config = FacerConfig.from_env()
    assert config.model_dir == "/tmp/facer-test-weights"
    assert config.verify_threshold == 0.73
    assert config.det_input_size == 320
    assert config.blur_score_max == 500
    assert config.max_queued_requests == 10
    assert FacerConfig().verify_threshold == 0.40


def test_inputs_and_errors(tmp_path):
    import cv2
    import numpy as np
    import pytest
    from facer import Facer, FacerConfig, InvalidImageError, ModelNotLoadedError
    f = Facer(FacerConfig(model_dir=str(tmp_path)))
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    path = tmp_path / "image.png"
    path.write_bytes(encoded)
    assert f.quality(image) == f.quality(encoded) == f.quality(path) == f.quality(str(path))
    for invalid in (b"", b"bad", np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((5, 5)), tmp_path / "missing"):
        with pytest.raises(InvalidImageError):
            f.quality(invalid)
    with pytest.raises(ModelNotLoadedError):
        f.detect_faces(image)
    assert not any(f.models_loaded.values())


def test_lazy_session_lifecycle(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from facer import Facer, FacerConfig, ModelNotLoadedError, verify
    from facer import verification
    from facer.weights import MODEL_FILES
    for name in MODEL_FILES.values():
        (tmp_path / name).write_bytes(b"fake weight")
    created = []

    def session(path, options, providers):
        assert providers == ["CPUExecutionProvider"]
        result = SimpleNamespace(path=path)
        created.append(result)
        return result

    monkeypatch.setattr(verification.ort, "InferenceSession", session)
    f = Facer(FacerConfig(model_dir=str(tmp_path)))
    assert not created
    assert f._session("detector") is f._session("detector")
    assert len(created) == 1
    f.load_models()
    assert len(created) == 4 and all(f.models_loaded.values())
    f.close()
    f.close()
    assert not any(f.models_loaded.values())
    with f:
        f._session("recognizer")
    assert not any(f.models_loaded.values())
    monkeypatch.setattr(Facer, "verify", lambda self, target, reference: self._session("detector"))
    closed = []
    original_close = Facer.close

    def close(self):
        original_close(self)
        closed.append(self.models_loaded)

    monkeypatch.setattr(Facer, "close", close)
    verify(b"target", b"reference", f.config)
    assert len(closed) == 1 and not any(closed[0].values())

    def fail(*args, **kwargs):
        raise RuntimeError("invalid ONNX")

    monkeypatch.setattr(verification.ort, "InferenceSession", fail)
    with pytest.raises(ModelNotLoadedError, match="detector"):
        f._session("detector")


def test_alignment_and_crops():
    import numpy as np
    import pytest
    from facer import Facer, InvalidImageError, CropOutOfBoundsError
    from facer.alignment import REFERENCE_FACIAL_POINTS
    from facer.attributes import _attrib_crop_face, _attrib_blob
    from facer.liveness import _minifas_crop_face
    image = np.full((112, 112, 3), 80, dtype=np.uint8)
    f = Facer()
    assert f.align_face(image, REFERENCE_FACIAL_POINTS).shape == (112, 112, 3)
    with pytest.raises(InvalidImageError):
        f.align_face(image, np.zeros((5, 2)))
    with pytest.raises(InvalidImageError):
        f.embed(image[:50])
    with pytest.raises(CropOutOfBoundsError):
        _attrib_crop_face(image, np.array([200, 200, 220, 220]))
    with pytest.raises(CropOutOfBoundsError):
        _minifas_crop_face(image, [300, 300, 20, 20], 2.7, 80, 80)
    assert _attrib_blob(image).shape == (1, 3, 128, 128)


def test_attribute_heads_and_liveness():
    from types import SimpleNamespace
    import numpy as np
    from facer import FacerConfig
    from facer.attributes import build_occlusion_result
    from facer.liveness import predict_liveness

    class Session:
        def __init__(self, output):
            self.output = np.array([output], dtype=np.float32)

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 80, 80])]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run(self, names, inputs):
            return [self.output]

    image = np.full((112, 112, 3), 128, dtype=np.uint8)
    box = np.array([20, 20, 80, 80])
    occ = build_occlusion_result(image, box, Session([0, 0, 0.9, 0.9, 0.9]), FacerConfig())
    assert occ.sunglasses_detected and occ.eyeglasses_detected and occ.mask_detected
    assert not occ.eye_closed_detected
    occ = build_occlusion_result(image, box, Session([0, 0, 0, 0, 0]), FacerConfig())
    assert occ.eye_closed_detected
    assert abs(predict_liveness(image, box, Session([1000, 1000, 1000])) - 1 / 3) < 1e-6


def test_verification_decision_and_failures(monkeypatch):
    import numpy as np
    import pytest
    from facer import Facer, FacerConfig, NoFaceError, ModelNotLoadedError, Occlusions
    f = Facer(FacerConfig(verify_threshold=0.40004))
    monkeypatch.setattr(f, "_session", lambda name: object())
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    face = (1.0, np.array([1, 1, 100, 100]), np.ones((5, 2)))
    monkeypatch.setattr(f, "detect_faces", lambda img: [face])
    monkeypatch.setattr(f, "align_face", lambda img, kps: img)
    monkeypatch.setattr(f, "attributes", lambda img, box: Occlusions(True, True, True, True))
    monkeypatch.setattr(f, "liveness", lambda img, box: 0.0)
    embeddings = iter([np.array([0.40003]), np.array([1.0])])
    monkeypatch.setattr(f, "embed", lambda img: next(embeddings))
    result = f.verify(image, image)
    assert result.similarity_score == 0.4 and not result.verified
    monkeypatch.setattr(f, "embed", lambda img: np.array([1.0]))
    assert f.verify(image, image).verified  # quality/liveness/occlusions do not gate
    monkeypatch.setattr(f, "detect_faces", lambda img: [])
    with pytest.raises(NoFaceError, match="target"):
        f.verify(image, image)
    faces = iter([[face], []])
    monkeypatch.setattr(f, "detect_faces", lambda img: next(faces))
    with pytest.raises(NoFaceError, match="reference"):
        f.verify(image, image)
    monkeypatch.setattr(f, "detect_faces", lambda img: [face])

    def session(name):
        if name in ("attribute", "liveness"):
            raise ModelNotLoadedError(name)
        return object()

    monkeypatch.setattr(f, "_session", session)
    with pytest.raises(ModelNotLoadedError, match="attribute, liveness"):
        f.verify(image, image)


def test_weight_download_skip_force_and_failure(tmp_path, monkeypatch):
    from io import BytesIO
    import hashlib
    import pytest
    from facer import weights
    content = b"test model content"
    monkeypatch.setattr(weights, "_MINIFAS_SHA256", hashlib.sha256(content).hexdigest())
    calls = []

    def download(url, timeout):
        calls.append(url)
        return BytesIO(content)

    monkeypatch.setattr(weights, "urlopen", download)
    paths = weights.download_weights(tmp_path)
    assert len(calls) == 4
    assert all(path.read_bytes() == content for path in paths.values())
    weights.download_weights(tmp_path)
    assert len(calls) == 4
    weights.download_weights(tmp_path, force=True)
    assert len(calls) == 8

    def fail(url, timeout):
        raise OSError("network unavailable")

    monkeypatch.setattr(weights, "urlopen", fail)
    with pytest.raises(OSError):
        weights.download_weights(tmp_path, force=True)
    assert all(path.read_bytes() == content for path in paths.values())
    assert len(list(tmp_path.iterdir())) == 4
    (tmp_path / "minifasnetv2.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="sha256"):
        weights.download_weights(tmp_path)

