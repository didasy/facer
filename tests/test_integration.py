"""Local-weight integration tests. Photos and weights are never packaged."""
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILENAMES = ("retinaface_r34.onnx", "edgeface_base.onnx", "minifasnetv2.onnx", "face_attrib_net.onnx")
PHOTOS = sorted((ROOT / "benchmark/testdata").glob("*.jpg"))
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def facer():
    from facer import Facer, FacerConfig
    if not all((ROOT / "models" / name).is_file() for name in FILENAMES):
        pytest.skip("local model weights absent")
    if not PHOTOS:
        pytest.skip("local benchmark photos absent")
    f = Facer(FacerConfig(model_dir=str(ROOT / "models")))
    yield f
    f.close()


@pytest.mark.parametrize("path", PHOTOS or [None], ids=[f"photo-{i}" for i in range(len(PHOTOS))] or ["absent"])
def test_detect_align_self_match(facer, path):
    from facer import VerificationResult
    faces = facer.detect_faces(path)
    assert len(faces) == 1
    _, box, kps = faces[0]
    assert box.shape == (4,)
    assert kps.shape == (5, 2)
    aligned = facer.align_face(path, kps)
    assert aligned.shape == (112, 112, 3)
    assert facer.embed(aligned).size == 512
    result = facer.verify(path, path.read_bytes())
    assert isinstance(result, VerificationResult)
    assert result.verified
    assert result.similarity_score == 1.0
    assert 0 <= result.liveness_score <= 1


def test_real_http_contract(facer, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(main, "facer", facer)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        data = PHOTOS[0].read_bytes()
        response = client.post("/api/v1/verify", files={"target_image": ("target.jpg", data), "reference_image": ("reference.jpg", data)})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"verified", "similarity_score", "liveness_score", "processing_time_ms", "quality", "occlusions"}
        assert set(body["quality"]) == {"blur_score", "is_blurry", "luminance_score", "is_dark"}
        assert set(body["occlusions"]) == {"sunglasses_detected", "eyeglasses_detected", "eye_closed_detected", "mask_detected", "hat_detected", "veil_detected"}
        assert body["similarity_score"] == 1.0


def test_http_error_and_concurrency_mapping(facer, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from facer import InvalidImageError, NoFaceError, CropOutOfBoundsError, ModelNotLoadedError
    from app import main
    original_semaphore = main._verify_semaphore
    original_facer = main.facer

    class StubFacer:
        config = facer.config

        def load_models(self):
            pass

        def close(self):
            pass

        def verify(self, target, reference):
            raise self.error

    stub = StubFacer()
    stub.error = NoFaceError("unused")
    monkeypatch.setattr(main, "facer", stub)
    main._verify_semaphore = asyncio.Semaphore(1)

    async def _hold():
        await main._verify_semaphore.acquire()

    asyncio.run(_hold())
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/v1/verify", files={"target_image": ("t.jpg", b"x"), "reference_image": ("r.jpg", b"x")})
            assert response.status_code == 429
            assert response.headers["Retry-After"] == "1"
    finally:
        main._verify_semaphore.release()
        main._verify_semaphore = original_semaphore
        monkeypatch.setattr(main, "facer", original_facer)

    for error, expected_status in (
        (InvalidImageError("bad"), 400),
        (NoFaceError("none"), 422),
        (CropOutOfBoundsError("outside"), 422),
        (ModelNotLoadedError("weights"), 503),
    ):
        stub = StubFacer()
        stub.error = error
        monkeypatch.setattr(main, "facer", stub)
        with TestClient(main.app) as client:
            response = client.post("/api/v1/verify", files={"target_image": ("t.jpg", b"x"), "reference_image": ("r.jpg", b"x")})
            assert response.status_code == expected_status
            assert response.json()["detail"] == str(error)
