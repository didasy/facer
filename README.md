# facer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/inference-ONNX_Runtime-green.svg)](https://onnxruntime.ai/)

Self-hosted 1:1 face verification library with an optional FastAPI server.
ONNX Runtime (CPU), Docker-ready. Detect with RetinaFace-R34, embed with EdgeFace-Base,
report liveness with MiniFASNetV2, flag occlusions with FaceAttribNet, and report image quality.

> [!NOTE]
> **Fully permissive stack.** Project code is [Apache-2.0](LICENSE).
> The four model weights are **not** covered by it: fetch them explicitly
> and each keeps its own terms (see [Models](#models) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
> All four are commercial-safe (MIT / BSD-3-Clause / Apache-2.0) — including the
> self-trained RetinaFace-R34 detector, which replaced the earlier
> research-only detector weight (that one was non-commercial only).

## How it works

```text
target.jpg + reference.jpg
        │ POST /api/v1/verify (multipart)
        ▼
RetinaFace-R34 detect (best face each) → face-ROI quality (blur, luminance)
        │
        ├── FaceAttribNet → sunglasses / eyeglasses / closed eyes / mask
        ├── MiniFASNetV2 → liveness score P(real)
        └── EdgeFace-Base 512-D embeddings → cosine similarity → verified?
        ▼
{ verified, similarity_score, liveness_score, quality, occlusions, processing_time_ms }
```

Quality is **reported, never blocking**: poor blur/lighting shows up in
`quality` but won't 422 the request. Missing liveness/attribute weights
fail closed (503).

## Quickstart

Prereqs: Docker + Compose (or Python 3.11), ~250 MB free for weights.

```bash
git clone <your-fork-url> facer && cd facer

# 1. Fetch weights (skips files already present; --force re-downloads)
bash models/download.sh   # verifies MiniFASNetV2 sha256

# 2. Configure (optional — sane defaults built in)
cp .env.example .env      # then edit thresholds if needed

# 3. Run
docker compose up --build
```

Check health, then verify:

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":true,"models":{"detector":true,"recognizer":true,"liveness":true,"attribute":true}}

curl -X POST http://localhost:8000/api/v1/verify \
  -F target_image=@target.jpg \
  -F reference_image=@reference.jpg
```

Interactive docs: `http://localhost:8000/docs` (Swagger), `/redoc`.

Local run without Docker:

```bash
pip install '.[server]'
bash models/download.sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Python library

Python **3.10+**. Install from this checkout (no PyPI release has been published):

```bash
pip install .             # NumPy, OpenCV headless, ONNX Runtime only
pip install '.[server]'  # optional FastAPI server dependencies
```

Weights are **not bundled or downloaded automatically**. Fetch them explicitly:

```python
from facer.weights import download_weights

download_weights("./models")  # skips existing files; force=True replaces them
```

The downloader uses the same yakhyo release assets as `models/download.sh` and
checks the published MiniFASNetV2 SHA256. Keep each model's separate license.

```python
from dataclasses import asdict
from facer import Facer, FacerConfig, verify

# Facer() reads the same environment names/defaults as the server.
with Facer() as f:
    result = f.verify("target.jpg", "reference.jpg")
    print(result.verified, result.similarity_score, result.liveness_score)
    payload = asdict(result)

# Explicit config does not read the environment; defaults include threshold 0.40.
config = FacerConfig(model_dir="./models", verify_threshold=0.40)
result = verify("target.jpg", "reference.jpg", config=config)
```

Reuse `Facer` for repeated calls. Each instance owns its ONNX sessions, loaded
lazily per model; `close()` releases them. Later calls can load them again.
Do not call `close()` during active inference. The one-shot `verify()` closes
its instance after every call. Construction/import never downloads or loads weights.

All image inputs accept `str`/`pathlib.Path` paths, encoded `bytes`, or nonempty
`numpy.ndarray` images with **uint8 BGR, H×W×3** layout (OpenCV convention).
Encoded images are decoded to BGR. Granular methods:

```python
with Facer(config) as f:
    image = "target.jpg"
    faces = f.detect_faces(image)  # [(score, box_xyxy, kps_5x2)], best-first
    if faces:
        _, box, kps = faces[0]
        aligned = f.align_face(image, kps)  # 112x112 BGR
        embedding = f.embed(aligned)      # L2-normalized 512D embedding
        quality = f.quality(image, box)   # box optional; no weights needed
        occlusions = f.attributes(image, box)
        liveness = f.liveness(image, box) # P(real), not a pass/fail gate
```

Results are plain `VerificationResult`, `Quality`, and `Occlusions` dataclasses.
`verified` depends **only** on unrounded similarity ≥ configured threshold.
Quality, attributes, and liveness are reported, not blocking gates.
`facer.recognition.cosine_similarity()` expects L2-normalized embeddings and
clamps their dot product to 0–1.

Catch exported `InvalidImageError`, `NoFaceError`, `CropOutOfBoundsError`, or
`ModelNotLoadedError`. The HTTP wrapper maps them to 400, 422, 422, and 503.
Missing required attribute/liveness weights never silently bypass checks.

### Tests

```bash
pip install '.[server]' pytest httpx
pytest                    # integration tests skip without local weights/photos
pytest -m 'not integration'
```

Integration tests use gitignored `benchmark/testdata/*.jpg`; neither photos nor
weights enter wheel/sdist artifacts. `app/main.py` remains the checkout's server
entry point; the wheel contains only the `facer` library.

## API

### `GET /health`

Returns per-model load state. `status` is `ok` only when all four weights load.

### `POST /api/v1/verify`

Multipart fields: `target_image` (live frame), `reference_image` (reference photo).
In-flight requests per worker capped by `MAX_QUEUED_REQUESTS` (default 10);
overflow returns `429` with a `Retry-After: 1` header.

Success `200` example:

```json
{
  "verified": true,
  "similarity_score": 0.62,
  "liveness_score": 0.98,
  "processing_time_ms": 210.5,
  "quality": {
    "blur_score": 0.2,
    "is_blurry": false,
    "luminance_score": 0.45,
    "is_dark": false
  },
  "occlusions": {
    "sunglasses_detected": false,
    "eyeglasses_detected": false,
    "eye_closed_detected": false,
    "mask_detected": false,
    "hat_detected": null,
    "veil_detected": null
  }
}
```

Field notes:

- `similarity_score`: cosine similarity clamped to 0–1. `verified` = score ≥ `VERIFY_THRESHOLD`.
- `liveness_score`: anti-spoof P(real) in 0–1. Reported only, no server-side gate — apply your own (0.90 is a reasonable start).
- `blur_score`: 0 = sharp, 1 = max blur (face ROI, padded 25%). `is_blurry` trips below the `BLUR_THRESHOLD` floor.
- `hat_detected` / `veil_detected`: always `null` — the attribute model has no such heads; fields exist for forward compatibility.

| Status | Meaning                                                |
| ------ | ------------------------------------------------------ |
| 200    | Verification ran (check `verified`, quality, occlusions) |
| 400    | Undecodable image bytes                                |
| 422    | No face in target/reference, or crop outside image     |
| 429    | Concurrency limit hit — retry after 1 s                |
| 503    | Weights missing at startup                             |

## Configuration

All tunables are env vars (see [.env.example](.env.example)). Code defaults apply
locally; `docker-compose.yml` overrides three of them — set explicit values in
`.env` to avoid surprises.

| Var                  | Code default | Compose fallback | Meaning                                        |
| -------------------- | ------------ | ---------------- | ---------------------------------------------- |
| `MODEL_DIR`            | `./models`     | `/app/models`      | Weight directory                               |
| `BLUR_THRESHOLD`       | `25.0`         | `25.0`             | Face-ROI Laplacian-variance floor (raw scale)  |
| `BLUR_SCORE_MAX`       | `250.0`        | `250.0`            | Raw variance mapping to `blur_score` 0 (sharp)   |
| `FACE_PAD_RATIO`       | `0.25`         | `0.25`             | Padding around face box for blur ROI           |
| `DET_THRESHOLD`        | `0.5`          | `0.5`              | RetinaFace face score gate                     |
| `DET_INPUT_SIZE`       | `640`          | `640`              | max-side cap (px), downscale-only              |
| `VERIFY_THRESHOLD`     | **`0.40`**         | **`0.60`**             | Cosine gate for `verified` — pick one explicitly |
| `ATTRIB_THRESHOLD`     | `0.5`          | `0.6`              | Per-attribute probability gate                 |
| `EYE_CLOSED_THRESHOLD` | `0.5`          | `0.6`              | Mean eye-openness floor                        |
| `MINIFAS_SCALE`        | `2.7`          | `2.7`              | Liveness crop expand factor                    |
| `MAX_QUEUED_REQUESTS`  | `10`           | `10`               | In-flight verifies per worker before 429       |

## Project structure

```text
├── src/facer/           # Reusable pipeline, config, results, explicit downloads
├── app/main.py          # Thin FastAPI wrapper and concurrency middleware
├── pyproject.toml       # Library packaging and optional server dependencies
├── tests/               # Weight-free units and local-weight integration checks
├── models/download.sh   # Weight fetcher (weights never committed)
├── requirements.txt     # Docker runtime dependencies
├── Dockerfile           # python:3.11-slim + OpenCV syslibs
├── docker-compose.yml   # 2 CPU / 2 GB limits, healthcheck, env fallbacks
├── benchmark/verify.js  # k6 load test (needs benchmark/testdata/*.jpg)
├── .env.example         # Tunable template (copy to .env)
├── LICENSE              # Apache-2.0 — project code only
└── THIRD_PARTY_NOTICES.md  # Weights + dependency licenses
```

## Benchmark

[k6](https://k6.io/) load test for `POST /api/v1/verify` (latency, server
`processing_time_ms`, success rate). Needs two real face photos, which are
gitignored and never committed:

```bash
# place photos first:
#   benchmark/testdata/target.jpg     (live frame, same or different identity)
#   benchmark/testdata/reference.jpg  (reference photo)
docker compose up --build   # wait for /health → models_loaded: true
k6 run benchmark/verify.js
BASE_URL=http://192.168.1.50:8000 k6 run benchmark/verify.js
```

Defaults assume stock compose limits (2 CPU): 1 VU baseline → 6 VU stress;
`p(95)` gates in `benchmark/verify.js`.

## Models

Weights download at install, never committed (`models/*.onnx` gitignored).
Per-file terms — the project LICENSE does not cover them:

| File                 | Source                                                                                                                                                                                             | Terms                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `retinaface_r34.onnx`  | [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) (self-trained RetinaFace-R34 on WIDERFACE)                                                                                | MIT                                             |
| `edgeface_base.onnx`   | [yakhyo/edgeface-onnx](https://github.com/yakhyo/edgeface-onnx), from [otroshi/edgeface](https://github.com/otroshi/edgeface)                                                                        | MIT (converter) + BSD-3-Clause (Idiap original) |
| `minifasnetv2.onnx`    | [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing), after [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)               | Apache-2.0                                      |
| `face_attrib_net.onnx` | [yakhyo/face-attribute](https://github.com/yakhyo/face-attribute), mirror of [Qualcomm FaceAttribNet](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net) | BSD-3-Clause (Qualcomm + converter)             |

Research references: [RetinaFace](https://arxiv.org/abs/1905.00641) (CVPR 2020),
[EdgeFace](https://github.com/otroshi/edgeface). Detector implementation and
self-trained weights: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch).

## Dependencies

Runtime (`requirements.txt`), all permissive — use freely under their terms:

| Package                                   | License            |
| ----------------------------------------- | ------------------ |
| fastapi                                   | MIT                |
| uvicorn[standard]                         | BSD-3-Clause       |
| opencv-python-headless (+ bundled OpenCV) | MIT + Apache-2.0   |
| onnxruntime                               | MIT                |
| numpy                                     | BSD-3-Clause style |
| pydantic                                  | MIT                |
| python-multipart                          | Apache-2.0         |

## Limitations

- 1:1 verification only (best face per image); no 1:N search, no tracking.
- CPU inference; throughput scales with workers/CPU, not GPU.
- Liveness is passive single-frame anti-spoof, not certified PAD — don't use as sole gate for high-risk flows.
- Face recognition carries misuse and bias risks: get consent, store embeddings
  securely, comply with BIPA/GDPR and local biometric law.

## Contributing

Issues and PRs welcome. Keep changes surgical (match existing style, no drive-by
refactors), note behavior changes in the PR body. By contributing you agree your
work lands under the same Apache-2.0 terms.

## License

Project code: [Apache-2.0](LICENSE) © 2026 didasy.
Weights and dependencies keep their own licenses — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
