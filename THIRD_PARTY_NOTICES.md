# Third-party notices

This file records licenses of artifacts this project **uses but does not own**.
The project [LICENSE](LICENSE) (Apache-2.0) covers project code only.

All entries verified against upstream GitHub repos on 2026-09-17.

## Model weights (downloaded at install, never committed)

### retinaface_r34.onnx — RetinaFace-R34 face detector
- Trainer + converter: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) — **MIT** © 2026 Yakhyokhuja Valikhujaev.
- Self-trained on WIDERFACE (training code `train.py` + WIDERFACE docs in repo);
  converter is the original trainer, no third-party weight lineage.
- Method reference: [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) (CVPR 2020).

### edgeface_base.onnx — EdgeFace-Base recognizer
- Converter: [yakhyo/edgeface-onnx](https://github.com/yakhyo/edgeface-onnx) — **MIT** © 2026 Yakhyokhuja Valikhujaev.
- Original: [otroshi/edgeface](https://github.com/otroshi/edgeface) — **BSD-3-Clause** © 2024 Idiap Research Institute.

### minifasnetv2.onnx — MiniFASNetV2 anti-spoof
- Converter: [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) — **Apache-2.0**.
- Original: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) — **Apache-2.0** © 2020 Minivision.

### face_attrib_net.onnx — FaceAttribNet attributes
- Converter: [yakhyo/face-attribute](https://github.com/yakhyo/face-attribute) — **BSD-3-Clause** © 2025 Qualcomm, © 2026 Yakhyokhuja Valikhujaev.
- Original: [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net) — **BSD-3-Clause** © 2025 Qualcomm.

## Runtime dependencies (installed via pip, not vendored)

| Package                | License                                     | Holder                            |
| ---------------------- | ------------------------------------------- | --------------------------------- |
| fastapi                | MIT                                         | Sebastián Ramírez                 |
| uvicorn[standard]      | BSD-3-Clause                                | Encode OSS                        |
| opencv-python-headless | MIT (wrapper) + Apache-2.0 (bundled OpenCV) | Olli-Pekka Heinisuo / OpenCV team |
| onnxruntime            | MIT                                         | Microsoft                         |
| numpy                  | BSD-3-Clause style                          | NumPy Developers                  |
| pydantic               | MIT                                         | Pydantic Services                 |
| python-multipart       | Apache-2.0                                  | Kludex / contributors             |

No copyleft (GPL/LGPL/AGPL) dependency. k6 (used only as an external load
generator in `benchmark/`) does not link into the project.
