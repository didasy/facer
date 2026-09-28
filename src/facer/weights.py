"""Explicit weight downloads. Importing facer never downloads model files."""
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

MODEL_FILES = {
    "detector": "retinaface_r34.onnx",
    "recognizer": "edgeface_base.onnx",
    "liveness": "minifasnetv2.onnx",
    "attribute": "face_attrib_net.onnx",
}
WEIGHT_URLS = {
    "retinaface_r34.onnx": "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx",
    "edgeface_base.onnx": "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_base.onnx",
    "minifasnetv2.onnx": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
    "face_attrib_net.onnx": "https://github.com/yakhyo/face-attribute/releases/download/weights/face_attrib_net.onnx",
}
_MINIFAS_SHA256 = "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"


def _verify_weight(path: Path, name: str) -> None:
    if name == "minifasnetv2.onnx":
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != _MINIFAS_SHA256:
            raise ValueError("minifasnetv2.onnx sha256 mismatch")


def download_weights(dest: str | Path = "./models", force: bool = False) -> dict[str, Path]:
    """Fetch four yakhyo ONNX assets, skipping existing files unless force=True.

    Downloads replace files only after success. MiniFASNetV2's published SHA256
    is checked, including for existing files. Network and integrity errors propagate.
    Weight licenses are separate from the library license.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in WEIGHT_URLS.items():
        target = dest / name
        if target.is_file() and not force:
            _verify_weight(target, name)
        else:
            with tempfile.NamedTemporaryFile(dir=dest, prefix=f".{name}.", delete=False) as output:
                temporary = Path(output.name)
            try:
                with urlopen(url, timeout=100) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                _verify_weight(temporary, name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        paths[name] = target
    return paths
