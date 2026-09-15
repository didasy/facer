#!/usr/bin/env bash
# Download ONNX weights (yakhyo builds) into this directory.
# Filenames match app/main.py: det_10g.onnx, edgeface_base.onnx, minifasnetv2.onnx, face_attrib_net.onnx.
# Skips files already present; pass --force to re-download.
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; fi

dl() {
  local url="$1" out="$2"
  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    echo "skip: $out exists"
    return 0
  fi
  echo "fetch: $out"
  curl -fSL --retry 3 --retry-delay 2 -o "$out.tmp" "$url"
  mv "$out.tmp" "$out"
}

cd "$(dirname "$0")"

dl "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx" "det_10g.onnx"
dl "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_base.onnx" "edgeface_base.onnx"
# Upstream asset is MiniFASNetV2.onnx; renamed to match MINIFAS_PATH in app/main.py
dl "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx" "minifasnetv2.onnx"
dl "https://github.com/yakhyo/face-attribute/releases/download/weights/face_attrib_net.onnx" "face_attrib_net.onnx"

# Integrity check: yakhyo publishes sha256 for MiniFASNetV2.onnx
EXPECTED_MINIFAS="b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_MINIFAS="$(sha256sum minifasnetv2.onnx | awk '{print $1}')"
  if [[ "$ACTUAL_MINIFAS" != "$EXPECTED_MINIFAS" ]]; then
    echo "ERROR: minifasnetv2.onnx sha256 mismatch (got $ACTUAL_MINIFAS)" >&2
    exit 1
  fi
  echo "ok: minifasnetv2.onnx sha256 verified"
fi

ls -la ./*.onnx
