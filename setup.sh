#!/bin/bash
# One-time setup: Python venv + pinned dependencies + model weights (checksum-verified).
# Safe to re-run — each step is skipped when it's already done.
# bash 3.2 compatible (macOS stock bash).
set -e
cd "$(dirname "$0")"

WEIGHTS=vda/checkpoints/video_depth_anything_vits.pth
WEIGHTS_URL=https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
WEIGHTS_SHA256=13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609

if [ "$(uname -s)/$(uname -m)" != "Darwin/arm64" ]; then
  echo "warning: built for Apple Silicon macOS — on this machine it will run on CPU (very slow)."
fi

# 1. venv (Python 3.11 — the version every pin below is tested with)
if [ ! -x .venv/bin/python ]; then
  PY=${PYTHON:-}
  if [ -z "$PY" ]; then
    for c in python3.11 /opt/homebrew/bin/python3.11 python3; do
      if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.11 not found — install it with: brew install python@3.11"
    exit 1
  fi
  echo "→ creating .venv with $("$PY" -V)"
  "$PY" -m venv .venv
fi

# 2. dependencies
echo "→ installing dependencies"
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt

# 3. model weights (~116 MB, Apache-2.0)
if [ -f "$WEIGHTS" ] && [ "$(shasum -a 256 "$WEIGHTS" | cut -d' ' -f1)" = "$WEIGHTS_SHA256" ]; then
  echo "→ model weights present"
else
  echo "→ downloading model weights"
  mkdir -p "$(dirname "$WEIGHTS")"
  curl -fL --progress-bar -o "$WEIGHTS.part" "$WEIGHTS_URL"
  if [ "$(shasum -a 256 "$WEIGHTS.part" | cut -d' ' -f1)" != "$WEIGHTS_SHA256" ]; then
    rm -f "$WEIGHTS.part"
    echo "checksum mismatch on downloaded weights — try again"
    exit 1
  fi
  mv "$WEIGHTS.part" "$WEIGHTS"
fi

echo "✓ ready — start it with ./run.sh, then open http://127.0.0.1:7860"
