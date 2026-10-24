#!/bin/bash
# One-time setup: Python venv + pinned dependencies + model weights (checksum-verified).
# Safe to re-run — each step is skipped when it's already done.
# bash 3.2 compatible (macOS stock bash).
set -e
cd "$(dirname "$0")"

HF=https://huggingface.co/depth-anything

# fetch <file> <hf-repo> <sha256> — download into vda/checkpoints and verify.
fetch() {
  local dest="vda/checkpoints/$1"
  if [ -f "$dest" ] && [ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" = "$3" ]; then
    echo "→ $1 present"
    return
  fi
  echo "→ downloading $1"
  mkdir -p vda/checkpoints
  curl -fL --progress-bar -o "$dest.part" "$HF/$2/resolve/main/$1"
  if [ "$(shasum -a 256 "$dest.part" | cut -d' ' -f1)" != "$3" ]; then
    rm -f "$dest.part"
    echo "checksum mismatch on $1 — try again"
    exit 1
  fi
  mv "$dest.part" "$dest"
}

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

# 3. model weights (Small, ~116 MB each, Apache-2.0)
fetch video_depth_anything_vits.pth Video-Depth-Anything-Small \
  13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609
fetch metric_video_depth_anything_vits.pth Metric-Video-Depth-Anything-Small \
  3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33

echo "✓ ready — start it with ./run.sh, then open http://127.0.0.1:7860"
