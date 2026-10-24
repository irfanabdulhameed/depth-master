#!/bin/bash
# Depth Studio supervisor — keeps the UI alive. If the server ever dies (OOM,
# crash), it relaunches in 2 s; a clean exit (Ctrl-C / normal quit) stops it.
# bash 3.2 compatible (macOS stock bash) — no associative arrays.
cd "$(dirname "$0")" || exit 1
trap 'echo; echo "supervisor stopped"; exit 0' INT TERM

if [ ! -x .venv/bin/python ] || [ ! -f vda/checkpoints/video_depth_anything_vits.pth ]; then
  echo "Not set up yet — run ./setup.sh first."
  exit 1
fi

while true; do
  .venv/bin/python app.py
  code=$?
  if [ "$code" -eq 0 ]; then
    exit 0
  fi
  echo "Depth Studio exited with code $code — restarting in 2 s (Ctrl-C to stop)"
  sleep 2
done
