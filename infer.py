"""Chunked depth inference — bounded memory, ANY length, with speed controls.

Reads/infers/writes in small overlapping chunks and frees each before the next,
so peak memory stays ~fixed regardless of clip length. Chunks are scale-aligned
on their overlap (the model's own technique) so there are no seams.

Env:
  INP, OUT, GRAYSCALE
  RES         max output side px; 0 = match input
  INPUT_SIZE  model internal resolution (speed/quality): 294 fast .. 518 best
  TARGET_FPS  process at this fps; 0 = original (fewer frames = linear speedup)
  CHUNK, OVERLAP, MAXF
Prints:  INFO ... | STAGE <name> | PROGRESS <done> <total> | GPU <gb>
"""
import os, sys, time, gc, signal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
VDA = os.path.join(ROOT, "vda")
sys.path.insert(0, VDA)
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402
from utils.util import compute_scale_and_shift  # noqa: E402

INP = os.environ["INP"]
OUT = os.environ["OUT"]
GRAY = os.environ.get("GRAYSCALE", "0") == "1"
RES = int(os.environ.get("RES", "0"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "384"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", "0"))
CHUNK = int(os.environ.get("CHUNK", "64"))
OVERLAP = int(os.environ.get("OVERLAP", "8"))
MAXF = int(os.environ.get("MAXF", "0"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# Graceful stop: on SIGTERM finish/flush what's done so far rather than corrupt output.
_STOP = False


class _Stopped(Exception):
    pass


def _on_term(*_a):
    global _STOP
    _STOP = True


signal.signal(signal.SIGTERM, _on_term)


def gpu():
    try:
        return torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        return 0.0


cap = cv2.VideoCapture(INP)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
# Downsample fps by taking every `stride`-th source frame.
stride = max(1, round(src_fps / TARGET_FPS)) if TARGET_FPS and TARGET_FPS > 0 else 1
out_fps = src_fps / stride
total = (src_total + stride - 1) // stride if src_total else 0
if MAXF:
    total = min(total, MAXF) if total else MAXF
print(f"INFO src_fps={src_fps:.1f} out_fps={out_fps:.1f} stride={stride} total={total} "
      f"chunk={CHUNK} input_size={INPUT_SIZE} res={'match' if RES == 0 else RES}", flush=True)

_src_idx = 0


def read_frames(cap, n):
    global _src_idx
    out = []
    while len(out) < n:
        ret, fr = cap.read()
        if not ret:
            break
        keep = (_src_idx % stride == 0)
        _src_idx += 1
        if not keep:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if RES > 0:
            h, w = fr.shape[:2]
            s = RES / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        out.append(fr)
    return np.stack(out) if out else None


print("STAGE loading", flush=True)
cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
model = VideoDepthAnything(**cfg)
ck = os.path.join(VDA, "checkpoints/video_depth_anything_vits.pth")
model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
model = model.to(dev).eval()

if not GRAY:
    cmap = matplotlib.colormaps["turbo"]
    LUT = (np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)

writer = None
gmin = gmax = None
prev_tail = None
prev_ov_frames = None
done = 0

print("STAGE inferring", flush=True)
t0 = time.time()
while True:
    if _STOP or (MAXF and done >= MAXF):
        break
    want = CHUNK if not MAXF else min(CHUNK, MAXF - done)
    new_frames = read_frames(cap, want)
    if new_frames is None:
        break

    if prev_ov_frames is not None:
        chunk = np.concatenate([prev_ov_frames, new_frames], axis=0)
        ov = len(prev_ov_frames)
    else:
        chunk = new_frames
        ov = 0

    chunk_new = len(new_frames)
    base = done

    def _cb(win, total_win, _base=base, _new=chunk_new):
        if _STOP:
            raise _Stopped()
        # Smooth sub-chunk progress: approximate frames finished within this chunk.
        approx = _base + int((win / max(1, total_win)) * _new)
        print(f"PROGRESS {min(approx, _base + _new)} {total}", flush=True)

    try:
        depths, _ = model.infer_video_depth(chunk, out_fps, input_size=INPUT_SIZE,
                                            device=dev, fp32=True, progress_callback=_cb)
    except _Stopped:
        print("STAGE stopped", flush=True)
        break

    if ov > 0 and prev_tail is not None:
        pred = depths[:ov].reshape(-1)
        tgt = prev_tail.reshape(-1)
        try:
            scale, shift = compute_scale_and_shift(pred, tgt, np.ones_like(pred, dtype=np.float32))
            if np.isfinite(scale) and np.isfinite(shift) and scale > 0:
                depths = depths * scale + shift
        except Exception:
            pass

    new_depths = depths[ov:]
    prev_tail = depths[-OVERLAP:].copy()
    prev_ov_frames = chunk[-OVERLAP:].copy()

    cmin, cmax = float(new_depths.min()), float(new_depths.max())
    gmin = cmin if gmin is None else min(gmin, cmin)
    gmax = cmax if gmax is None else max(gmax, cmax)
    den = (gmax - gmin) or 1.0
    norm = np.clip((new_depths - gmin) / den * 255, 0, 255).astype(np.uint8)
    col = np.repeat(norm[..., None], 3, axis=-1) if GRAY else LUT[norm]

    if writer is None:
        writer = imageio.get_writer(OUT, fps=out_fps, codec="libx264", macro_block_size=16,
                                    pixelformat="yuv420p", ffmpeg_params=["-crf", "20", "-movflags", "+faststart"])
    for fr in col:
        writer.append_data(np.ascontiguousarray(fr))

    done += len(new_depths)
    print(f"PROGRESS {done} {total}", flush=True)
    print(f"GPU {gpu():.2f}", flush=True)

    del depths, new_depths, col, norm, chunk, new_frames
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()

cap.release()
if writer is None:
    print("ERROR no frames read", flush=True)
    sys.exit(2)
writer.close()
print(f"STAGE done OUT={OUT} frames={done} secs={time.time() - t0:.1f}", flush=True)
print("MINI_OK", flush=True)
