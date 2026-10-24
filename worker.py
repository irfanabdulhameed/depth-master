"""Persistent warm-model depth worker — loads the model ONCE, then serves jobs.

Architecture (replaces one-shot infer.py):
  - Model + MPS kernels stay warm between runs → no cold start per Generate.
  - Frames are fed to the model at BOUNDED resolution (short side ≈ INPUT_SIZE).
    The network downsizes to INPUT_SIZE internally anyway, so full-res input only
    inflated the returned float depth buffers; we upscale the final colorized
    uint8 instead. Peak memory is now independent of source/output resolution.
  - decode / infer / encode run in a pipeline (bounded queues) so the CPU decodes
    chunk N+1 and encodes chunk N-1 while the GPU infers chunk N.
  - Chunks are multiples of the model's internal window step (22) so we don't pay
    extra padding, and only OVERLAP=8 frames are re-inferred for seam alignment.
  - Hardware H.264 (VideoToolbox) encoding when available; libx264 fallback.
  - MPS allocator is capped via PYTORCH_MPS_HIGH_WATERMARK_RATIO → an oversized
    allocation raises a clean error instead of swap-thrashing the machine.
  - Optional 16-bit export: raw aligned depth is cached to disk during the run,
    then a second pass normalizes it over the WHOLE clip and writes 16-bit PNG,
    float EXR or ProRes 4444 (12-bit, hardware).

Protocol: one JSON object per stdin line.
  {"cmd":"job", "inp","out","grayscale","res","input_size","target_fps","maxf",
                "model":"relative|metric", "start":<s>, "end":<s, 0=end>,
                "export":"|png16|exr|prores"}
  {"cmd":"cancel"}   stop current job, finalize partial output
  {"cmd":"trim"}     release cached GPU/CPU memory (stays warm)
  {"cmd":"ping"} | {"cmd":"quit"}
stdout lines: READY | INFO ... | STAGE <s> | PROGRESS <done> <total> |
  EXPORT <done> <total> | GPU <gb> | TRIMMED <gb_before> <gb_after> | PONG |
  DONE {"out","frames","secs","peak_gb","partial","export"} | ERROR <msg>
stderr is diagnostics only (tqdm etc.) — keep it out of the protocol pipe.
"""
import os, sys, gc, json, time, signal, threading, queue, resource, shutil

# Must be set before torch import. High-watermark caps the MPS allocator at a
# fraction of the GPU's recommended working set — beyond it we get a clean
# RuntimeError (job fails, worker survives) instead of a system-wide swap spiral.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # before cv2 import

import numpy as np
import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
import torch

try:
    _nice = int(os.environ.get("DS_NICE", "5"))
    if _nice:
        os.nice(_nice)  # inference yields to the UI/system; GPU work is unaffected
except Exception:
    pass
_cvt = int(os.environ.get("DS_CV2_THREADS", "0"))
if _cvt:
    cv2.setNumThreads(_cvt)
_tt = int(os.environ.get("DS_TORCH_THREADS", "0"))
if _tt:
    torch.set_num_threads(_tt)

ROOT = os.path.dirname(os.path.abspath(__file__))
VDA = os.path.join(ROOT, "vda")
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
sys.path.insert(0, VDA)
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402
from utils.util import compute_scale_and_shift  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
WINDOW_STEP = 22            # model's INFER_LEN(32) - its internal OVERLAP(10)
SEAM_OVERLAP = 8            # frames re-inferred per chunk for cross-chunk alignment
MAX_WINDOWS_PER_CHUNK = 4   # chunk = k*22 frames, k<=4 → bounded latency + memory
CHECKPOINTS = {             # Small (vits) weights — both Apache-2.0
    "relative": "video_depth_anything_vits.pth",
    "metric": "metric_video_depth_anything_vits.pth",
}

LUT = (np.asarray(matplotlib.colormaps["turbo"](np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)

_CANCEL = threading.Event()
_QUIT = threading.Event()
_CMDQ = queue.Queue()
_LAST_SHAPE = None   # (model, quality) of the previous job, to detect shape changes


def say(line):
    print(line, flush=True)


def _on_term(*_a):
    # SIGTERM = cancel the running job (graceful partial save); quit when idle.
    _CANCEL.set()
    _QUIT.set()


signal.signal(signal.SIGTERM, _on_term)


def gpu_gb():
    try:
        return torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        return 0.0


def rss_gb():
    # ru_maxrss is bytes on macOS — but that's the PEAK; current RSS via /proc is
    # not available on macOS, so report peak-so-far here and let the parent
    # sample live RSS itself (it has psutil and our pid).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


class Stopped(Exception):
    pass


# ---------------------------------------------------------------- model load
_MODELS = {}


def get_model(kind):
    """Load a checkpoint once and keep it on the GPU. The relative model loads at
    startup; the metric one only when a job first asks for it."""
    if kind in _MODELS:
        return _MODELS[kind]
    ck = os.path.join(VDA, "checkpoints", CHECKPOINTS[kind])
    if not os.path.exists(ck):
        raise FileNotFoundError(f"{kind} model weights missing — run ./setup.sh")
    m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384],
                           metric=(kind == "metric"))
    try:
        state = torch.load(ck, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # older torch without mmap kwarg
        state = torch.load(ck, map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=True)
    del state
    _MODELS[kind] = m.to(DEV).eval()
    return _MODELS[kind]


say("STAGE loading")
t_load = time.time()
try:
    model = get_model("relative")
except FileNotFoundError as e:
    say(f"ERROR {e}")
    sys.exit(1)
load_secs = time.time() - t_load

def feed_dims(src_w, src_h, input_size):
    """The exact resolution we hand the model for this source.

    Mirrors the model transform's sizing (incl. its wide-video shrink), so the
    network sees the same detail as if we'd fed full-res frames — but the float
    depth it returns stays this small too. Never upscales a smaller source."""
    eff = input_size
    ratio = max(src_h, src_w) / max(1, min(src_h, src_w))
    if ratio > 1.78:
        eff = round(input_size * 1.777 / ratio / 14) * 14

    def net_dim(x):
        y = int(round(x / 14) * 14)
        return y if y >= eff else int(np.ceil(x / 14) * 14)

    if min(src_w, src_h) > eff:
        s = eff / min(src_w, src_h)
        return net_dim(src_w * s), net_dim(src_h * s), True
    return src_w, src_h, False


say("STAGE warmup")
t_warm = time.time()


def warm(input_size):
    """Compile MPS kernels for this input size — one dummy window at the SAME
    feed dims a real 16:9 job uses, so the shapes match exactly."""
    if DEV != "mps":
        return
    w, h, _ = feed_dims(1920, 1080, input_size)
    dummy = np.zeros((2, h, w, 3), dtype=np.uint8)
    with torch.no_grad():
        model.infer_video_depth(dummy, 24, input_size=input_size, device=DEV, fp32=True)
    torch.mps.synchronize()


try:
    warm(384)  # Balanced — the default the UI opens with
except Exception as e:
    print(f"warmup failed (non-fatal): {e}", file=sys.stderr, flush=True)
torch.mps.empty_cache() if DEV == "mps" else None
gc.collect()
say(f"INFO ready load_secs={load_secs:.1f} warmup_secs={time.time() - t_warm:.1f} device={DEV}")
say("READY")


# ---------------------------------------------------------------- encoder pick
_ENCODER = None
_PRORES = None


def pick_encoder():
    """Try hardware VideoToolbox once; cache the verdict."""
    global _ENCODER
    if _ENCODER:
        return _ENCODER
    test = os.path.join(ROOT, "outputs", ".enc_test.mp4")
    try:
        w = imageio.get_writer(test, fps=24, codec="h264_videotoolbox", macro_block_size=16,
                               pixelformat="yuv420p",
                               ffmpeg_params=["-b:v", "2000k", "-allow_sw", "1"])
        for _ in range(4):
            w.append_data(np.zeros((64, 64, 3), np.uint8))
        w.close()
        _ENCODER = "h264_videotoolbox"
    except Exception:
        _ENCODER = "libx264"
    finally:
        try:
            os.remove(test)
        except Exception:
            pass
    return _ENCODER


def make_writer(path, fps, out_w, out_h):
    # macro_block_size=2: yuv420p only needs even sizes. 16 made imageio STRETCH
    # 1080 → 1088 rows, so the depth no longer lined up with the source pixels.
    enc = pick_encoder()
    if enc == "h264_videotoolbox":
        kbps = int(min(16000, max(1500, out_w * out_h * max(fps, 10) * 0.12 / 1000)))
        params = ["-b:v", f"{kbps}k", "-allow_sw", "1", "-movflags", "+faststart"]
        return imageio.get_writer(path, fps=fps, codec="h264_videotoolbox", macro_block_size=2,
                                  pixelformat="yuv420p", ffmpeg_params=params), enc
    params = ["-crf", "20", "-preset", "veryfast", "-movflags", "+faststart"]
    return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2,
                              pixelformat="yuv420p", ffmpeg_params=params), enc


def _prores_open(path, fps, w, h, codec, pix):
    wr = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, pix_fmt_in="gray16le", pix_fmt_out=pix,
                                     codec=codec, quality=None, macro_block_size=1,
                                     output_params=["-profile:v", "4444"])
    wr.send(None)
    return wr


def prores_writer(path, fps, out_w, out_h):
    """16-bit gray in → ProRes 4444. The M4 media engine keeps 12 bits (measured
    max error 35/65535); prores_ks is the software fallback (10-bit input).
    Like pick_encoder, try hardware once on a throwaway file and cache the verdict."""
    global _PRORES
    if _PRORES is None:
        test = os.path.join(ROOT, "outputs", ".prores_test.mov")
        _PRORES = ("prores_ks", "yuv444p10le")
        try:
            wr = _prores_open(test, 24, 64, 64, "prores_videotoolbox", "p416le")
            for _ in range(2):
                wr.send(np.zeros((64, 64), np.uint16).tobytes())
            wr.close()
            _PRORES = ("prores_videotoolbox", "p416le")
        except Exception:
            pass
        finally:
            try:
                os.remove(test)
            except Exception:
                pass
    return _prores_open(path, fps, out_w, out_h, *_PRORES)


# ---------------------------------------------------------------- fit probe
def largest_fitting_input_size(mdl, src_w, src_h, requested):
    """Find the highest quality whose single 32-frame window fits under the MPS
    ceiling on THIS machine, by actually running one dummy window and stepping
    down on OOM. `infer_video_depth` pads any clip to a full 32-frame window, so
    a 2-frame dummy has the exact same peak as a real window — a faithful probe.

    With fused attention, Best (518) on 16:9 fits the 8.29 GiB ceiling of a 16 GB
    M4 (peak ~5.9 GiB). The probe still guards odd aspect ratios and smaller
    machines: instead of erroring we quietly drop to the best quality that fits.
    On success the allocator cache is already warm for the real run."""
    fw, fh, dz = feed_dims(src_w, src_h, requested)
    if DEV != "mps":
        return requested, fw, fh, dz, False
    size = requested
    while size >= 294:
        fw, fh, dz = feed_dims(src_w, src_h, size)
        oom = False
        try:
            dummy = np.zeros((2, fh, fw, 3), dtype=np.uint8)
            with torch.no_grad():
                mdl.infer_video_depth(dummy, 24, input_size=size, device=DEV, fp32=True)
            torch.mps.synchronize()
        except RuntimeError as e:
            if "memory" not in str(e).lower() and "MPS" not in str(e):
                raise
            oom = True
        # Reclaim OUTSIDE the except block: while `e` is in scope its traceback
        # pins the model's GPU tensors, so empty_cache() there frees nothing (and
        # would give a false "fits" on the next probe). Once the block exits,
        # Python drops `e` and the tensors release, so this actually frees them.
        if not oom:
            if size == requested:
                return size, fw, fh, dz, False   # requested quality fits outright
            # We had to reduce, and this size only *just* fit a single clean
            # window. The real run also holds decode+encode pipeline buffers in
            # the same unified memory, which can tip a marginal fit over the
            # ceiling mid-run. Take one extra step down for reliable headroom —
            # "never crash" beats squeezing out one marginal quality level.
            safe = max(294, size - 56)
            fw, fh, dz = feed_dims(src_w, src_h, safe)
            return safe, fw, fh, dz, True
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        say(f"INFO quality {size} won't fit this machine, trying {size - 56}")
        size -= 56  # one patch-step (4*14) down
    fw, fh, dz = feed_dims(src_w, src_h, 294)
    return 294, fw, fh, dz, True


# ---------------------------------------------------------------- 16-bit export
EXPORT_SUFFIX = {"png16": "_png16", "exr": "_exr", "prores": "_prores.mov"}


def to_vis(d, metric):
    """Depth → 'nearness' (bigger = nearer), the convention every visual output
    uses: relative output already is inverse depth; metric output is meters."""
    return 1.0 / np.maximum(d, 1e-3) if metric else d


def export_pass(fmt, raw_path, n, feed_w, feed_h, out_w, out_h, fps, lo, hi, metric, base):
    """Second pass over the cached raw depth. PNG16/ProRes are normalized with ONE
    range for the whole clip (near = white), so brightness never shifts between
    frames. EXR keeps raw values: relative inverse depth, or meters (metric)."""
    raw = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n, feed_h, feed_w))
    dest = base + EXPORT_SUFFIX[fmt]
    up = (out_w, out_h) != (feed_w, feed_h)
    writer = None
    if fmt == "prores":
        writer = prores_writer(dest, fps, out_w, out_h)
    else:
        os.makedirs(dest, exist_ok=True)
    scale = 65535.0 / (hi - lo) if hi - lo > 1e-9 else 0.0
    try:
        for i in range(n):
            if _CANCEL.is_set():
                break
            d = np.asarray(raw[i])
            if fmt == "exr":
                if up:
                    d = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(dest, f"depth_{i + 1:06d}.exr"), d,
                            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                             cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
            else:
                v = (to_vis(d, metric) - lo) * scale
                if up:
                    v = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                u16 = np.clip(v, 0, 65535).astype(np.uint16)
                if writer is not None:
                    writer.send(u16.tobytes())
                else:
                    cv2.imwrite(os.path.join(dest, f"depth_{i + 1:06d}.png"), u16,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if (i + 1) % 8 == 0 or i + 1 == n:
                say(f"EXPORT {i + 1} {n}")
    finally:
        if writer is not None:
            writer.close()
        del raw
    return dest


# ---------------------------------------------------------------- job
def run_job(job):
    _CANCEL.clear()
    inp = job["inp"]
    out = job["out"]
    gray = bool(job.get("grayscale"))
    res = int(job.get("res", 0))
    input_size = int(job.get("input_size", 384))
    target_fps = float(job.get("target_fps", 0))
    maxf = int(job.get("maxf", 0))
    kind = "metric" if job.get("model") == "metric" else "relative"
    metric = kind == "metric"
    start_s = max(0.0, float(job.get("start") or 0))
    end_s = max(0.0, float(job.get("end") or 0))
    export = job.get("export") or ""
    if export and export not in EXPORT_SUFFIX:
        say(f"ERROR unknown export format {export}")
        return

    try:
        mdl = get_model(kind)
    except FileNotFoundError as e:
        say(f"ERROR {e}")
        return

    # A model/quality change (e.g. Balanced→Best) leaves differently-shaped
    # buffers cached that the new run can't reuse — drop them so the first big
    # window allocates against a clean cache instead of overflowing the ceiling.
    global _LAST_SHAPE
    if DEV == "mps" and (kind, input_size) != _LAST_SHAPE:
        gc.collect()
        torch.mps.empty_cache()
    _LAST_SHAPE = (kind, input_size)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        say("ERROR cannot open video")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        ok, probe = cap.read()
        if not ok:
            say("ERROR no frames read")
            cap.release()
            return
        src_h, src_w = probe.shape[:2]
        cap.release()
        cap = cv2.VideoCapture(inp)

    # Trim: [first, last) in source frames. Seeking lands on the exact frame
    # (OpenCV's FFmpeg backend decodes forward from the previous keyframe).
    first = int(round(start_s * src_fps))
    last = int(round(end_s * src_fps)) if end_s > 0 else 0
    if src_total:
        last = min(last, src_total) if last else src_total
    if last and first >= last:
        say("ERROR trim start is at or after the end of the clip")
        cap.release()
        return
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    stride = max(1, round(src_fps / target_fps)) if target_fps > 0 else 1
    out_fps = src_fps / stride
    span = (last - first) if last else 0
    total = (span + stride - 1) // stride if span else 0
    if maxf:
        total = min(total, maxf) if total else maxf

    # Feed frames at the network's own resolution for this source — the model
    # computes depth at this size regardless; feeding full-res frames would only
    # inflate the returned float depth buffers. The probe also clamps quality to
    # what fits in memory, so the run can never OOM-crash.
    requested_size = input_size
    input_size, feed_w, feed_h, downsize, reduced = largest_fitting_input_size(
        mdl, src_w, src_h, input_size)
    if reduced:
        say(f"INFO auto-reduced quality {requested_size}->{input_size} to fit "
            f"this machine's memory")

    # Output size: match input, or cap the long side at `res`.
    out_scale = min(1.0, res / max(src_h, src_w)) if res > 0 else 1.0
    out_w, out_h = int(round(src_w * out_scale)), int(round(src_h * out_scale))

    # 16-bit export caches raw depth at feed res on disk (float32) for pass 2.
    raw_f = raw_path = None
    if export:
        need = total * (feed_w * feed_h * 4 + out_w * out_h * 4) * 1.1 if total else 0
        free = shutil.disk_usage(os.path.dirname(out) or ".").free
        if need and free < need:
            say(f"ERROR not enough disk space for the {export} export "
                f"(needs ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free)")
            cap.release()
            return
        raw_path = os.path.splitext(out)[0] + ".raw.f32"
        raw_f = open(raw_path, "wb")
    samples = []                # sparse nearness samples → one global export range

    # Chunk = k * 22 (model window step) incl. SEAM_OVERLAP re-inferred frames.
    avail = 4e9
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        pass
    budget = min(2.5e9, max(0.6e9, avail * 0.3))
    per_frame = feed_w * feed_h * 16  # uint8 RGB + float32 depth + working copies
    k = max(1, min(MAX_WINDOWS_PER_CHUNK, int(budget / (WINDOW_STEP * per_frame))))
    chunk_total = WINDOW_STEP * k

    say(f"INFO src={src_w}x{src_h}@{src_fps:.1f} feed={feed_w}x{feed_h} out={out_w}x{out_h} "
        f"stride={stride} total={total} chunk={chunk_total} overlap={SEAM_OVERLAP} "
        f"input_size={input_size} encoder={pick_encoder()} model={kind} "
        f"start={first} export={export or 'none'}")

    err = {"msg": None}
    frame_q = queue.Queue(maxsize=chunk_total + SEAM_OVERLAP)
    write_q = queue.Queue(maxsize=2)
    written = {"n": 0}

    def decoder():
        idx = first
        kept = 0
        try:
            while not _CANCEL.is_set():
                if maxf and kept >= maxf:
                    break
                if last and idx >= last:
                    break
                if (idx - first) % stride:
                    if not cap.grab():        # skipped frame: decode-lite, no retrieve
                        break
                    idx += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    break
                idx += 1
                kept += 1
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if downsize:
                    fr = cv2.resize(fr, (feed_w, feed_h), interpolation=cv2.INTER_AREA)
                frame_q.put(fr)
        except Exception as e:
            err["msg"] = f"decode: {e}"
            _CANCEL.set()
        finally:
            frame_q.put(None)

    def encoder():
        writer = None
        try:
            while True:
                item = write_q.get()
                if item is None:
                    break
                if writer is None:
                    writer, _ = make_writer(out, out_fps, out_w, out_h)
                for plane in item:            # uint8 normalized depth at feed res
                    if (plane.shape[1], plane.shape[0]) != (out_w, out_h):
                        plane = cv2.resize(plane, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                    rgb = np.repeat(plane[..., None], 3, axis=-1) if gray else LUT[plane]
                    writer.append_data(rgb)
                    written["n"] += 1
        except Exception as e:
            err["msg"] = f"encode: {e}"
            _CANCEL.set()
            # drain so the producer never blocks on a dead consumer
            while True:
                try:
                    if write_q.get_nowait() is None:
                        break
                except queue.Empty:
                    break
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

    t_dec = threading.Thread(target=decoder, daemon=True)
    t_enc = threading.Thread(target=encoder, daemon=True)
    t_dec.start()
    t_enc.start()

    say("STAGE inferring")
    t0 = time.time()
    done = 0
    lo = hi = None
    prev_tail = None            # aligned depth of last SEAM_OVERLAP frames
    prev_frames = None          # their feed-res RGB frames
    eof = False

    def pull(n):
        nonlocal eof
        got = []
        while len(got) < n and not eof:
            fr = frame_q.get()
            if fr is None:
                eof = True
                break
            got.append(fr)
        return got

    try:
        while not eof and not _CANCEL.is_set():
            # First chunk is small (2 windows) so the writer gets frames — and an
            # early Stop keeps a partial — within ~5 s instead of ~10 s.
            this_total = min(chunk_total, WINDOW_STEP * 2) if prev_frames is None else chunk_total
            want_new = this_total - (SEAM_OVERLAP if prev_frames is not None else 0)
            if maxf:
                want_new = min(want_new, maxf - done)
                if want_new <= 0:
                    break
            new_frames = pull(want_new)
            if not new_frames:
                break
            frames = (list(prev_frames) + new_frames) if prev_frames is not None else new_frames
            ov = len(frames) - len(new_frames)
            chunk = np.stack(frames)
            n_new = len(new_frames)
            base = done
            del frames, new_frames

            def cb(win, nwin, _base=base, _new=n_new, _ov=ov):
                if _CANCEL.is_set():
                    raise Stopped()
                approx = min(_new, max(0, win * WINDOW_STEP - _ov))
                say(f"PROGRESS {_base + approx} {total}")

            oom_retry = False
            try:
                depths, _ = mdl.infer_video_depth(chunk, out_fps, input_size=input_size,
                                                  device=DEV, fp32=True, progress_callback=cb)
            except RuntimeError as e:
                if DEV != "mps" or ("memory" not in str(e).lower() and "MPS" not in str(e)):
                    raise
                oom_retry = True
            if oom_retry:
                # Reclaim only after the except block exits — while `e` was alive
                # its traceback pinned the model's GPU tensors and empty_cache()
                # freed nothing. Now they're released, so retry from a clean slate.
                gc.collect()
                torch.mps.empty_cache()
                torch.mps.synchronize()
                say(f"INFO reclaimed cache after ceiling, retrying chunk (gpu {gpu_gb():.2f} GB)")
                depths, _ = mdl.infer_video_depth(chunk, out_fps, input_size=input_size,
                                                  device=DEV, fp32=True, progress_callback=cb)

            # Metric depth is already in meters — chunks agree without a fit
            # (the model itself skips its window alignment for metric too).
            if ov > 0 and prev_tail is not None and not metric:
                pred = depths[:ov].reshape(-1)
                tgt = prev_tail.reshape(-1)
                step = max(1, pred.size // 20000)          # 2-param fit needs ~20k px, not 2M
                try:
                    s, b = compute_scale_and_shift(pred[::step], tgt[::step],
                                                   np.ones_like(pred[::step], dtype=np.float32))
                    if np.isfinite(s) and np.isfinite(b) and s > 0:
                        depths *= s
                        depths += b
                except Exception:
                    pass

            prev_tail = depths[-SEAM_OVERLAP:].copy()
            # .copy() so these 8 frames don't pin the whole chunk array in memory
            prev_frames = list(chunk[-SEAM_OVERLAP:].copy())
            new_depths = depths[ov:]
            if raw_f is not None:
                new_depths.astype(np.float32, copy=False).tofile(raw_f)
            vis = to_vis(new_depths, metric)

            samp = vis[:, ::4, ::4]
            if raw_f is not None:
                samples.append(samp.reshape(-1)[:: max(1, samp.size // 50000)].copy())
            # Robust running range: expand instantly (values get clipped otherwise),
            # shrink smoothly — no brightness jumps at chunk seams.
            lo_c, hi_c = np.percentile(samp, 1.0), np.percentile(samp, 99.0)
            lo = lo_c if lo is None else (min(lo, lo_c) if lo_c < lo else lo + 0.25 * (lo_c - lo))
            hi = hi_c if hi is None else (max(hi, hi_c) if hi_c > hi else hi + 0.25 * (hi_c - hi))
            den = (hi - lo) if (hi - lo) > 1e-6 else 1.0

            vis -= lo
            vis *= 255.0 / den
            np.clip(vis, 0, 255, out=vis)
            u8 = vis.astype(np.uint8)
            write_q.put(u8)

            done += n_new
            say(f"PROGRESS {done} {total}")
            say(f"GPU {gpu_gb():.2f}")
            del depths, new_depths, vis, chunk, u8
            # Reclaim the allocator's cache before the next chunk when it's grown
            # toward the ceiling — this is what makes Best-quality runs survive:
            # a fresh chunk's first window (~4 GiB) then allocates against a clean
            # cache instead of stacking on stale buffers. Cheap at Best (few big
            # chunks); doesn't trigger at Fast/Balanced where resident stays low.
            if DEV == "mps" and torch.mps.driver_allocated_memory() > 3e9:
                gc.collect()
                torch.mps.empty_cache()
    except Stopped:
        say("STAGE stopped")
    except RuntimeError as e:
        msg = str(e).replace("\n", " ")[:300]
        if "memory" in msg.lower() or "MPS" in msg:
            err["msg"] = f"gpu-memory-ceiling: {msg} — try Fast speed or a lower resolution"
        else:
            err["msg"] = msg
    except Exception as e:
        err["msg"] = str(e).replace("\n", " ")[:300]

    stopped = _CANCEL.is_set()
    _CANCEL.set() if err["msg"] else None
    t_dec.join(timeout=5)
    while True:                          # drain any decoded leftovers
        try:
            frame_q.get_nowait()
        except queue.Empty:
            break
    write_q.put(None)
    t_enc.join(timeout=60)
    cap.release()
    gc.collect()
    if DEV == "mps":
        # Between jobs the allocator can hold ~6 GB of cached buffers — on unified
        # memory that's real RAM. Release it once per job (re-allocating is cheap;
        # kernel COMPILATION is the expensive part, and that stays warm).
        torch.mps.empty_cache()
        say(f"GPU {gpu_gb():.2f}")

    partial = 1 if (stopped or err["msg"]) else 0

    # Pass 2: the export covers every frame that made it into the preview, so a
    # stopped run still exports its partial. A Stop during export ends it early.
    export_path = None
    if raw_f is not None:
        raw_f.close()
        n_raw = os.path.getsize(raw_path) // (feed_w * feed_h * 4)
        if n_raw and not err["msg"]:
            _CANCEL.clear()
            say("STAGE exporting")
            allv = np.concatenate(samples)
            g_lo, g_hi = float(np.percentile(allv, 0.1)), float(np.percentile(allv, 99.9))
            del allv, samples
            try:
                export_path = export_pass(export, raw_path, n_raw, feed_w, feed_h, out_w, out_h,
                                          out_fps, g_lo, g_hi, metric, os.path.splitext(out)[0])
                partial = 1 if (partial or _CANCEL.is_set()) else 0
            except Exception as e:
                say(f"INFO export failed: {str(e)[:200]}")
        try:
            os.remove(raw_path)
        except Exception:
            pass

    ok_file = os.path.exists(out) and os.path.getsize(out) > 1000 and written["n"] > 0
    if err["msg"] and not ok_file:
        say(f"ERROR {err['msg']}")
    elif not ok_file:
        say("ERROR cancelled before any frames" if partial else "ERROR no frames read")
    else:
        if err["msg"]:
            say(f"INFO finished-early {err['msg']}")
        say("DONE " + json.dumps({"out": out, "frames": written["n"],
                                  "secs": round(time.time() - t0, 1), "peak_gb": round(rss_gb(), 2),
                                  "partial": partial, "export": export_path}))


# ---------------------------------------------------------------- command loop
def stdin_reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        cmd = obj.get("cmd")
        if cmd == "cancel":
            _CANCEL.set()
        elif cmd == "quit":
            _QUIT.set()
            _CANCEL.set()
            _CMDQ.put(None)
            return
        else:
            _CMDQ.put(obj)
    _QUIT.set()
    _CANCEL.set()
    _CMDQ.put(None)


threading.Thread(target=stdin_reader, daemon=True).start()

while not _QUIT.is_set():
    try:
        obj = _CMDQ.get(timeout=0.5)
    except queue.Empty:
        continue
    if obj is None:
        break
    cmd = obj.get("cmd")
    if cmd == "job":
        # No pre-warm here: the job's first window compiles any new kernel shapes
        # itself; a dummy warm with the same shapes would just double that cost.
        try:
            run_job(obj)
        except Exception as e:  # a bad job must never take the warm worker down
            say(f"ERROR {str(e).replace(chr(10), ' ')[:300]}")
    elif cmd == "ping":
        say("PONG")
    elif cmd == "trim":
        before = gpu_gb()
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        say(f"TRIMMED {before:.2f} {gpu_gb():.2f}")

say("INFO bye")
