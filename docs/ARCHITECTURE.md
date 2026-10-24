# Architecture

How Depth Studio turns a video into a depth map on a 16 GB Apple Silicon Mac
without ever taking the machine down. Read this before changing `worker.py`,
`app.py` or anything under `vda/`.

**Priority order: never crash → correct output → speed.**

---

## Processes

```
run.sh  (supervisor: restarts app.py if it dies, exits on a clean quit)
  └─ app.py  Gradio UI on :7860 — holds NO torch
       │  watchdogs · device analytics · ETA · settings
       │
       │  JSON lines on stdin  ──►   stdout protocol lines  ◄──
       ▼                                  (stderr → worker.log)
     worker.py  persistent, warm — loads the model ONCE
       ├─ decoder thread   OpenCV → RGB at network resolution
       ├─ main thread      Video-Depth-Anything on MPS, chunk by chunk
       └─ encoder thread   uint8 → output size → turbo/gray → H.264 (VideoToolbox)
```

- The UI process stays light, so a bad run can only kill the worker. A dead worker is
  respawned on the next Generate.
- The worker loads the model and compiles MPS kernels once at launch (warmed at
  Balanced), so repeat runs skip the ~10 s cold start.

### Worker protocol

| Direction | Message |
|---|---|
| stdin | `{"cmd":"job","inp","out","grayscale","res","input_size","target_fps","maxf"}` |
| stdin | `{"cmd":"cancel"}` stop and finalize a playable partial · `{"cmd":"trim"}` drop caches, stay warm · `{"cmd":"ping"}` · `{"cmd":"quit"}` |
| stdout | `READY` · `INFO …` · `STAGE <name>` · `PROGRESS <done> <total>` · `GPU <gb>` · `TRIMMED <before> <after>` · `PONG` |
| stdout | `DONE out=<path> frames=<n> secs=<s> peak_gb=<g> partial=<0\|1>` or `ERROR <msg>` |

SIGTERM behaves like `cancel`, then quits. **Keep stderr out of stdout.** Merged
stderr once glued tqdm output onto `PROGRESS` lines and broke parsing.

---

## The job pipeline

1. **Fit probe**: `largest_fitting_input_size()` runs one dummy 32-frame window
   at the job's real feed size. If it fails with an out-of-memory error, it steps
   `input_size` down by 56 (4 × 14 px patches) until a window fits. After any
   reduction it drops **one extra step** for headroom. The UI then shows "quality
   auto-reduced…". The model pads every clip to 32 frames, so a 2-frame dummy
   peaks exactly like a real window.
2. **Decode** (thread): skipped frames (fps stride) use `cap.grab()` only. Kept
   frames are resized to `feed_dims()`, the resolution the network computes at.
   The model returns depth at the size it was fed, so full-res frames would only
   inflate memory.
3. **Infer** in chunks of `k × 22` frames, where 22 = the model's window of 32
   minus its internal overlap of 10. The first chunk is 44 frames, so output (and a
   partial on Stop) exists within seconds.
4. **Seam alignment**: each chunk re-infers the previous chunk's last 8 frames. A
   scale + shift fit (`compute_scale_and_shift`) on ~20 k sampled pixels of that
   overlap aligns the new chunk to the previous one.
5. **Normalize**: a running 1st–99th percentile range. It expands instantly
   (nothing clips) and shrinks smoothly, so brightness never jumps at chunk seams.
6. **Encode** (thread): uint8 depth is upscaled to the output size, then turbo LUT
   or grayscale, then `h264_videotoolbox` (falls back to `libx264`).

Decode, infer and encode overlap through bounded queues, so the CPU and GPU work
at the same time without raising peak memory.

---

## Never-crash layers

| Layer | Where | What it does |
|---|---|---|
| MPS allocator cap | env `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` | Oversized allocations raise a clean error and the worker survives. |
| Fit probe | `worker.py` | Lowers quality before the run instead of failing in the middle. |
| Per-chunk OOM retry | `worker.py` | Reclaims the cache and retries the chunk once. |
| Cache reclaim | `worker.py` | `empty_cache()` when the allocator holds over 3 GB, and after every job. |
| RSS ceiling | `app.py` `RSS_CEIL_GB=6` | Worker + ffmpeg over the cap → graceful cancel, partial kept. |
| Pressure watchdog | `app.py` `CRIT_SECS=10` | Sustained macOS *critical* pressure → graceful cancel. |
| Idle release | `app.py` | Critical pressure while idle → the warm model is shut down. |
| Supervisor | `run.sh` | Relaunches the UI in 2 s if it ever dies. |

---

## Measured on M4 / 16 GB (don't re-derive)

- **~2.6–2.8 s per 32-frame window at Balanced (384).** The run is GPU-bound.
- **MPS ceiling = 8.29 GiB**, which is 0.7 × the GPU's ~11.84 GiB recommended
  working set, *not* 0.7 × 16 GB.
- **Best (518) on a 16:9 source does not fit.** Feed size is 924×518, and one window
  needs ~11 GiB in a single `forward()` (6.61 GiB resident + 4.27 GiB requested).
  That is one call, so cache reclaim or retries can't fix it. The probe lands on
  **406** (Best 518 → 462 marginal → 406). Balanced (686×392 feed) peaks at ~6.14 GiB.
- Feeding network-res frames cut peak RSS at 1080p from **6 GB to 1.4 GB**.
- The allocator caches ~6 GB during inference. Idle GPU after a job ≈ **0.13 GB**.
  Kernel compilation is the expensive part; re-allocation is cheap.
- 4K source at Best → 720p output: all 145 frames, peak 1.2 GB, no crash.

---

## Hard constraints (why it runs on Apple Silicon at all)

- Device `mps`, **fp32 only**: fp16 autocast produces NaNs on MPS.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set **before** importing torch.
- **No xformers** (no macOS wheels). The model falls back to standard attention.
- **Read video with OpenCV, never decord.** decord's bundled dylibs register the
  same Objective-C Metal classes as OpenCV and crash an MPS process.
- Keep the two `vda/` patches (see [`vda/VENDORED.md`](../vda/VENDORED.md)).
- `matplotlib.colormaps["turbo"]`: `cm.get_cmap` is gone in matplotlib 3.9 and later.
- Everything is local, with no sudo by default. `purge` (Free memory) asks for the
  password and is opt-in.
- Shell scripts must run on macOS's **bash 3.2** (no associative arrays).

---

## Gotchas

- **A caught exception pins GPU memory.** Inside `except RuntimeError as e:`, the
  traceback keeps the model's MPS tensors alive, so `empty_cache()` there frees
  nothing. Set a flag and reclaim **after** the except block, as the probe and the
  OOM retry do.
- **Stale server on :7860.** `pkill -f app.py` can miss the port owner, so old code
  keeps serving. Restart reliably with
  `lsof -nP -iTCP:7860 -sTCP:LISTEN -t | xargs kill -9 && ./run.sh`.
- **Don't press Free memory from automated tests.** It can open the native admin
  password prompt.

---

## Tunables (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `RSS_CEIL_GB` | `6` | Worker + ffmpeg memory cap before a graceful cancel |
| `CRIT_SECS` | `10` | Seconds of critical pressure before cancelling |
| `TIMEOUT` | `36000` | Max seconds per job |
| `DS_NICE` | `5` | Worker CPU niceness (GPU work is unaffected) |
| `DS_CV2_THREADS` / `DS_TORCH_THREADS` | library default | Thread caps |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.7` | MPS allocator ceiling |

---

## Optimization status

Status of the 14 items from the original optimization audit:

| # | Item | Status |
|---|---|---|
| 1 | Cold start on every run | ✅ persistent warm worker |
| 2 | Full-res frame and depth buffers | ✅ frames fed at network resolution |
| 3 | Silent MPS → CPU op fallback | ⬜ not audited yet |
| 4 | UI killed under pressure | ✅ supervisor, worker respawn, watchdogs |
| 5 | Decode / infer / encode serialized | ✅ threaded pipeline |
| 6 | Overlap frames inferred twice | ◐ chunks aligned to the 22-frame step; 8 seam frames still re-inferred on purpose |
| 7 | Skipped frames fully decoded | ✅ `grab()` only |
| 8 | Software x264 encode | ✅ VideoToolbox (frames still appended one by one) |
| 9 | `empty_cache()` every chunk | ✅ only above 3 GB and once per job |
| 10 | Colorize on CPU at full res | ◐ CPU LUT, but on uint8 at network resolution |
| 11 | Alignment fit on every pixel | ✅ ~20 k sampled pixels |
| 12 | Checkpoint load without mmap | ✅ `mmap=True` |
| 13 | Analytics forking ~5×/s | ✅ `sysctlbyname` via ctypes (chart still rebuilt every 2 s) |
| 14 | Brightness drift across chunks | ✅ robust running range |

`infer.py` is the old one-shot pipeline. It is kept only as the
`bench.py legacy` baseline.
