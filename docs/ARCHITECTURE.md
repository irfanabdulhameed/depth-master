# Architecture

How Depth Studio turns a video into a depth map on a 16 GB Apple Silicon Mac
without ever taking the machine down. Read this before changing `worker.py`,
`app.py` or anything under `vda/`.

**Priority order: never crash → correct output → speed.**

---

## Processes

```
Depth Studio.app  (mac/make_app.sh: starts run.sh if needed, opens the browser)
run.sh  (supervisor: restarts app.py if it dies, exits on a clean quit / exit 0)
  └─ app.py  Gradio UI on :7860 — holds NO torch
       │  watchdogs · device analytics · ETA · settings · batch loop
       │
       │  JSON lines on stdin  ──►   stdout protocol lines  ◄──
       ▼                                  (stderr → worker.log)
     worker.py  persistent, warm — loads each model ONCE
       ├─ decoder thread   OpenCV → RGB at network resolution (trim-aware)
       ├─ main thread      Video-Depth-Anything on MPS, chunk by chunk
       ├─ encoder thread   uint8 → output size → turbo/gray → H.264 (VideoToolbox)
       └─ export pass      raw depth cache → 16-bit PNG / EXR / ProRes 4444
```

- The UI process stays light, so a bad run can only kill the worker. A dead worker is
  respawned on the next run, and a failed job never takes down the warm worker.
- The relative model loads and compiles MPS kernels at launch (warmed at Balanced).
  The metric model loads the first time a job asks for it, and both stay resident.

### Worker protocol

| Direction | Message |
|---|---|
| stdin | `{"cmd":"job","inp","out","grayscale","res","input_size","target_fps","maxf","model","start","end","export"}` |
| stdin | `{"cmd":"cancel"}` stop and finalize a playable partial · `{"cmd":"trim"}` drop caches, stay warm · `{"cmd":"ping"}` · `{"cmd":"quit"}` |
| stdout | `READY` · `INFO …` · `STAGE <name>` · `PROGRESS <done> <total>` · `EXPORT <done> <total>` · `GPU <gb>` · `TRIMMED <before> <after>` · `PONG` |
| stdout | `DONE {"out","frames","secs","peak_gb","partial","export"}` (JSON, so paths with spaces are safe) or `ERROR <msg>` |

`model` is `relative` (default) or `metric`. `start` / `end` are seconds, where
an `end` of 0 means the end of the clip. `export` is `""`, `png16`, `exr` or
`prores`. SIGTERM behaves like `cancel`, then quits.

**Keep stderr out of stdout.** Merged stderr once glued tqdm output onto `PROGRESS`
lines and broke parsing. `bench.py` and the tests follow the same rule.

---

## The job pipeline

1. **Trim**: `cap.set(CAP_PROP_POS_FRAMES, first)` lands on the exact frame (the
   tests check it against a sequential read). The fps stride counts from the trim
   start.
2. **Fit probe**: `largest_fitting_input_size()` runs one dummy 32-frame window
   at the job's real feed size. If that runs out of memory, it steps `input_size`
   down by 56 (4 × 14 px patches) until a window fits. After any reduction it
   drops **one extra step** for headroom, and the UI says so. The model pads every
   clip to 32 frames, so a 2-frame dummy peaks exactly like a real window. With
   fused attention, Best on 16:9 fits outright on 16 GB. The probe still guards
   odd aspect ratios and smaller Macs.
3. **Decode** (thread): skipped frames use `cap.grab()` only. Kept frames are
   resized to `feed_dims()`, the resolution the network computes at. The model
   returns depth at the size it was fed, so full-res frames would only inflate
   memory.
4. **Infer** in chunks of `k × 22` frames, where 22 = the model's window of 32
   minus its internal overlap of 10. The first chunk is 44 frames, so output (and a
   partial on Stop) exists within seconds.
5. **Seam alignment** (relative only): each chunk re-infers the previous chunk's
   last 8 frames. A scale + shift fit on ~20 k sampled pixels of that overlap
   aligns the new chunk. Metric depth is already in meters and needs no fit (the
   model skips its own window alignment for metric too).
6. **Preview normalize**: depth is converted to "nearness" (relative is already
   inverse depth; metric becomes 1/m), then a running 1st–99th percentile range is
   applied. It expands instantly (nothing clips) and shrinks smoothly.
7. **Encode** (thread): uint8 depth is upscaled to the output size, then turbo LUT
   or grayscale, then `h264_videotoolbox` (falls back to `libx264`), at the exact
   output size (`macro_block_size=2`).
8. **Export pass** (only when asked): during the run, raw aligned depth is
   appended to `<out>.raw.f32` at feed resolution. Sparse samples give **one
   global range** (0.1–99.9 percentile) for the whole clip. A second pass reads
   the cache (memmap) and writes:
   - **PNG16**: nearness normalized to 0–65535.
   - **EXR**: raw float, linear-upscaled.
   - **ProRes 4444**: gray16 into `prores_videotoolbox` (12-bit, max error
     35/65535), with `prores_ks` (10-bit) as the fallback.

   A Stop exports what was finished. The cache is deleted afterwards, and free disk
   space is checked before the run starts.

Decode, infer and encode overlap through bounded queues, so the CPU and GPU work
at the same time without raising peak memory.

---

## Never-crash layers

| Layer | Where | What it does |
|---|---|---|
| Fused attention | `vda/…/dinov2_layers/attention.py` | No N×N matrix: ~45% less GPU memory per window. |
| MPS allocator cap | env `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` (+ `LOW=0.5`, required) | Oversized allocations raise a clean error and the worker survives. |
| Fit probe | `worker.py` | Lowers quality before the run instead of failing in the middle. |
| Per-chunk OOM retry | `worker.py` | Reclaims the cache and retries the chunk once. |
| Cache reclaim | `worker.py` | `empty_cache()` when the allocator holds over 3 GB, and after every job. |
| Disk check | `worker.py` | An export that won't fit on disk is refused up front. |
| RSS ceiling | `app.py` `RSS_CEIL_GB=6` | Worker + ffmpeg over the cap → graceful cancel, partial kept. |
| Pressure watchdog | `app.py` `CRIT_SECS=10` | Sustained macOS *critical* pressure → graceful cancel. |
| Idle release | `app.py` | Critical pressure while idle → the warm model is shut down. |
| Supervisor | `run.sh` | Relaunches the UI in 2 s if it ever dies. Exit 0 (Quit) stops it. |

---

## Measured on M4 / 16 GB (don't re-derive)

- **MPS ceiling = 8.29 GiB**, which is 0.7 × the GPU's 11.84 GiB recommended
  working set, *not* 0.7 × 16 GB.
- **One window (32 frames, 16:9), fused attention:**
  - Balanced (feed 686×392): peak 3.22 GiB.
  - Best (924×518): peak 5.89 GiB, so it fits.
  - Before the change, Balanced peaked at 5.73 GiB, and Best needed ~11 GiB in one
    `forward()` and could not run.
- **Bundled clip (70 frames, 960×540), `bench.py worker`:**
  - Balanced: 12.8 s warm, 6.3 fps steady, 0.46 GB peak RSS.
  - Best: 25.1 s warm, 3.2 fps steady.
  - Before the attention change, Balanced took 16.8 s warm.
- Feeding network-res frames cut peak RSS at 1080p from 6 GB to 1.4 GB.
- The allocator caches several GB during inference. Idle GPU after a job is
  ≈ 0.13 GB. Kernel compilation is the expensive part; re-allocation is cheap.

## Experiments that didn't make it (so nobody repeats them blind)

| Idea | Result | Verdict |
|---|---|---|
| Run the encoder 8 frames at a time | Exact output, but slower and more memory than fused attention (Best 13.9 s / 8.14 GiB vs 9.3 s / 5.89 GiB per 2 windows) | Dropped |
| Fused attention in the temporal head too | More memory (Best 8.13 GiB) and slower | Dropped; encoder only |
| `torch.compile` (encoder / head) | Encoder no faster (4.7–5.0 s vs 4.5 s), head slower (5.7 s) | Not used |
| Audit CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=0`) | Whole pipeline runs: **no op falls back to CPU** | Flag kept on only as a safety net |
| Upstream streaming model as a fast preview | Slower (12.0 vs 15.1 fps at Fast, 7.1 vs 8.0 at Balanced); lower memory (1.4 GiB); 0.6% from offline | Preview = offline model on a 3 s clip instead |
| CoreML encoder | Neural Engine 0.63× (slower); GPU fp32 0.96×; GPU fp16 1.38× on the encoder alone, which is only ~15% end to end after extra copies | Not used, see below |

CoreML details: coremltools 9.0 is untested with torch 2.13 and needed two
workarounds to convert at all. NumPy 2 broke a scalar cast in its torch frontend,
and bicubic resizing of the position embeddings is unsupported, so they have to
be frozen per input shape. Beyond that, CoreML memory sits **outside the MPS cap**
that the never-crash guarantee relies on. The only faster mode needs fp16, the
precision this project avoids on Apple GPUs. And every quality × aspect ratio
would need its own compiled model. Revisit only if a coremltools release supports
the current torch and fp32 becomes faster.

---

## Hard constraints (why it runs on Apple Silicon at all)

- Device `mps`, **fp32 only**: fp16 autocast produces NaNs on MPS.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set **before** importing torch.
- **No xformers** (no macOS wheels). Keep the fused attention in
  `dinov2_layers/attention.py`; without it Best doesn't fit on 16 GB.
- **Read video with OpenCV, never decord.** decord's bundled dylibs register the
  same Objective-C Metal classes as OpenCV and crash an MPS process.
- `OPENCV_IO_ENABLE_OPENEXR=1` before importing cv2, or EXR writing is disabled.
- Keep the `vda/` patches (see [`vda/VENDORED.md`](../vda/VENDORED.md)).
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
- **imageio's `macro_block_size=16` stretches frames** (1080 → 1088 rows). The
  depth then no longer lines up with the source. Use 2 for yuv420p.
- **Hardware ProRes rejects imageio's default `quality`.** Pass `quality=None`.
- **Stale server on :7860.** `pkill -f app.py` can miss the port owner, so old code
  keeps serving. Restart reliably with
  `lsof -nP -iTCP:7860 -sTCP:LISTEN -t | xargs kill -9 && ./run.sh`.
- **Don't press Free memory from automated tests.** It can open the native admin
  password prompt.
- **Buttons aren't API outputs in Gradio 6.** `/generate` returns
  `(video, stats, batch files, last path)`, and `/preview` returns
  `(video, stats, last path)`.

---

## Tunables (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `DS_PORT` | `7860` | UI port (the tests use 7861) |
| `DS_SETTINGS` | `.settings.json` | Where last-used settings live (the tests use a temp file) |
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
| 3 | Silent MPS → CPU op fallback | ✅ audited: none |
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
| 14 | Brightness drift across chunks | ✅ robust running range (preview); one global range (16-bit exports) |

`infer.py` is the old one-shot pipeline. It is kept only as the
`bench.py legacy` baseline.
