# Depth Studio

Turn any video into a **depth-map video**, locally on your Mac's GPU. Near
surfaces glow warm and far ones cool, or choose clean black-to-white grayscale.
When you need it for compositing, it can also save a 16-bit master (PNG, EXR or
ProRes 4444). Nothing is uploaded and no cloud GPU is needed.

Built on [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
and tuned to run on a **16 GB Apple Silicon Mac** without ever freezing it.

---

## Quick start

```bash
git clone https://github.com/irfanabdulhameed/depth-master.git
cd depth-master
./setup.sh            # once: Python env, dependencies, model weights
mac/make_app.sh       # optional: adds "Depth Studio" to ~/Applications
```

Then double-click **Depth Studio** in Applications, or run `./run.sh` and open
http://127.0.0.1:7860. Drop a video in and click **Generate depth**. Results are
saved in `outputs/`, named after the source clip.

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1 or later). Tuned on **M4, 16 GB**. |
| Python | 3.11 (`brew install python@3.11`) |
| Disk | ~1.4 GB (environment ~1.1 GB, two model weights at 116 MB each) |

Intel Macs, Windows and Linux fall back to the CPU, which is far too slow to
be practical.

---

## Controls

| Control | Options | Notes |
|---|---|---|
| **Clip / Batch** | tabs | Batch runs every dropped file with the same settings, one after another. |
| Trim start / end | seconds | Process only part of a clip. An end of 0 means the end of the clip. |
| Speed / quality | Fast · **Balanced** · Best | Fast is ~3× quicker. Best runs at full quality on 16 GB. |
| Process fps | Original · 24 · 15 | Fewer frames, proportionally faster. |
| Output resolution | Match input · 1080p · 720p · 540p | Output is pixel-for-pixel the source size, so it lines up in a composite. |
| Depth type | **Relative** · Metric | Metric gives real distance in meters. Save it as EXR to keep the values. |
| 16-bit master | None · PNG · EXR · ProRes | See [Exports](#exports). |
| Grayscale | on / off | Black-to-white depth instead of the colour map. |

- **Preview 3 s** renders three seconds from the trim start at 540p, so you can
  check a setting before committing to a full run.
- **Stop** ends a run at any point and keeps a playable partial. In Batch it also
  stops the remaining clips.
- **Show in Finder** reveals the last result.
- **Quit** shuts down Depth Studio cleanly.
- **Free memory** drops the model's caches and purges inactive macOS memory. The
  purge may ask for your password.
- **Device usage** shows live CPU, RAM, compressed memory, swap, GPU use and a
  🟢 🟡 🔴 headroom light based on the system's real memory-pressure level.
- Your last settings are remembered between sessions.

## Exports

The MP4 preview is 8-bit, which is fine to watch but bands when you use it as a
displacement, depth-of-field or fog matte. A 16-bit master avoids that:

| Format | Saved as | Values |
|---|---|---|
| **16-bit PNG** | `<clip>_depth_<n>_png16/depth_000001.png …` | 0–65535, near = white |
| **EXR** | `<clip>_depth_<n>_exr/depth_000001.exr …` | Raw 32-bit float: relative inverse depth, or **meters** with Metric |
| **ProRes 4444** | `<clip>_depth_<n>_prores.mov` | 12-bit grayscale (hardware encoder), near = white |

PNG and ProRes are normalized with **one brightness range for the whole clip**,
so the depth of a surface never drifts between frames. Exports cache raw depth
to disk during the run, and Depth Studio checks there is enough free space before
it starts.

---

## Why it doesn't crash or crawl

- **Warm engine.** The model loads in the background as soon as the app starts
  and stays ready, so every Generate starts warm.
- **Any length, fixed memory.** Video is processed in chunks that are aligned at
  the seams, so long clips use no more memory than short ones.
- **Lean attention.** Fused attention never builds the full attention matrix.
  It cut GPU memory per window by ~45% and time by ~30%, which is what lets Best
  run at full quality on 16 GB.
- **Quality fits the machine.** Before each run it tests the chosen quality
  against the GPU memory limit and lowers it only if it truly won't fit. It never
  fails halfway through.
- **Safety nets.** GPU memory is capped, and watchdogs stop a run gracefully,
  keeping the partial video, long before macOS would kill anything. If the UI
  itself ever dies, `run.sh` restarts it.
- **CPU and GPU work in parallel.** Decoding, depth estimation and hardware
  (VideoToolbox) encoding run at the same time.

## Performance (M4, 16 GB)

Bundled test clip: 70 frames, 960×540.

| | Before | Now |
|---|---|---|
| Balanced, warm | 16.8 s | **12.8 s** (5.5 fps) |
| Balanced, including model startup | 39.4 s | **19.3 s** |
| Best | lowered automatically to fit | **full quality**: 25.1 s warm (2.8 fps) |
| Peak memory (Balanced) | 0.75 GB | **0.46 GB** |
| GPU memory per window, Balanced / Best | 5.7 GiB / doesn't fit | 3.2 GiB / 5.9 GiB (limit 8.3) |

To measure on your machine, run a cold and then a warm pass with timings and
peak memory:

```bash
.venv/bin/python bench.py worker                    # bundled test clip, Balanced
.venv/bin/python bench.py worker --clip my.mp4 --input-size 518
```

---

## Project layout

```
run.sh                Start here: launches the UI and restarts it if it dies
setup.sh              One-time setup (venv, pinned deps, checksum-verified weights)
app.py                Web UI, device monitor, watchdogs (no torch in this process)
worker.py             Warm depth engine: chunked, pipelined, 16-bit export pass
bench.py              Headless benchmark (cold vs warm, steady fps, peak memory)
infer.py              Old one-shot pipeline, kept only as the benchmark baseline
mac/                  make_app.sh builds Depth Studio.app; icon.py draws its icon
tests/                End-to-end tests of the engine and the web UI
docs/ARCHITECTURE.md  How it works, measured limits, experiments, rules not to break
vda/                  Video-Depth-Anything model code with local patches
```

Generated files (`outputs/`, logs, `.settings.json`) and the model weights are
not committed.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q        # ~2 min, uses the GPU
```

They cover the worker protocol, correct depth orientation, exact output size,
trim and seek accuracy, Stop with a partial, every export format, metric depth,
Best fitting in 16 GB, and the web UI end to end (generate, preview, batch,
validation, Quit). The UI tests run on port 7861 and use their own settings
file, so an open Depth Studio and your saved settings are never touched.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Not set up yet", or the app offers to run setup | Run `./setup.sh`. |
| The app says the project folder moved | Run `mac/make_app.sh` again from the new location. |
| The first run is slow | The model is still loading. Give it ~10 s after launch, and later runs reuse the warm engine. |
| A run stops with a memory message | Try **Fast**, a lower fps or a smaller output resolution. The partial video is kept. |
| "not enough disk space for the export" | Free some space, choose a smaller output resolution or trim the clip. |
| An old version is still running, or port 7860 is busy | `lsof -nP -iTCP:7860 -sTCP:LISTEN -t \| xargs kill -9 && ./run.sh` |
| Something else went wrong | Look in `worker.log` for the engine's own output. |

---

## Credits and licenses

The depth models are **Video-Depth-Anything-Small** and
**Metric-Video-Depth-Anything-Small** by ByteDance, under Apache-2.0 (code and
weights). The Base and Large weights are non-commercial (CC-BY-NC-4.0) and are
not used. See [`vda/VENDORED.md`](vda/VENDORED.md) for the exact upstream version
and the local patches.
