# Depth Studio

Turn any video into a **depth-map video**, locally on your Mac's GPU. Near
surfaces glow warm and far ones cool, or choose clean black-to-white grayscale.
Nothing is uploaded and no cloud GPU is needed.

Built on [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
and tuned to run on a **16 GB Apple Silicon Mac** without ever freezing it.

---

## Quick start

```bash
git clone https://github.com/irfanabdulhameed/depth-master.git
cd depth-master
./setup.sh      # once: Python env, dependencies, model weights
./run.sh        # then open http://127.0.0.1:7860
```

Drop a video in, pick your settings and click **Generate depth**. Results are
saved in `outputs/`.

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1 or later). Tuned on **M4, 16 GB**. |
| Python | 3.11 (`brew install python@3.11`) |
| Disk | ~1.2 GB (environment ~1.1 GB, model weights 116 MB) |

Intel Macs, Windows and Linux fall back to the CPU, which is far too slow to
be practical.

---

## Controls

| Control | Options | Notes |
|---|---|---|
| Speed / quality | Fast · **Balanced** · Best | Fast is ~3× quicker. Best is lowered automatically if it won't fit in memory, and the result line says so. |
| Process fps | Original · 24 · 15 | Fewer frames, proportionally faster. |
| Output resolution | Match input · 1080p · 720p · 540p | Depth is computed at the model's resolution and then upscaled, so a larger output costs little. |
| Grayscale | on / off | Black-to-white depth instead of the colour map. |

- **Stop** ends a run at any point and keeps a playable partial video.
- **Free memory** drops the model's caches and purges inactive macOS memory. The
  purge may ask for your password.
- **Device usage** shows live CPU, RAM, compressed memory, swap, GPU use and a
  🟢 🟡 🔴 headroom light based on the system's real memory-pressure level.
- Your last settings are remembered between sessions.

---

## Why it doesn't crash or crawl

- **Warm engine.** The model loads in the background as soon as the app starts
  and stays ready, so every Generate starts warm.
- **Any length, fixed memory.** Video is processed in chunks that are aligned at
  the seams, so long clips use no more memory than short ones.
- **Quality fits the machine.** Before each run it tests the chosen quality
  against the GPU memory limit and lowers it if needed. It never fails halfway
  through.
- **Safety nets.** GPU memory is capped, and watchdogs stop a run gracefully,
  keeping the partial video, long before macOS would kill anything. If the UI
  itself ever dies, `run.sh` restarts it.
- **CPU and GPU work in parallel.** Decoding, depth estimation and hardware
  (VideoToolbox) encoding run at the same time.

## Performance (M4, 16 GB)

| | |
|---|---|
| Bundled test clip (70 frames, 960×540, Balanced) | **16.8 s warm** (4.2 fps) · 39 s including model startup |
| Peak memory on that clip | 0.75 GB |
| Model speed | ~2.7 s per 32-frame window at Balanced |
| Peak memory, 1080p source | ~1.4 GB |
| Best quality, 16:9 source | Doesn't fit on 16 GB, so it runs automatically at a level between Balanced and Best |
| GPU memory when idle | ~0.13 GB |

To measure on your machine, run a cold and then a warm pass with timings and
peak memory:

```bash
.venv/bin/python bench.py worker                    # bundled test clip
.venv/bin/python bench.py worker --clip my.mp4 --input-size 294
```

---

## Project layout

```
run.sh              Start here: launches the UI and restarts it if it dies
setup.sh            One-time setup (venv, pinned deps, verified weights)
app.py              Web UI, device monitor, watchdogs (no torch in this process)
worker.py           Warm depth engine: chunked, memory-bounded, pipelined
bench.py            Headless benchmark (cold vs warm, peak memory)
infer.py            Old one-shot pipeline, kept only as the benchmark baseline
docs/ARCHITECTURE.md  How it works, measured limits, rules not to break
vda/                Video-Depth-Anything model code with Apple Silicon patches
```

Generated files (`outputs/`, `worker.log`, `.settings.json`) and the model weights
are not committed.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Not set up yet" | Run `./setup.sh`. |
| The first run is slow | The model is still loading. Give it ~20 s after launch, and later runs reuse the warm engine. |
| A run stops with a memory message | Try **Fast**, a lower fps or a smaller output resolution. The partial video is kept. |
| An old version is still running, or port 7860 is busy | `lsof -nP -iTCP:7860 -sTCP:LISTEN -t \| xargs kill -9 && ./run.sh` |
| Something else went wrong | Look in `worker.log` for the engine's own output. |

---

## Credits and licenses

The depth model is **Video-Depth-Anything-Small** by ByteDance, under
Apache-2.0 (code and weights). The Base and Large weights are non-commercial
(CC-BY-NC-4.0). See [`vda/VENDORED.md`](vda/VENDORED.md) for the exact upstream
version and the local patches.
