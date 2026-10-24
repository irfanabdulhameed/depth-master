# Vendored: Video-Depth-Anything

| | |
|---|---|
| Upstream | https://github.com/DepthAnything/Video-Depth-Anything |
| Commit | `4f5ae23172ba60fd7bc11ef671cca678842c7072` (2025-10-07) |
| Code license | Apache-2.0 (`LICENSE`) |
| Weights used | Video-Depth-Anything-**Small** (`vits`) — Apache-2.0 |

Base and Large weights are **CC-BY-NC-4.0 (non-commercial)** — check the license
before switching encoders.

## Local patches

Two files differ from upstream so the model runs on Apple Silicon (MPS). The full
diff is in [`patches/apple-silicon.patch`](patches/apple-silicon.patch).

- `video_depth_anything/dpt_temporal.py` — autocast uses the tensor's own device
  instead of a hardcoded `"cuda"`.
- `video_depth_anything/video_depth.py` — `fp32=True` skips autocast entirely
  (fp16 autocast produces NaNs on MPS), and `infer_video_depth()` takes a
  `progress_callback(window, total_windows)` used for progress, ETA and Stop.

## Not in the repo

- `checkpoints/*.pth` — downloaded and checksum-verified by `../setup.sh`.
- `assets/teaser_video_v2.png` — upstream README image only.

## Updating to a newer upstream

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything /tmp/vda-new
cd /tmp/vda-new && git apply /path/to/depth-master/vda/patches/apple-silicon.patch
```

Then copy the tree over `vda/` (keep `checkpoints/`, `patches/` and this file) and
re-run `bench.py worker` to compare speed and peak memory before committing.
