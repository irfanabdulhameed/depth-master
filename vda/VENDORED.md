# Vendored: Video-Depth-Anything

| | |
|---|---|
| Upstream | https://github.com/DepthAnything/Video-Depth-Anything |
| Commit | `4f5ae23172ba60fd7bc11ef671cca678842c7072` (2025-10-07) |
| Code license | Apache-2.0 (`LICENSE`) |
| Weights used | Video-Depth-Anything-**Small** and Metric-Video-Depth-Anything-**Small** (`vits`), both Apache-2.0 |

Base and Large weights are **CC-BY-NC-4.0 (non-commercial)**. Check the license
before switching encoders.

## Local patches

Three files differ from upstream, and each says so in a "Modified for Depth
Studio" line in its header (Apache-2.0 §4b). The full diff is in
[`patches/apple-silicon.patch`](patches/apple-silicon.patch); it applies cleanly
to the commit above. The rest of the project is MIT ([`../LICENSE`](../LICENSE)),
but everything in `vda/` stays under its own Apache-2.0 license.

- `video_depth_anything/dinov2_layers/attention.py`: the encoder's attention
  uses `F.scaled_dot_product_attention` instead of `q @ k.T`. It never builds the
  N×N score matrix, so on MPS it cuts peak memory per window ~45% and time ~30%
  (output identical to 1.6e-5 of the depth range). This is what lets Best (518)
  fit on a 16 GB Mac.
- `video_depth_anything/dpt_temporal.py`: autocast uses the tensor's own device
  instead of a hardcoded `"cuda"`.
- `video_depth_anything/video_depth.py`: `fp32=True` skips autocast entirely
  (fp16 autocast produces NaNs on MPS), and `infer_video_depth()` takes a
  `progress_callback(window, total_windows)` used for progress, ETA and Stop.

## Not in the repo

- `checkpoints/*.pth`: downloaded and checksum-verified by `../setup.sh`.
- `assets/teaser_video_v2.png`: upstream README image only.

## Updating to a newer upstream

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything /tmp/vda-new
cd /tmp/vda-new && git apply /path/to/depth-master/vda/patches/apple-silicon.patch
```

Then copy the tree over `vda/` (keep `checkpoints/`, `patches/` and this file),
run `pytest tests/` and `bench.py worker`, and compare speed and peak memory
before committing.
