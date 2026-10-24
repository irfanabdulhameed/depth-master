"""Regenerates every image the README uses, from real Depth Studio output.

    .venv/bin/python docs/make_images.py          # ~2 min, uses the GPU

Runs the warm worker on the bundled Tokyo clip (Best, 16-bit PNG export, plus a
grayscale MP4), then writes docs/images/:
  hero.webp                 source | depth, animated
  bit-depth.webp            one street patch from the 8-bit MP4 and the 16-bit
                            master, contrast-stretched the same way
  icon.png                  the app icon (mac/icon.py)
  ui-light.webp, ui-dark.webp   the app with that result loaded (headless Chrome)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
OUT = os.path.join(ROOT, "docs", "images")
CLIP = os.path.join(ROOT, "vda", "assets", "example_videos", "Tokyo-Walk_rgb.mp4")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 7862

HERO_FRAMES = range(60, 120)     # 4 s at 15 fps
HERO_W = 480                     # per panel
BAND_FRAME = 90
BAND_BOX = (40, 470, 200, 570)   # x0, y0, x1, y1 on the 1280x720 frame: smooth street, no people
BAND_ZOOM = 3                    # nearest-neighbour, so steps stay steps


def run_worker(jobs):
    """Run jobs on one warm worker; returns the DONE dicts."""
    p = subprocess.Popen([PY, "-u", os.path.join(ROOT, "worker.py")], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    results = []
    try:
        for job in jobs:
            p.stdin.write(json.dumps({"cmd": "job", "grayscale": False, "res": 0, "target_fps": 0,
                                      "maxf": 0, **job}) + "\n")
            p.stdin.flush()
            for line in p.stdout:
                if line.startswith("DONE "):
                    results.append(json.loads(line[5:]))
                    break
                if line.startswith("ERROR"):
                    sys.exit(f"worker: {line.strip()}")
            print(f"  {os.path.basename(job['out'])}: {results[-1]['frames']} frames in {results[-1]['secs']}s")
    finally:
        p.stdin.write('{"cmd":"quit"}\n')
        p.stdin.flush()
        p.wait(timeout=20)
    return results


def frames(path, idx):
    cap, out, i = cv2.VideoCapture(path), {}, 0
    want = set(idx)
    while i <= max(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    return [out[i] for i in idx]


def hero(depth_mp4):
    h = round(HERO_W * 720 / 1280)
    src, dep = frames(CLIP, HERO_FRAMES), frames(depth_mp4, HERO_FRAMES)
    path = os.path.join(OUT, "hero.webp")
    w = imageio_ffmpeg.write_frames(path, (HERO_W * 2, h), fps=15, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                    codec="libwebp_anim", quality=None, macro_block_size=1,
                                    output_params=["-loop", "0", "-quality", "72", "-compression_level", "6"])
    w.send(None)
    for s, d in zip(src, dep):
        pair = np.hstack([cv2.resize(s, (HERO_W, h), interpolation=cv2.INTER_AREA),
                          cv2.resize(d, (HERO_W, h), interpolation=cv2.INTER_AREA)])
        w.send(np.ascontiguousarray(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)).tobytes())
    w.close()
    return path


def bit_depth(gray_mp4, png_dir):
    """Same patch, same stretch (what Levels does to a fog/DOF matte): the 8-bit MP4
    breaks into bands, the 16-bit master stays a smooth ramp. Shown in the app's turbo map."""
    x0, y0, x1, y1 = BAND_BOX
    eight = cv2.cvtColor(frames(gray_mp4, [BAND_FRAME])[0], cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float64)
    sixteen = cv2.imread(os.path.join(png_dir, f"depth_{BAND_FRAME + 1:06d}.png"),
                         cv2.IMREAD_UNCHANGED)[y0:y1, x0:x1].astype(np.float64)
    lut = (np.asarray(matplotlib.colormaps["turbo"](np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)

    def panel(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        u8 = (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        rgb = lut[cv2.resize(u8, None, fx=BAND_ZOOM, fy=BAND_ZOOM, interpolation=cv2.INTER_NEAREST)]
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])

    a, b = panel(eight), panel(sixteen)
    gap = np.zeros((a.shape[0], 16, 4), np.uint8)              # transparent: works on light and dark pages
    path = os.path.join(OUT, "bit-depth.webp")
    Image.fromarray(np.hstack([a, gap, b]), "RGBA").save(path, quality=92, method=6)
    return path


def icon():
    tmp = tempfile.mkdtemp()
    subprocess.run([PY, os.path.join(ROOT, "mac", "icon.py"), tmp], check=True)
    shutil.copy(os.path.join(tmp, "icon_128x128@2x.png"), os.path.join(OUT, "icon.png"))
    shutil.rmtree(tmp)


def screenshots(depth_mp4, stat, tmp):
    if not os.path.exists(CHROME):
        print("  Google Chrome not found — skipping UI screenshots")
        return
    env = {**os.environ, "DS_PORT": str(PORT), "DS_SETTINGS": os.path.join(tmp, "settings.json"),
           "GRADIO_ANALYTICS_ENABLED": "False", "GRADIO_TEMP_DIR": os.path.join(tmp, "gradio")}
    server = subprocess.Popen([PY, "-c", SERVER, CLIP, depth_mp4, stat], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
                break
            except Exception:
                time.sleep(1)
        time.sleep(3)
        for theme in ("light", "dark"):
            raw = os.path.join(tmp, f"{theme}.png")
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                            f"--user-data-dir={os.path.join(tmp, 'chrome')}", "--window-size=1440,1245",
                            "--force-device-scale-factor=1", "--virtual-time-budget=15000",
                            f"--screenshot={raw}", f"http://127.0.0.1:{PORT}/?__theme={theme}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            im = Image.open(raw).convert("RGB")
            im = im.crop((96, 0, 1344, im.height - 8))       # trim side margins and the next panel's edge
            im.save(os.path.join(OUT, f"ui-{theme}.webp"), quality=90, method=6)
    finally:
        server.terminate()
        server.wait(timeout=20)


# The app as-is, with one load event that fills in a finished run for the screenshot.
SERVER = r"""
import sys, gradio as gr
sys.path.insert(0, ".")
import app
src, depth, stat = sys.argv[1:4]
with app.demo:
    app.demo.load(lambda: (src, depth, stat, "Best", "16-bit PNG sequence", gr.Button(interactive=True)),
                  outputs=[app.inp, app.out, app.run_stat, app.speed, app.export, app.finder_btn])
app.demo.queue().launch(server_name="127.0.0.1", server_port=app.PORT,
                        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"))
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    try:
        print("depth (Best):")
        color, gray = run_worker([
            {"inp": CLIP, "out": os.path.join(tmp, "Tokyo-Walk_depth.mp4"), "input_size": 518, "export": "png16"},
            {"inp": CLIP, "out": os.path.join(tmp, "Tokyo-Walk_gray.mp4"), "input_size": 518, "grayscale": True,
             "maxf": BAND_FRAME + 2},
        ])
        print("images:")
        print("  " + hero(color["out"]))
        print("  " + bit_depth(gray["out"], color["export"]))
        icon()
        stat = (f"✅ done · {color['frames']} frames in {color['secs']:.0f}s "
                f"({color['frames'] / max(color['secs'], 0.1):.1f} fps) · peak {color['peak_gb']:.1f} GB · "
                f"h264_videotoolbox · 16-bit PNG saved")
        screenshots(color["out"], stat, tmp)
        for f in sorted(os.listdir(OUT)):
            print(f"  {f:16s} {os.path.getsize(os.path.join(OUT, f)) / 1024:7.0f} KB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
