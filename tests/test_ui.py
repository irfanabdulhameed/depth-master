"""End-to-end tests through the real web UI (gradio_client): generate with trim +
16-bit export, preview, batch, input validation, and Quit. The app runs on a spare
port so a Depth Studio you have open on :7860 is never touched. Never presses
Free memory — it can open the macOS admin password prompt."""
import os
import shutil
import subprocess
import time
import urllib.request

import pytest
from gradio_client import Client, handle_file

from conftest import DAVIS, PY, ROOT, TOKYO, frame_count

PORT = 7861
URL = f"http://127.0.0.1:{PORT}/"
OUTDIR = os.path.join(ROOT, "outputs")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.makedirs(OUTDIR, exist_ok=True)
    before = set(os.listdir(OUTDIR))
    env = {**os.environ, "DS_PORT": str(PORT),
           "DS_SETTINGS": str(tmp_path_factory.mktemp("settings") / "settings.json"),  # keep the user's
           "GRADIO_TEMP_DIR": str(tmp_path_factory.mktemp("gradio")), "GRADIO_ANALYTICS_ENABLED": "False"}
    log = open(tmp_path_factory.mktemp("logs") / "app.log", "ab")
    proc = subprocess.Popen([PY, os.path.join(ROOT, "app.py")], cwd=ROOT, env=env, stdout=log, stderr=log)
    end = time.time() + 120
    while time.time() < end:
        try:
            urllib.request.urlopen(URL, timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail("UI did not come up")
    client = Client(URL, verbose=False)
    yield proc, client
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=15)
    for name in set(os.listdir(OUTDIR)) - before:        # leave outputs/ as we found it
        p = os.path.join(OUTDIR, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def path(v):
    return v["video"] if isinstance(v, dict) else v


# The Show-in-Finder button isn't an API output: /generate returns
# (video, stats, batch files, last path) and /preview (video, stats, last path).
def generate(client, mode="clip", video=None, files=None, start=0, end=0, grayscale=False,
             speed="Fast", fps="Original", res="540p (fastest)", depth="Relative",
             export="None (MP4 preview only)"):
    return client.predict(mode, handle_file(video) if video else None,
                          [handle_file(f) for f in files] if files else None,
                          start, end, grayscale, speed, fps, res, depth, export, api_name="/generate")


def test_generate_with_trim_and_png16(app):
    _, client = app
    video, stat, _, last = generate(client, video=DAVIS, end=1.5, export="16-bit PNG sequence")
    assert stat.startswith("✅ done") and "16-bit PNG saved" in stat
    assert frame_count(path(video)) == 36                      # 1.5 s at 24 fps
    assert last.endswith("_png16") and len(os.listdir(last)) == 36
    assert os.path.basename(last).startswith("davis_rollercoaster_depth_")


def test_preview_is_three_seconds_at_540p(app):
    _, client = app
    video, stat, last = client.predict(handle_file(DAVIS), 0.5, False, "Fast", "Original",
                                          "Relative", api_name="/preview")
    assert stat.startswith("preview · ✅ done")
    assert frame_count(path(video)) == 58                      # frames 12..69: 3 s window clipped at the end
    import cv2
    assert int(cv2.VideoCapture(path(video)).get(cv2.CAP_PROP_FRAME_WIDTH)) == 540
    assert "_preview_" in last


def test_batch_processes_every_clip(app):
    _, client = app
    video, stat, files, last = generate(client, mode="batch", files=[DAVIS, TOKYO], fps="15 fps")
    assert "2/2 clips done" in stat
    files = files["value"] if isinstance(files, dict) else files
    assert len(files) == 2 and all(os.path.exists(f) for f in files)


def test_trim_validation_message(app):
    _, client = app
    with pytest.raises(Exception, match="Trim end must be after the start"):
        generate(client, video=DAVIS, start=2, end=1)


def test_empty_batch_message(app):
    _, client = app
    with pytest.raises(Exception, match="Add one or more videos"):
        generate(client, mode="batch")


def test_quit_stops_server_cleanly(app):
    proc, client = app
    msg = client.predict(api_name="/quit")
    assert "has quit" in msg
    assert proc.wait(timeout=20) == 0                             # exit 0 → run.sh won't restart
