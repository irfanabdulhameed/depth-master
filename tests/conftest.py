"""Shared fixtures: one warm worker process for the whole session (loading the
model once keeps the suite to a couple of minutes), driven over the same JSON-lines
protocol app.py uses."""
import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
CLIPS = os.path.join(ROOT, "vda", "assets", "example_videos")
DAVIS = os.path.join(CLIPS, "davis_rollercoaster.mp4")       # 70 frames, 960x540 @ 24
TOKYO = os.path.join(CLIPS, "Tokyo-Walk_rgb.mp4")             # 195 frames, 1280x720 @ 15
WEIGHTS = os.path.join(ROOT, "vda", "checkpoints", "video_depth_anything_vits.pth")
METRIC_WEIGHTS = os.path.join(ROOT, "vda", "checkpoints", "metric_video_depth_anything_vits.pth")

if not os.path.exists(WEIGHTS):
    pytest.skip("model weights missing — run ./setup.sh", allow_module_level=True)


class Worker:
    def __init__(self, log_path):
        self.proc = subprocess.Popen([PY, "-u", os.path.join(ROOT, "worker.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(log_path, "ab"), text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.expect("READY", timeout=180)

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip())
        self.lines.put("__EOF__")

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def expect(self, prefix, timeout=60):
        """Collect lines until one starts with `prefix`; returns (that line, all lines)."""
        seen, end = [], time.time() + timeout
        while True:
            line = self.lines.get(timeout=max(0.1, end - time.time()))
            seen.append(line)
            if line == "__EOF__":
                raise AssertionError(f"worker exited while waiting for {prefix}: {seen[-5:]}")
            if line.startswith(prefix):
                return line, seen

    def job(self, inp, out, timeout=300, on_line=None, **kw):
        """Run one job. Returns (result dict | None, error str | None, all lines)."""
        self.send({"cmd": "job", "inp": inp, "out": str(out), "grayscale": False, "res": 0,
                   "input_size": 294, "target_fps": 0, "maxf": 0, **kw})
        seen, end = [], time.time() + timeout
        while True:
            line = self.lines.get(timeout=max(0.1, end - time.time()))
            seen.append(line)
            if on_line:
                on_line(line)
            if line == "__EOF__":
                raise AssertionError(f"worker died: {seen[-5:]}")
            if line.startswith("DONE "):
                return json.loads(line[5:]), None, seen
            if line.startswith("ERROR"):
                return None, line[6:], seen

    def close(self):
        if self.proc.poll() is None:
            self.send({"cmd": "quit"})
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    w = Worker(tmp_path_factory.mktemp("logs") / "worker.log")
    yield w
    w.close()


def frame_count(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


sys.path.insert(0, ROOT)
