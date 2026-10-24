"""Black-box tests of the warm worker: protocol, output correctness, Stop,
trim, 16-bit exports, metric depth and the memory fit on Best."""
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from conftest import DAVIS, METRIC_WEIGHTS, TOKYO, frame_count


def test_ping(worker):
    worker.send({"cmd": "ping"})
    worker.expect("PONG", timeout=10)


def test_basic_job_matches_input(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "d.mp4", maxf=44)
    assert err is None and res["partial"] == 0 and res["frames"] == 44
    cap = cv2.VideoCapture(res["out"])
    assert (int(cap.get(3)), int(cap.get(4))) == (960, 540)
    assert frame_count(res["out"]) == 44


def test_depth_looks_right(worker, tmp_path):
    """Sky (blue pixels in the source) must come out far = dark; the coaster near = bright."""
    res, err, _ = worker.job(DAVIS, tmp_path / "g.mp4", maxf=44, grayscale=True)
    assert err is None
    cap = cv2.VideoCapture(res["out"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    g = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(int)
    src = cv2.VideoCapture(DAVIS)
    src.set(cv2.CAP_PROP_POS_FRAMES, 30)
    b, gr, r = cv2.split(src.read()[1].astype(int))
    sky = (b > 150) & (b > r + 60) & (b > gr + 10)
    assert sky.mean() > 0.2                               # the frame really has sky
    assert np.median(g[~sky]) > np.median(g[sky]) + 80


def test_trim_frame_count_and_seek(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "t.mp4", start=1.0, end=2.0)
    assert err is None and res["frames"] == 24          # 1 s at 24 fps


def test_seek_is_frame_accurate():
    cap = cv2.VideoCapture(DAVIS)
    for _ in range(36):
        cap.grab()
    ok, seq = cap.read()                                  # frame 36, sequential
    cap2 = cv2.VideoCapture(DAVIS)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 36)
    ok2, sought = cap2.read()
    assert ok and ok2 and np.array_equal(seq, sought)


def test_trim_invalid_is_clean_error(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "x.mp4", start=5.0, end=2.0)
    assert res is None and "trim" in err


def test_bad_input_keeps_worker_alive(worker, tmp_path):
    res, err, _ = worker.job(str(tmp_path / "missing.mp4"), tmp_path / "m.mp4")
    assert res is None and "cannot open" in err
    worker.send({"cmd": "ping"})
    worker.expect("PONG", timeout=10)


def test_stop_keeps_playable_partial(worker, tmp_path):
    sent = []

    def stop_after_first_chunk(line):
        if line.startswith("GPU") and not sent:        # GPU follows each finished chunk
            worker.send({"cmd": "cancel"})
            sent.append(1)

    res, err, _ = worker.job(TOKYO, tmp_path / "s.mp4", input_size=384, on_line=stop_after_first_chunk)
    assert err is None and res["partial"] == 1
    assert 0 < res["frames"] < 195
    assert frame_count(res["out"]) == res["frames"]


def test_export_png16_whole_clip_range(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "p.mp4", maxf=44, res=540, export="png16")
    assert err is None and res["export"].endswith("_png16")
    files = sorted(os.listdir(res["export"]))
    assert len(files) == 44
    frames = [cv2.imread(os.path.join(res["export"], f), cv2.IMREAD_UNCHANGED) for f in files]
    assert frames[0].dtype == np.uint16 and frames[0].shape == (304, 540)   # 540p long side
    stack = np.stack(frames)
    assert stack.min() < 2000 and stack.max() > 63000     # uses the full 16-bit range
    assert len(np.unique(stack[20])) > 4096               # real 16-bit detail, not 8-bit steps


def test_export_exr_is_raw_float(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "e.mp4", maxf=22, export="exr")
    assert err is None
    files = sorted(os.listdir(res["export"]))
    assert len(files) == 22 and files[0].endswith(".exr")
    d = cv2.imread(os.path.join(res["export"], files[0]), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.float32 and d.shape == (540, 960) and np.isfinite(d).all()


def test_export_prores_4444(worker, tmp_path):
    res, err, _ = worker.job(DAVIS, tmp_path / "r.mp4", maxf=44, res=540, export="prores")
    assert err is None and res["export"].endswith(".mov")
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", res["export"]],
                          capture_output=True, text=True).stderr
    assert "prores (4444)" in info and "yuv444p1" in info
    assert frame_count(res["export"]) == 44
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", res["export"],
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
                         capture_output=True).stdout
    first = np.frombuffer(raw, np.uint16)
    assert first.max() > 30000                             # no black lead-in frame


@pytest.mark.skipif(not os.path.exists(METRIC_WEIGHTS), reason="metric weights missing")
def test_metric_depth_in_meters(worker, tmp_path):
    res, err, lines = worker.job(DAVIS, tmp_path / "mt.mp4", maxf=44, model="metric", export="exr")
    assert err is None and any("model=metric" in l for l in lines)
    d = cv2.imread(os.path.join(res["export"], sorted(os.listdir(res["export"]))[10]),
                   cv2.IMREAD_UNCHANGED)
    near, far = np.median(d[200:300, 0:150]), np.median(d[0:150, 800:960])
    assert 0.3 < near < far                                # meters: near track closer than sky


def test_best_quality_fits_16gb(worker, tmp_path):
    """Fused attention lets Best (518) on 16:9 run at full quality on 16 GB."""
    res, err, lines = worker.job(DAVIS, tmp_path / "b.mp4", maxf=44, input_size=518)
    assert err is None and res["frames"] == 44
    assert not any(l.startswith("INFO auto-reduced") for l in lines)
    assert any("input_size=518" in l for l in lines)


def test_trim_command(worker):
    worker.send({"cmd": "trim"})
    line, _ = worker.expect("TRIMMED", timeout=10)
    before, after = map(float, line.split()[1:])
    assert after <= before + 0.01
