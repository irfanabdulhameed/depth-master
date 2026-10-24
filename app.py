"""Depth Studio — lightweight local UI with honest device analytics.

The server holds no torch. A single PERSISTENT warm worker (worker.py) loads the
model once and serves every Generate — no ~10 s cold start per run. The server
supervises it: RSS ceiling + sustained-critical-pressure watchdogs cancel a job
gracefully (partial video still saved) long before macOS would OOM-kill anything,
and a dead worker is respawned on the next run, so the UI itself never goes down.

Analytics use direct sysctl calls (no subprocess forks) — the 1 Hz panel costs
~nothing even while a job runs, and the 🟢/🟡/🔴 light stays honest (OS pressure
level, not stale swap).
"""
import atexit
import ctypes
import ctypes.util
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque

import gradio as gr
import pandas as pd
import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv/bin/python")
WORKER = os.path.join(ROOT, "worker.py")
OUTDIR = os.path.join(ROOT, "outputs")
SETTINGS = os.path.join(ROOT, ".settings.json")
WORKER_LOG = os.path.join(ROOT, "worker.log")
os.makedirs(OUTDIR, exist_ok=True)

TOTAL_GB = psutil.virtual_memory().total / 1e9
TIMEOUT_S = int(os.environ.get("TIMEOUT", "36000"))
CRIT_ABORT_S = float(os.environ.get("CRIT_SECS", "10"))     # sustained critical → cancel
RSS_CEIL_GB = float(os.environ.get("RSS_CEIL_GB", "6"))     # hard cap for worker + ffmpeg

SPEED_MAP = {"Fast": 294, "Balanced": 384, "Best": 518}
FPS_MAP = {"Original": 0, "24 fps": 24, "15 fps": 15}
RES_MAP = {"Match input (full res)": 0, "1080p": 1080, "720p": 720, "540p (fastest)": 540}

_HIST = deque(maxlen=90)
_T0 = time.time()
psutil.cpu_percent()  # prime


# ---------------------------------------------------------------- fork-free sysctls
_libc = None
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
except Exception:
    pass


def _sysctl_u64(name):
    if _libc is None:
        return None
    val = ctypes.c_uint64(0)
    sz = ctypes.c_size_t(ctypes.sizeof(val))
    try:
        if _libc.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(sz), None, 0) == 0:
            return val.value
    except Exception:
        pass
    return None


_PRESS = {"v": 1, "ts": 0.0}


def _pressure():
    """macOS memory-pressure level: 1 normal, 2 warning, 4 critical. Cached 1 s,
    read via sysctlbyname — no subprocess fork (the old version forked ~5×/s)."""
    now = time.time()
    if now - _PRESS["ts"] < 1.0:
        return _PRESS["v"]
    v = _sysctl_u64("kern.memorystatus_vm_pressure_level")
    if v is None:
        try:
            v = int(subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip())
        except Exception:
            v = 1
    _PRESS["v"], _PRESS["ts"] = int(v), now
    return _PRESS["v"]


def _compressor_gb():
    v = _sysctl_u64("vm.compressor_bytes_used")
    return v / 1e9 if v is not None else 0.0


# ---------------------------------------------------------------- warm worker manager
class WorkerMgr:
    """Owns the persistent worker.py subprocess and its line protocol."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.ready = threading.Event()
        self.jobq = None          # queue for the active job's protocol lines
        self.ctlq = queue.Queue() # TRIMMED / PONG outside jobs
        self.gpu = {"gb": 0.0, "ts": 0.0}
        self.state = "off"        # off | loading | ready | busy

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self):
        with self.lock:
            if self.alive():
                return
            self.ready.clear()
            self.state = "loading"
            env = {**os.environ,
                   "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                   "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7",
                   "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5"}
            log = open(WORKER_LOG, "ab", buffering=0)
            self.proc = subprocess.Popen(
                [VENV_PY, "-u", WORKER], env=env, cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, bufsize=1)
            threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("READY"):
                self.ready.set()
                if self.state == "loading":
                    self.state = "ready"
            elif line.startswith("GPU"):
                try:
                    self.gpu = {"gb": float(line.split()[1]), "ts": time.time()}
                except Exception:
                    pass
            if self.jobq is not None:
                self.jobq.put(line)
            elif line.startswith(("TRIMMED", "PONG")):
                self.ctlq.put(line)
        # EOF — worker died or quit
        if self.jobq is not None:
            self.jobq.put("__DEAD__")
        self.ready.clear()
        if self.state != "off":
            self.state = "off"
        self.gpu = {"gb": 0.0, "ts": 0.0}

    def send(self, obj):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
            return True
        except Exception:
            return False

    def submit(self, job):
        self.ensure()
        q = queue.Queue()
        self.jobq = q
        self.state = "busy"
        if not self.send({"cmd": "job", **job}):
            q.put("__DEAD__")
        return q

    def finish_job(self):
        self.jobq = None
        if self.alive():
            self.state = "ready"

    def cancel(self):
        """Graceful stop; escalate to SIGTERM/SIGKILL if the worker wedges."""
        if not self.alive():
            return
        self.send({"cmd": "cancel"})
        proc = self.proc

        def escalate():
            time.sleep(8)
            if proc.poll() is None and self.jobq is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                time.sleep(4)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        threading.Thread(target=escalate, daemon=True).start()

    def trim(self, timeout=6):
        """Ask the warm worker to drop GPU/CPU caches. Returns (before, after) GB."""
        if not self.alive() or self.jobq is not None:
            return None
        while not self.ctlq.empty():
            self.ctlq.get_nowait()
        if not self.send({"cmd": "trim"}):
            return None
        try:
            line = self.ctlq.get(timeout=timeout)
            _, b, a = line.split()
            return float(b), float(a)
        except Exception:
            return None

    def rss_gb(self):
        if not self.alive():
            return 0.0
        try:
            p = psutil.Process(self.proc.pid)
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                rss += c.memory_info().rss
            return rss / 1e9
        except Exception:
            return 0.0

    def shutdown(self):
        if self.alive():
            self.send({"cmd": "quit"})
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.state = "off"


MGR = WorkerMgr()
atexit.register(MGR.shutdown)

# Pre-warm on launch so the FIRST Generate is already fast.
threading.Thread(target=MGR.ensure, daemon=True).start()


# ---------------------------------------------------------------- analytics
_IDLE_RELEASE = {"ts": 0.0}


def sample_stats():
    vm = psutil.virtual_memory()
    swap_gb = psutil.swap_memory().used / 1e9
    comp_gb = _compressor_gb()
    cpu = psutil.cpu_percent()
    ram_gb = vm.used / 1e9
    p = _pressure()
    _HIST.append((round(time.time() - _T0, 1), round(ram_gb, 2), round(swap_gb, 2)))
    if p >= 4:
        status = "🔴 critical — real memory pressure"
    elif p == 2:
        status = "🟡 some pressure (stale swap; a reboot clears it)"
    else:
        status = "🟢 headroom"
    # If the system is critical while we're idle, give the model's RAM back.
    if p >= 4 and MGR.state == "ready" and time.time() - _IDLE_RELEASE["ts"] > 30:
        _IDLE_RELEASE["ts"] = time.time()
        threading.Thread(target=MGR.shutdown, daemon=True).start()
        status += " · released warm model"
    gpu_txt = f"{MGR.gpu['gb']:.1f} GB · {MGR.state}" if MGR.state != "off" else "0.0 GB · off"
    return (f"{cpu:.0f}%", f"{ram_gb:.1f} / {TOTAL_GB:.0f} GB", f"{comp_gb:.1f} GB",
            f"{swap_gb:.1f} GB", gpu_txt, status)


def sample_plot():
    rows = []
    for tt, r, s in _HIST:
        rows.append({"t": tt, "GB": r, "metric": "RAM"})
        rows.append({"t": tt, "GB": s, "metric": "swap"})
    return pd.DataFrame(rows) if rows else pd.DataFrame({"t": [], "GB": [], "metric": []})


def _fmt_eta(s):
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------- settings memory
def load_settings():
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


# ---------------------------------------------------------------- generate
def generate(video, grayscale, speed, fps_choice, resolution, progress=gr.Progress()):
    if not video:
        raise gr.Error("Drop a video in first.")
    save_settings({"grayscale": bool(grayscale), "speed": speed,
                   "fps": fps_choice, "resolution": resolution})
    out = os.path.join(OUTDIR, f"depth_{int(time.time())}.mp4")
    job = {"inp": video, "out": out,
           "grayscale": bool(grayscale),
           "res": RES_MAP.get(resolution, 0),
           "input_size": SPEED_MAP.get(speed, 384),
           "target_fps": FPS_MAP.get(fps_choice, 0),
           "maxf": 0}
    progress(0.02, desc="starting")
    q = MGR.submit(job)

    start = time.time()
    infer_start = None
    last_done, last_t, rate_ema = 0, None, None
    total = done = 0
    err = None
    info = {}
    note = ""          # e.g. auto-reduced-quality message to show the user
    crit_since = None
    peak_rss = 0.0
    finished = None  # DONE dict when the worker completes/partials
    SIZE_NAME = {294: "Fast", 350: "Fast+", 384: "Balanced", 406: "Balanced+",
                 462: "Best-", 518: "Best"}

    while True:
        try:
            line = q.get(timeout=0.3)
        except queue.Empty:
            line = None
        now = time.time()

        if line == "__DEAD__":
            MGR.finish_job()
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                progress(1.0, desc="worker died — partial saved")
                return out, "⚠️ engine crashed mid-run — partial saved; it will reload on the next Generate"
            raise gr.Error("The inference engine died (likely memory). It reloads automatically — "
                           "try again with Fast speed or a lower resolution.")
        if line:
            if line.startswith("PROGRESS"):
                _, d, t = line.split()
                d, t = int(d), int(t)
                total = t or total
                done = d
                if infer_start is None:
                    infer_start = now
                if d > last_done:
                    if last_t is not None:
                        inst = (d - last_done) / max(0.05, now - last_t)
                        rate_ema = inst if rate_ema is None else 0.5 * inst + 0.5 * rate_ema
                    elif d > 0:
                        rate_ema = d / max(0.05, now - infer_start)
                    last_done, last_t = d, now
                eta = f" · ~{_fmt_eta((total - d) / rate_ema)} left" if (rate_ema and total) else ""
                fps_txt = f" · {rate_ema:.1f} fps" if rate_ema else ""
                if total:
                    progress(min(0.99, d / total), desc=f"estimating depth · {d}/{total}{fps_txt}{eta}")
                else:
                    progress(0.5, desc=f"estimating depth · {d} frames{fps_txt}{eta}")
            elif line.startswith("STAGE loading") or line.startswith("STAGE warmup"):
                progress(0.04, desc="loading model (one-time)")
            elif line.startswith("STAGE inferring"):
                progress(0.06, desc="estimating depth")
            elif line.startswith("INFO auto-reduced"):
                m = re.search(r"(\d+)->(\d+)", line)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    note = (f"quality auto-reduced {SIZE_NAME.get(a, a)}→{SIZE_NAME.get(b, b)} "
                            f"so it fits in memory")
                    progress(0.06, desc="↓ " + note)
            elif line.startswith("INFO quality") and "won't fit" in line:
                progress(0.05, desc="checking memory fit…")
            elif line.startswith("INFO src="):
                info = dict(kv.split("=", 1) for kv in line[5:].split() if "=" in kv)
            elif line.startswith("ERROR"):
                err = line[6:].strip()
                break
            elif line.startswith("DONE"):
                finished = dict(kv.split("=", 1) for kv in line[5:].split() if "=" in kv)
                break

        # ---- watchdogs (no forks: cached sysctl + psutil) ----
        rss = MGR.rss_gb()
        peak_rss = max(peak_rss, rss)
        if rss > RSS_CEIL_GB:
            MGR.cancel()
            err = f"memory ceiling hit ({rss:.1f} GB > {RSS_CEIL_GB:.0f} GB) — run stopped safely"
        if _pressure() >= 4:
            crit_since = crit_since or now
            if now - crit_since >= CRIT_ABORT_S:
                MGR.cancel()
                err = "sustained critical memory pressure — run stopped safely"
        else:
            crit_since = None
        if now - start > TIMEOUT_S:
            MGR.cancel()
            err = "timed out"
        if err and finished is None and line is None:
            # give the worker a moment to finalize the partial, then bail out
            try:
                while True:
                    l2 = q.get(timeout=10)
                    if l2.startswith("DONE"):
                        finished = dict(kv.split("=", 1) for kv in l2[5:].split() if "=" in kv)
                        break
                    if l2.startswith("ERROR") or l2 == "__DEAD__":
                        break
            except queue.Empty:
                pass
            break

    MGR.finish_job()
    stopped = _STOP_FLAG.pop("stopped", False)

    if finished:
        secs = float(finished.get("secs", 0))
        frames = int(finished.get("frames", 0))
        wpeak = float(finished.get("peak_gb", 0)) or peak_rss
        partial = finished.get("partial") == "1"
        desc = "stopped — partial saved" if (partial or stopped) else "done"
        if err and partial:
            desc = err + " — partial saved"
        progress(1.0, desc=desc)
        stat = (f"{'⏹ partial' if partial or stopped else '✅ done'} · {frames} frames in {secs:.0f}s "
                f"({frames / max(secs, 0.1):.1f} fps) · peak {wpeak:.1f} GB · {info.get('encoder', '')}")
        if note:
            stat += f" · ⓘ {note}"
        return finished.get("out", out), stat
    if stopped:
        return None, "⏹ stopped before any frames were ready"
    raise gr.Error(f"Couldn't process this clip — {err or 'unknown error'}. "
                   "Try Fast speed or a lower resolution.")


_STOP_FLAG = {}


def stop_run():
    """Kill switch: graceful cancel (partial video finalizes) with SIGTERM/SIGKILL
    escalation if the worker doesn't respond."""
    if MGR.jobq is not None:
        _STOP_FLAG["stopped"] = True
        MGR.cancel()


def clear_memory():
    """Free memory NOW: drop the worker's GPU/CPU caches in place (fast, stays
    warm), reap stray processes, then purge inactive memory (no-sudo first, else
    the native admin prompt — still opt-in). Reports exactly what it freed."""
    def snap():
        vm = psutil.virtual_memory()
        return vm.used / 1e9, psutil.swap_memory().used / 1e9, _compressor_gb()

    b_ram, b_swap, b_comp = snap()
    _HIST.clear()
    bits = []

    if MGR.jobq is not None:
        bits.append("run in progress — caches left alone")
    else:
        t = MGR.trim()
        if t:
            bits.append(f"model cache {t[0]:.1f}→{t[1]:.1f} GB (kept warm)")

    cur_pid = MGR.proc.pid if MGR.alive() else -1
    killed = 0
    try:
        pids = subprocess.check_output(["pgrep", "-f", "worker.py|infer.py"], text=True).split()
        for pid in pids:
            if int(pid) not in (cur_pid, os.getpid()):
                try:
                    os.kill(int(pid), 9)
                    killed += 1
                except Exception:
                    pass
    except Exception:
        pass
    if killed:
        bits.append(f"reaped {killed} stray process(es)")

    purge = None
    try:
        r = subprocess.run(["/usr/sbin/purge"], capture_output=True, text=True, timeout=25)
        if r.returncode == 0:
            purge = "purged inactive memory"
        else:
            r = subprocess.run(
                ["osascript", "-e", 'do shell script "/usr/sbin/purge" with administrator privileges'],
                capture_output=True, text=True, timeout=90)
            purge = "purged inactive memory" if r.returncode == 0 else \
                "purge cancelled — run `sudo purge` in Terminal to clear swap"
    except subprocess.TimeoutExpired:
        purge = "purge timed out"
    except Exception:
        purge = "purge unavailable — run `sudo purge` in Terminal"
    bits.append(purge)

    a_ram, a_swap, a_comp = snap()
    bits.append(f"RAM {b_ram:.1f}→{a_ram:.1f} GB · swap {b_swap:.1f}→{a_swap:.1f} GB · "
                f"compressor {b_comp:.1f}→{a_comp:.1f} GB")
    return " · ".join(bits)


# ---------------------------------------------------------------- UI
_S = load_settings()

with gr.Blocks(title="Depth Studio") as demo:
    gr.Markdown(
        "## Depth Studio\n"
        "Drop a video, get its depth map — runs locally in memory-bounded chunks, so it "
        "**finishes any length**. The model stays warm between runs, so repeat runs start instantly."
    )
    with gr.Row(equal_height=True):
        with gr.Column():
            inp = gr.Video(label="Input video", sources=["upload"])
            grayscale = gr.Checkbox(label="grayscale", value=bool(_S.get("grayscale", False)),
                                    info="desaturate the depth — black → white instead of color")
            with gr.Row():
                speed = gr.Dropdown(choices=list(SPEED_MAP.keys()),
                                    value=_S.get("speed", "Balanced"),
                                    label="speed / quality", info="Fast is ~3× quicker")
                fps_choice = gr.Dropdown(choices=list(FPS_MAP.keys()),
                                         value=_S.get("fps", "Original"),
                                         label="process fps", info="lower = fewer frames = faster")
            resolution = gr.Dropdown(choices=list(RES_MAP.keys()),
                                     value=_S.get("resolution", "Match input (full res)"),
                                     label="output resolution")
            with gr.Row():
                btn = gr.Button("Generate depth", variant="primary", scale=3)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
            run_stat = gr.Textbox(show_label=False, interactive=False, container=False,
                                  placeholder="last run stats appear here")
        with gr.Column():
            out = gr.Video(label="Depth map", autoplay=True, loop=True, interactive=False)

    with gr.Accordion("Device usage", open=True):
        with gr.Row():
            cpu_v = gr.Textbox(label="CPU", interactive=False)
            ram_v = gr.Textbox(label="RAM", interactive=False)
            comp_v = gr.Textbox(label="Compressed", interactive=False)
            swap_v = gr.Textbox(label="Swap (cumulative)", interactive=False)
            gpu_v = gr.Textbox(label="GPU (model)", interactive=False)
            status_v = gr.Textbox(label="Headroom", interactive=False)
        plot = gr.LinePlot(x="t", y="GB", color="metric", y_lim=[0, round(TOTAL_GB) + 2],
                           title="RAM vs swap (GB) over time", height=200)
        with gr.Row():
            clear_btn = gr.Button("Free memory", variant="secondary", scale=1)
            clear_status = gr.Textbox(show_label=False, interactive=False, scale=4, container=False,
                                      placeholder="clears model caches, strays & macOS inactive memory — may ask for your password")

    timer = gr.Timer(1.0)
    timer.tick(sample_stats, outputs=[cpu_v, ram_v, comp_v, swap_v, gpu_v, status_v], queue=False)
    plot_timer = gr.Timer(2.0)
    plot_timer.tick(sample_plot, outputs=[plot], queue=False)
    btn.click(generate, inputs=[inp, grayscale, speed, fps_choice, resolution],
              outputs=[out, run_stat])
    # queue=False so Stop/Free-memory run immediately, not behind the running Generate.
    stop_btn.click(stop_run, inputs=None, outputs=None, queue=False)
    clear_btn.click(clear_memory, inputs=None, outputs=clear_status, queue=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    )
