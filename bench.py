"""Headless benchmark for the depth pipeline — before/after evidence.

Modes:
  legacy: one-shot infer.py subprocess (the old architecture)
  worker: persistent worker.py — runs the SAME job twice (cold + warm)

Reports per run: wall-clock, time-to-first-progress (model load + MPS warmup),
steady frames/sec, peak RSS of the child, output size.

Usage:
  .venv/bin/python bench.py legacy --clip vda/assets/example_videos/davis_rollercoaster.mp4
  .venv/bin/python bench.py worker --clip ... --input-size 384 --res 0 --maxf 0
"""
import argparse, json, os, subprocess, sys, time

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv/bin/python")


def track(proc, on_line):
    """Poll child RSS while streaming its stdout lines. Returns peak RSS bytes."""
    peak = 0
    ps = psutil.Process(proc.pid)
    import threading
    stop = threading.Event()

    def poll():
        nonlocal peak
        while not stop.is_set():
            try:
                rss = ps.memory_info().rss
                for c in ps.children(recursive=True):  # ffmpeg child counts too
                    rss += c.memory_info().rss
                peak = max(peak, rss)
            except Exception:
                return
            time.sleep(0.2)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    for line in proc.stdout:
        on_line(line.rstrip())
    stop.set()
    t.join(timeout=1)
    return peak


class RunStats:
    def __init__(self):
        self.t0 = time.time()
        self.first_progress = None
        self.last_progress = None
        self.done = 0
        self.total = 0
        self.out = None

    def feed(self, line):
        now = time.time()
        if line.startswith("PROGRESS"):
            _, d, t = line.split()[:3]
            self.done, self.total = int(d), int(t)
            if self.first_progress is None and self.done > 0:
                self.first_progress = now
            self.last_progress = now
        elif line.startswith(("STAGE done", "DONE")):
            self.out = line

    def report(self, label, peak):
        wall = time.time() - self.t0
        ttfp = (self.first_progress - self.t0) if self.first_progress else None
        steady = None
        if self.first_progress and self.last_progress and self.last_progress > self.first_progress:
            steady = self.done / (self.last_progress - self.t0)
        print(f"\n=== {label} ===")
        print(f"wall            {wall:8.1f} s")
        print(f"first progress  {ttfp:8.1f} s   (load + warmup + chunk1 lead-in)" if ttfp else "first progress      n/a")
        print(f"frames          {self.done}/{self.total}")
        print(f"avg fps         {self.done / wall:8.2f}")
        print(f"peak RSS        {peak / 1e9:8.2f} GB")
        return {"label": label, "wall": wall, "ttfp": ttfp, "frames": self.done,
                "fps": self.done / wall if wall else 0, "peak_rss_gb": peak / 1e9}


def run_legacy(a):
    out = os.path.join(ROOT, "outputs", f"bench_legacy_{int(time.time())}.mp4")
    env = {**os.environ, "INP": a.clip, "OUT": out, "GRAYSCALE": "0",
           "RES": str(a.res), "INPUT_SIZE": str(a.input_size),
           "TARGET_FPS": str(a.fps), "MAXF": str(a.maxf),
           "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    st = RunStats()
    proc = subprocess.Popen([PY, "-u", os.path.join(ROOT, "infer.py")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    peak = track(proc, lambda l: (st.feed(l), a.verbose and print(f"{time.time()-st.t0:6.2f} {l}")))
    proc.wait()
    r = st.report("legacy one-shot", peak)
    print(f"output          {out}  ({os.path.getsize(out)/1e6:.1f} MB)" if os.path.exists(out) else "NO OUTPUT")
    return [r]


def run_worker(a):
    proc = subprocess.Popen([PY, "-u", os.path.join(ROOT, "worker.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    results = []
    try:
        for i, label in enumerate(["worker cold (run 1)", "worker warm (run 2)"]):
            out = os.path.join(ROOT, "outputs", f"bench_worker_{int(time.time())}_{i}.mp4")
            job = {"cmd": "job", "inp": a.clip, "out": out, "grayscale": False,
                   "res": a.res, "input_size": a.input_size, "target_fps": a.fps, "maxf": a.maxf}
            st = RunStats()
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
            peak = 0
            ps = psutil.Process(proc.pid)
            while True:
                line = proc.stdout.readline()
                if not line:
                    print("worker died"); sys.exit(1)
                line = line.rstrip()
                if a.verbose:
                    print(f"{time.time()-st.t0:6.2f} {line}")
                st.feed(line)
                try:
                    rss = ps.memory_info().rss
                    for c in ps.children(recursive=True):
                        rss += c.memory_info().rss
                    peak = max(peak, rss)
                except Exception:
                    pass
                if line.startswith(("DONE", "ERROR", "CANCELLED")):
                    break
            results.append(st.report(label, peak))
            print(f"output          {out}  ({os.path.getsize(out)/1e6:.1f} MB)" if os.path.exists(out) else "NO OUTPUT")
    finally:
        try:
            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["legacy", "worker"])
    p.add_argument("--clip", default=os.path.join(ROOT, "vda/assets/example_videos/davis_rollercoaster.mp4"))
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--res", type=int, default=0)
    p.add_argument("--fps", type=float, default=0)
    p.add_argument("--maxf", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    rs = run_legacy(a) if a.mode == "legacy" else run_worker(a)
    print("\nJSON " + json.dumps(rs))
