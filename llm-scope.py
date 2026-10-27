"""llm-scope - a continuous LLM-conversation debugger for the maya engine on the GB10.

One process, two jobs:
  1. collector thread: tails the engine/server log (live timestamps - the log
     itself is untimestamped), tails the gpu_monitor log (it has timestamps),
     and samples /proc every 2 s (PSI-io, diskstats MB/s, Dirty, in-flight).
     Everything lands as JSONL in data/events.jsonl.
  2. HTTP server: GET /            -> the dashboard (index.html)
                  GET /api/events?since=<epoch>  -> new events as JSON
                  GET /api/state                 -> a one-screen summary

Run:  python3 llm-scope.py [--port 8090] [--log <engine log>] [--monitor <gpu_monitor_log>]
Read-only against the engine; safe to run beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parser as P  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "events.jsonl"
HOME = Path.home()
DEFAULT_LOG = Path(os.environ["LLM_SCOPE_LOG"]) if os.environ.get("LLM_SCOPE_LOG") else None
DEFAULT_MONITOR = HOME / "gpu_monitor_log.txt"

_lock = threading.Lock()
EVENTS: list[dict] = []          # newest last; kept capped in memory
MAX_EVENTS = 200_000

MON_FIELDS = re.compile(r"U:(\d+)% .*?T:([\d.]+).*?P:([\d.]+)W .*?M:(\d+)MiB .*?"
                        r"F:([\d.]+)GiB .*?A:([\d.]+)GiB .*?N:(\d+)")


def record(ev: dict):
    ev["ts"] = round(time.time(), 3)
    with _lock:
        EVENTS.append(ev)
        if len(EVENTS) > MAX_EVENTS:
            del EVENTS[: len(EVENTS) - MAX_EVENTS]
    try:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA, "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[llm-scope] write failed: {e}", file=sys.stderr, flush=True)


def tail(path: Path, on_line, poll=0.5):
    """Tail a file from EOF; call on_line(str) per new line; survives truncation."""
    while not path.exists():
        time.sleep(1)
    with open(path, "r", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                try:
                    on_line(line)
                except Exception as e:          # never let one line kill the tail
                    print(f"[llm-scope] parse error: {e}", file=sys.stderr, flush=True)
            else:
                if path.stat().st_size < f.tell():       # truncated/rotated: reopen
                    f.close()
                    time.sleep(0.5)
                    return tail(path, on_line, poll)
                time.sleep(poll)


def engine_tail():
    if DEFAULT_LOG is None:
        print("[llm-scope] no engine log: pass --log /path/to/engine.log or set LLM_SCOPE_LOG",
              file=sys.stderr, flush=True)
        return
    def on_line(line):
        ev = P.parse_line(line)
        if ev is not None:
            record(ev)
    tail(DEFAULT_LOG, on_line)


def monitor_tail():
    def on_line(line):
        if "GPU: NVIDIA GB10" not in line:
            return
        m = MON_FIELDS.search(line)
        if m:
            record({"type": "gpu", "u": int(m.group(1)), "t": float(m.group(2)),
                    "p": float(m.group(3)), "m": int(m.group(4)), "f": float(m.group(5)),
                    "a": float(m.group(6)), "n": int(m.group(7))})
    tail(DEFAULT_MONITOR, on_line)


_DISK = re.compile(r"nvme0n1 (\d+) (\d+) (\d+) \d+ (\d+) \d+ (\d+) \d+ (\d+) \d+ (\d+)")


def sampler(poll=2.0):
    last = {"sectors_r": None, "sectors_w": None, "t": None}
    while True:
        try:
            psi = {}
            for line in open("/proc/pressure/io"):
                parts = line.split()
                psi[parts[0]] = float(dict(x.split("=") for x in parts[1:])["avg10"])
            d = None
            for line in open("/proc/diskstats"):
                m = _DISK.match(line.split(None, 3)[3] if False else " ".join(line.split()[2:]))
                if m:
                    sr, sw, inf = int(m.group(3)), int(m.group(5)), int(m.group(6))
                    now = time.time()
                    r_mb = w_mb = 0.0
                    if last["sectors_r"] is not None and now > last["t"]:
                        dt = now - last["t"]
                        r_mb = max(0.0, (sr - last["sectors_r"]) * 512 / dt / 1e6)
                        w_mb = max(0.0, (sw - last["sectors_w"]) * 512 / dt / 1e6)
                    last.update(sectors_r=sr, sectors_w=sw, t=now)
                    d = (r_mb, w_mb, inf)
                    break
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                if k in ("Dirty", "Writeback"):
                    mem[k] = int(v.strip().split()[0]) / 1024.0
            ev = {"type": "sys", "io_some": psi.get("some"), "io_full": psi.get("full"),
                  "disk_r_mbs": round(d[0], 1) if d else None,
                  "disk_w_mbs": round(d[1], 1) if d else None,
                  "in_flight": d[2] if d else None,
                  "dirty_mib": round(mem.get("Dirty", 0.0), 1),
                  "writeback_mib": round(mem.get("Writeback", 0.0), 1)}
            record(ev)
        except Exception as e:
            print(f"[llm-scope] sampler error: {e}", file=sys.stderr, flush=True)
        time.sleep(poll)


HTML = (ROOT / "index.html").read_text(encoding="utf-8") if (ROOT / "index.html").exists() else "<h1>index.html missing</h1>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/events"):
            since = 0.0
            if "since=" in self.path:
                try:
                    since = float(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    pass
            with _lock:
                out = [e for e in EVENTS if e["ts"] > since]
            body = json.dumps({"now": time.time(), "events": out[-5000:]}).encode()
            self._send(200, body, "application/json")
            return
        if self.path.startswith("/api/state"):
            with _lock:
                reqs = [e for e in EVENTS if e["type"] == "req"]
                refused = [e for e in EVENTS if e["type"] == "refused"]
                done = [e for e in EVENTS if e["type"] == "done"]
                sysv = [e for e in EVENTS if e["type"] == "sys"]
            s = {"requests": len(reqs), "refusals": len(refused), "completed": len(done),
                 "last_prompt": reqs[-1]["prompt_tokens"] if reqs else None,
                 "io_full": sysv[-1]["io_full"] if sysv else None,
                 "engine_log": str(DEFAULT_LOG)}
            self._send(200, json.dumps(s).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--log", default=None, help="the engine/server log to tail")
    a = ap.parse_args()
    global DEFAULT_LOG
    if a.log:
        DEFAULT_LOG = Path(a.log)
    if DATA.exists():                     # reload the record so a restart keeps history
        try:
            lines = DATA.read_text(encoding="utf-8", errors="replace").splitlines()
            loaded = [json.loads(l) for l in lines[-MAX_EVENTS:] if l.strip()]
            with _lock:
                EVENTS.extend(loaded)
            print(f"[llm-scope] loaded {len(loaded)} events from {DATA}", flush=True)
        except Exception as e:
            print(f"[llm-scope] history load failed: {e}", file=sys.stderr, flush=True)
    t1 = threading.Thread(target=engine_tail, daemon=True)
    t2 = threading.Thread(target=monitor_tail, daemon=True)
    t3 = threading.Thread(target=sampler, daemon=True)
    for t in (t1, t2, t3):
        t.start()
    print(f"[llm-scope] collecting -> {DATA}", flush=True)
    print(f"[llm-scope] dashboard on http://0.0.0.0:{a.port}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()