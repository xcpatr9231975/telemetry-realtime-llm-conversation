"""llm-scope/parser.py - one maya engine-log line -> one event dict (or None).

Every pattern is grounded in the real lines the engine+server emit (see the
256k run log).  Lines that match nothing are kept as type "log" with the raw
line truncated, so nothing is silently dropped.
"""
from __future__ import annotations

import re

def _n(s):
    """'41,911' -> 41911; '12.3' -> 12.3; None on failure."""
    try:
        return float(s.replace(",", "")) if s else None
    except ValueError:
        return None

# [maya] req openai: msgs=44 tools=80 max_tokens_raw=32000/None max_new=32000 thinking=True stream=True prompt_tokens=41912 last='user':'...'
RE_REQ = re.compile(
    r"\[maya\] req (\w+): msgs=(\d+) tools=(\d+) max_tokens_raw=(\S+) max_new=(\S+) "
    r"thinking=(\w+) stream=(\w+) prompt_tokens=([\d,]+) last='([^']*)':'(.*)'")
# [maya] reading the prompt: 41,911 of 41,912 tokens, 3 s so far   |  ...: 50,511 tokens, 10 s so far
RE_READ = re.compile(r"\[maya\] reading the prompt: ([\d,]+)(?: of ([\d,]+))? tokens, ([\d.]+) s so far")
# glm prefill: CUDA0 chunk 1/1, 422 tokens, wall 3433 ms | the copy lane 3348 ms: copies 2733 (5.2 GB/s), ...
RE_CHUNK = re.compile(
    r"glm prefill: (?:CUDA0|HIP) chunk (\d+)/(\d+), ([\d,]+) tokens, wall (\d+) ms \| the copy lane (\d+) ms: "
    r"copies (\d+) \(([\d.]+) GB/s\).*pcie ([\d.]+) GB \(ram ([\d.]+), disk ([\d.]+)\)(?: \+ ([\d.]+) GB vram)?"
    r".*experts: staged (\d+), resident (\d+)")
# glm prefill: 422 tokens in 3434.5 ms (8.14 ms/token, 123 tok/s), 1 chunk of 448 tokens | CUDA0 3433 ms
RE_PREFILL_SUM = re.compile(
    r"glm prefill: ([\d,]+) tokens in ([\d.]+) ms \(([\d.]+) ms/token, ([\d.]+) tok/s\), (\d+) chunk(?:s)? of ([\d,]+) tokens")
# glm stat: decode 92.21 ms/tok (10.84 tok/s) | vram hit 94.35% | ram fetches 0.18/tok | disk 20.60 reads/tok in 15.70 waits/tok, 32.02 ms/tok | ...
RE_STAT = re.compile(
    r"glm stat: decode ([\d.]+) ms/tok \(([\d.]+) tok/s\) \| vram hit ([\d.]+)% \| ram fetches ([\d.]+)/tok \| "
    r"disk ([\d.]+) reads/tok in ([\d.]+) waits/tok, ([\d.]+) ms/tok \| promotions ([\d.]+)/tok")
# glm spec: 215 of 239 drafts accepted (90.0%)
RE_SPEC = re.compile(r"glm spec: (\d+) of (\d+) drafts accepted \(([\d.]+)%\)")
# glm kv: KV streaming: 99.47% of 16356615 block reads hit VRAM, 341.1 MiB read from RAM
RE_KV = re.compile(r"glm kv: KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
# glm slots: saved a 20219-token conversation (0.41 GB in 0.2 s); 1 kept, 0.41 GB
RE_SLOT = re.compile(r"glm slots: saved a ([\d,]+)-token conversation \(([\d.]+) GB in ([\d.]+) s\); (\d+) kept")
# glm checkpoints: 3352 of the prompt's 50511 tokens from a checkpoint
RE_CKPT = re.compile(r"glm checkpoints: ([\d,]+) of the prompt's ([\d,]+) tokens from a checkpoint")
# [maya] thinking: 4452 of max 32000 tokens, 10.9 tok/s, 442 s   | answering: ... | writing a tool call: bash: ... | tool call complete: ...
RE_PHASE = re.compile(r"\[maya\] (thinking|answering|writing a tool call: ([^:]+)|tool call complete): "
                      r"([\d,]+) of max ([\d,]+) tokens, ([\d.]+) tok/s, ([\d,]+) s")
# [maya] done: 4930 tokens in 487 s (10.9 tok/s) (stop, cancel=False), expert cache 95.5% hit
RE_DONE = re.compile(r"\[maya\] done: ([\d,]+) tokens in ([\d,]+) s \(([\d.]+) tok/s\) \(([^,]+), cancel=(\w+)\)"
                     r"(?:, expert cache ([\d.]+)% hit)?")
# [maya] tool bash: ok, 12,345 characters in 1.2 s (truncated for the model)
RE_TOOL = re.compile(r"\[maya\] tool ([^:]+): (ok|error), ([\d,]+) characters in ([\d.]+) s(.*)")
# [maya] prompt cap: refused a 82,770-token prompt (cap 65,536; msgs=30 tools=80 last='tool':'...')
RE_REFUSED = re.compile(r"\[maya\] prompt cap: refused a ([\d,]+)-token prompt \(cap ([\d,]+); msgs=(\d+) tools=(\d+) "
                        r"last='([^']*)':'(.*)'\)")
# [maya] WARNING: 2 request(s) have waited for the engine for up to 495 s ...
RE_WARN = re.compile(r"\[maya\] WARNING: (\d+) request\(s\) have waited for the engine for up to ([\d,]+) s")
# [maya] raw: '...'  (the full answer; keep its length only)
RE_RAW = re.compile(r"\[maya\] raw: '(.*)'\s*$", re.S)


def parse_line(line: str) -> dict | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None
    ev = {}
    m = RE_REQ.search(line)
    if m:
        return {"type": "req", "api": m.group(1), "msgs": int(m.group(2)), "tools": int(m.group(3)),
                "max_new": _n(m.group(5)), "thinking": m.group(6) == "True", "stream": m.group(7) == "True",
                "prompt_tokens": _n(m.group(8)), "last_role": m.group(9),
                "last_preview": m.group(10)[:70]}
    m = RE_READ.search(line)
    if m:
        return {"type": "read", "done": _n(m.group(1)),
                "total": _n(m.group(2)) or _n(m.group(1)), "elapsed": _n(m.group(3))}
    m = RE_CHUNK.search(line)
    if m:
        return {"type": "prefill_chunk", "chunk": int(m.group(1)), "chunks": int(m.group(2)),
                "tokens": _n(m.group(3)), "wall_ms": _n(m.group(4)), "copy_ms": _n(m.group(5)),
                "copies": int(m.group(6)), "copy_gb_s": _n(m.group(7)), "pcie_gb": _n(m.group(8)),
                "pcie_ram": _n(m.group(9)), "pcie_disk": _n(m.group(10)),
                "vram_gb": _n(m.group(11)), "staged": int(m.group(12)), "resident": int(m.group(13))}
    m = RE_PREFILL_SUM.search(line)
    if m:
        return {"type": "prefill_sum", "tokens": _n(m.group(1)), "ms": _n(m.group(2)),
                "ms_tok": _n(m.group(3)), "tok_s": _n(m.group(4)), "chunks": int(m.group(5)),
                "chunk_tokens": _n(m.group(6))}
    m = RE_STAT.search(line)
    if m:
        return {"type": "stat", "ms_tok": _n(m.group(1)), "tok_s": _n(m.group(2)),
                "vram_hit": _n(m.group(3)), "ram_fetch": _n(m.group(4)),
                "disk_reads_tok": _n(m.group(5)), "disk_waits_tok": _n(m.group(6)),
                "disk_ms_tok": _n(m.group(7)), "promotions_tok": _n(m.group(8))}
    m = RE_SPEC.search(line)
    if m:
        return {"type": "spec", "accepted": int(m.group(1)), "drafts": int(m.group(2)), "pct": _n(m.group(3))}
    m = RE_KV.search(line)
    if m:
        return {"type": "kv", "hit_pct": _n(m.group(1)), "blocks": int(m.group(2)), "ram_mib": _n(m.group(3))}
    m = RE_SLOT.search(line)
    if m:
        return {"type": "slot_save", "tokens": _n(m.group(1)), "gb": _n(m.group(2)),
                "s": _n(m.group(3)), "kept": int(m.group(4))}
    m = RE_CKPT.search(line)
    if m:
        return {"type": "ckpt", "reused": _n(m.group(1)), "prompt_tokens": _n(m.group(2))}
    m = RE_PHASE.search(line)
    if m:
        return {"type": "phase", "phase": (m.group(2) or m.group(1)).strip(),
                "tokens": _n(m.group(3)), "max": _n(m.group(4)),
                "tok_s": _n(m.group(5)), "elapsed": _n(m.group(6))}
    m = RE_DONE.search(line)
    if m:
        return {"type": "done", "tokens": _n(m.group(1)), "seconds": _n(m.group(2)),
                "tok_s": _n(m.group(3)), "finish": m.group(4), "cancel": m.group(5) == "True",
                "cache_pct": _n(m.group(6))}
    m = RE_TOOL.search(line)
    if m:
        return {"type": "tool_result", "name": m.group(1), "ok": m.group(2) == "ok",
                "chars": _n(m.group(3)), "s": _n(m.group(4)), "truncated": "truncated" in m.group(5)}
    m = RE_REFUSED.search(line)
    if m:
        return {"type": "refused", "tokens": _n(m.group(1)), "cap": _n(m.group(2)),
                "msgs": int(m.group(3)), "tools": int(m.group(4)),
                "last_role": m.group(5), "last_preview": m.group(6)[:70]}
    m = RE_WARN.search(line)
    if m:
        return {"type": "warning", "waiters": int(m.group(1)), "waited_s": _n(m.group(2))}
    if RE_RAW.match(line):
        return {"type": "answer", "chars": len(line)}
    for tag, typ in (("starting the engine", "engine_start"), ("starting it again", "engine_start"),
                     ("the engine had stopped", "engine_exit"), ("ready: http", "engine_ready"),
                     ("stopping (", "engine_stopping"), ("still starting", "engine_warming"),
                     ("the engine reported an error", "engine_error"),
                     ("out of memory: the device has", "gate_refusal")):
        if tag in line:
            return {"type": typ, "text": line[:160]}
    return {"type": "log", "text": line[:160]}