# llm-scope — a continuous LLM-conversation debugger for the maya engine

One picture of one conversation, from the first token to the last: prompt size,
context growth, prefill bursts, decode rate, and the I/O underneath — live.

The collector tails the engine+server log (live timestamps; the log itself is
untimestamped), the gpu_monitor log, and samples `/proc` every 2 s
(PSI-io, NVMe MB/s, Dirty). Everything lands in `data/events.jsonl` and is
served as a multi-lane SVG dashboard.

## Run

```bash
cd ~/projects/llm-scope
setsid nohup python3 llm-scope.py --port 8090 --log /path/to/engine.log > llm-scope.out 2>&1 &
```

Dashboard: `http://<box-ip>:8090/` (any host that can reach the box).
Read-only against the engine; safe to run beside it.

## The lanes

| Lane | Shows | What to look for |
|---|---|---|
| PROMPT TOKENS | each request's prompt size; red dashed line = the 65,536 cap; red ✕ = a refusal; a sudden drop = a compaction | the sawtooth (grow → compact → grow); how close sessions ride the cap; the 82,770-token refusal class |
| PREFILL | one amber bar per request (height = wall ms); green dots = % of the prompt reused from a checkpoint | the fixed ~24-28 GB expert staging per 16K chunk — a long prompt is N back-to-back bars; high reuse = the slot is earning its keep |
| DECODE | tok/s per request (blue); dim line = draft acceptance % | the 10-11 tok/s streaming band; acceptance ~90% |
| GPU % | monitor samples (util/temp/power/held/free in tooltip) | 95% = working; 0% with io busy = a pause between requests |
| PSI-IO full % | the meter the history was blind to | idle 0%; decode band 5-10%; >20% sustained = heavier than anything measured; climbing while GPU idle = the write-path stall class |
| NVMe read MB/s | device throughput | ~1.5 GB/s decode, 5-6 GB/s prefill bursts, 8.8 GB/s ceiling |
| Dirty MiB | unflushed writes | ~7 MB idle; ~400-900 MB spikes = the slot-save boundary write; *stuck* high + PSI-io climbing while GPU idles = the pre-freeze signature |
| EVENTS | engine life, warnings, refusals, slot saves, tool results | restarts, queued-request warnings, refusals |

## API

- `GET /api/events?since=<epoch>` — new events as JSON
- `GET /api/state` — one-screen summary

## Files

- `parser.py` — engine-log line → event (every real line pattern; unmatched lines kept as `log`, nothing dropped)
- `llm-scope.py` — collector threads + HTTP server
- `index.html` — the dashboard
- `data/events.jsonl` — the record (append-only; survives restarts of this tool, not of the box)

## Why this exists

The 2026-10-26 hard locks died at the decode→prefill boundary with every
subsystem "healthy" and no witness. The conversation-level telemetry the engine
already prints plus the io pressure we added make that boundary visible
per-request and per-second — this tool is where it gets seen.
Cross-refs: `~/projects/maya/docs/PSI-IO.md`, `docs/COMPACTION.md`,
`docs/2026-10-26-hardlock-1521-headroom40.md`.