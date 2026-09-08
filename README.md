# Ollama Trace Lab

A local, zero-dependency tracer for your Ollama host. Send a prompt, watch the model
generate token by token, and see every timing, wire frame and probability the server
will give up.

## Run it

```bash
python3 server.py --ollama http://ollama.example.com:11434
open http://localhost:8777
```

Point it at your host in whichever way suits you — resolution order is
`--ollama` > `$OLLAMA_HOST` > `tracer.local.json` > `http://localhost:11434`:

```bash
python3 server.py --ollama http://ollama.example.com:11434 --port 8777
OLLAMA_HOST=http://ollama.example.com:11434 python3 server.py
```

Or write it once into `tracer.local.json` beside `server.py` — that file is
gitignored, so a private hostname never reaches the repo:

```json
{"ollama": "http://ollama.example.com:11434", "port": 8777}
```

Python 3.8+ stdlib only — no pip installs, no internet needed.

The server is the reason there is no CORS problem: the browser talks to
`localhost:8777`, and `server.py` talks to your Ollama host. It also sits on the socket, so it
can time things the browser cannot see (TCP connect, time-to-headers, per-frame byte
counts and arrival timestamps).

## What you're looking at

| Panel | What it shows |
|---|---|
| **04 Metrics** | TTFT, live tok/s, engine eval + prompt-eval rates, model load time, ITL p50/p95/stdev, wire bytes, bytes-per-token, mean confidence and entropy, finish reason |
| **05 Phase waterfall** | Top block is wall clock measured at the socket: TCP connect → request sent → response headers → first byte → first token → generation. Bottom block is Ollama's own nanosecond timers: `load_duration`, `prompt_eval_duration` (prefill), `eval_duration` (decode), and whatever is left over |
| **06 Token stream** | Every token as it arrives, colored by latency, confidence or entropy. Hover for a per-token readout, click to pin it in the inspector. The scrubber replays the generation at its real recorded timing |
| **07 Throughput** | Instantaneous tok/s (12-token sliding window) against cumulative average — the gap between them is where the run sped up or stalled |
| **08 Inter-token latency** | One bar per decode step with p50/p95/p99 markers, plus a histogram. Batch-scheduler stalls and KV-cache pressure show up here as isolated spikes |
| **09 Model certainty** | p(top-1) per step, top-k entropy, and the p1−p2 margin. Flat high line = the model is coasting; entropy spikes = branch points where sampling actually mattered |
| **Inspect tab** | The full candidate distribution for one decode step — what the model almost said, with probabilities, and whether the sampler took the argmax or a lower-ranked branch |
| **Model tab** | `/api/ps` (VRAM split, context, TTL) and `/api/show` (architecture, block count, head count, rope base, vocab size, quantization, chat template, baked system prompt, modelfile) |
| **Wire tab** | Every raw frame with arrival offset and byte count. Click one to expand the JSON |
| **Runs tab** | Run history; tick 2+ runs for an A/B table with the best value per metric highlighted, plus tok/s variance across the selection |
| **JSON tab** | Exact request body, response headers, and the final computed summary |

## Notes on the numbers

- **TTFT** is measured from the moment `server.py` starts the request to the first frame
  carrying non-empty content — it includes queueing, model load and prefill.
- **tok/s (wall)** counts stream frames over wall time. **eval rate** comes from Ollama's
  own `eval_count / eval_duration`. They differ by network and serialisation overhead;
  the "transport overhead" tile is the difference.
- **Entropy** is computed over the returned top-k only, so it is a truncated lower bound.
  Raise `top_logprobs` for a better estimate.
- One stream frame is usually one token, but a server is free to coalesce. Where Ollama
  reports `eval_count`, that number is shown alongside the stream count.

## Endpoint modes

- **native** → `POST /api/chat`. Gives the nanosecond duration timers, `num_ctx`,
  `top_k`, `min_p`, `repeat_penalty`, `think`, and `keep_alive`. Use this by default.
- **/v1** → `POST /v1/chat/completions`. OpenAI-compatible; returns token usage but no
  duration timers, and only the OpenAI-standard sampler knobs.

Logprobs support depends on your Ollama version. Leave "Request logprobs" on — if the
server rejects it, the tracer says so and retries the same request without them, and the
certainty panel just stays empty. If native mode returns no logprobs, try `/v1`.

## Keeping private data out of the repo

`scripts/scan_secrets.py` refuses to let credentials, real addresses or captured
runs reach a commit. Install the hook once after cloning:

```bash
scripts/install-hooks.sh     # sets core.hooksPath, runs the scan on every commit
python3 scripts/scan_secrets.py          # scan tracked files on demand
python3 scripts/scan_secrets.py --staged # what the hook runs
```

It flags:

| Class | Examples |
|---|---|
| Credentials | private keys, `AKIA…` / `ghp_…` / `xox…` / `sk-ant-…` / `AIza…` / `hf_…` tokens, JWTs, `user:pass@host` URLs, generic `password = "…"` assignments |
| Real IPs | anything outside loopback and the RFC 5737/3849 documentation ranges — **private addresses included**, since `10.x` and `192.168.x` are exactly what you don't want published |
| Internal hostnames | URL hosts with no public TLD (`http://gpu-node-07:11434`), plus `*.local`, `*.internal`, `*.corp`, `*.lan` | <!-- scan:allow -->
| Your own terms | anything in `.secretscan-denylist`, one per line — that file is gitignored, so your internal names never enter the repo |
| Run captures | any file under `runs/` — they contain your real prompts and completions |

Placeholders like `changeme`, `<your-token>` and `$VAR` are ignored, and
documentation addresses (`192.0.2.0/24`, `203.0.113.0/24`, `example.com`) pass
cleanly — use those in docs rather than your real host. For a deliberate
exception, put `scan:allow` in a comment on that line; to skip a whole file, add
a glob to `.secretscanignore`. The same scan runs in CI on every push and PR.

Two things are gitignored by default and should stay that way: `runs/` and
`tracer.local.json`.

## Files

- `server.py` — proxy, instrumentation, SSE trace stream, run persistence
- `index.html` — the whole UI (no CDN, no build step)
- `runs/` — one JSON per run: summary, every token, every frame (gitignored)
- `scripts/scan_secrets.py` — the secret / IP / hostname scan
- `.githooks/pre-commit` — runs that scan before every commit
- `mock_ollama.py` — fake Ollama on port 11999 for working on the UI offline:
  `python3 mock_ollama.py` then `python3 server.py --ollama http://127.0.0.1:11999`

## If something is off

- **"Cannot reach upstream"** — check `curl http://<your-host>:11434/api/version` from
  the same machine. The tracer resolves the hostname from wherever `server.py` runs.
- **No candidate bars** — the model or Ollama build isn't returning logprobs; see above.
- **Load time is huge on the first run** — that's a cold model load; run again to see
  warm numbers, or use the "Unload model from VRAM" button to force a cold start.
