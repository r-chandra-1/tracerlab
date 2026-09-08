# Metrics reference — Ollama Trace Lab

Every number the tracer records, where it comes from, and what it actually tells you.

**Source key** — `socket`: measured by `server.py` on the connection itself · `engine`: reported by
Ollama in the final frame (native `/api/chat` only) · `derived`: computed by the tracer ·
`logprobs`: from the model's returned candidate distribution · `api`: a plain Ollama endpoint.

---

## 1. Wall-clock phases — where the time actually went

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **tcp connect** | socket | Time to open the TCP connection to the host | Isolates network/DNS cost from model cost. If this is tens of ms on a LAN, you have a name-resolution or routing problem, not a model problem |
| **req → headers** | socket | Request sent → response status line back | Ollama's queue admission. Large here = the server is busy with another request; the model hasn't started yet |
| **headers → 1st byte** | socket | Headers received → first stream byte | Where cold model load and prompt prefill live. The single biggest lever on perceived latency |
| **1st byte → 1st token** | socket | First byte → first frame carrying content | Should be ~0. If not, the server is emitting empty/keepalive frames before generating |
| **TTFT** | derived | Request start → first content token | The number your users feel. Everything above sums into it |
| **generation** | socket | First token → last token | Pure decode wall time |
| **wall total** | socket | Request start → stream end | End-to-end, including everything the engine timers don't see |
| **peer** | socket | Resolved IP:port actually connected to | Confirms you're hitting the box you think you are |

## 2. Engine timers — Ollama's own nanosecond accounting

Native `/api/chat` only. The `/v1` endpoint doesn't return these.

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **total_duration** | engine | Everything the server did, its own clock | Compare against wall total to size up transport overhead |
| **load_duration** | engine | Time to bring the model into memory | Huge on the first call, ~0 when warm. If it's large on *every* call, `keep_alive` is expiring or something is evicting the model from VRAM |
| **prompt_eval_count** | engine | Prompt tokens processed (prefill) | The real token cost of your prompt — check your char/4 estimate against it |
| **prompt_eval_duration** | engine | Time spent on prefill | Scales with prompt length. This is what long system prompts and RAG context cost you |
| **prompt eval rate** | derived | `prompt_eval_count / prompt_eval_duration` | Prefill throughput, tok/s. Compute-bound; a good proxy for raw GPU throughput. Falls sharply when the model spills out of VRAM |
| **eval_count** | engine | Tokens actually generated | Ground truth for output length; compare with the stream frame count to detect coalesced frames |
| **eval_duration** | engine | Time spent decoding | Pure generation |
| **eval rate** | derived | `eval_count / eval_duration` | The headline tok/s, unpolluted by network. Memory-bandwidth-bound — this is the number to quote when comparing quantizations |
| **done_reason** | engine | `stop`, `length`, `load`… | `length` means you hit `num_predict` and the answer is truncated — a very common silent failure |
| **context returned** | engine | Length of the returned context vector | Confirms how much conversation state the server is carrying forward |

## 3. Throughput

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **tok/s (wall)** | derived | Stream tokens ÷ wall generation time | What the client experiences, network included |
| **instantaneous tok/s** | derived | 12-token sliding window (chart) | Shows *within-run* degradation — the curve sagging as context grows is KV-cache pressure |
| **cumulative avg tok/s** | derived | Running average from first token (chart) | The gap between this and the instantaneous line tells you whether the run sped up or stalled after the start |
| **peak tok/s** | derived | Max of the instantaneous curve | Best-case decode rate when nothing else is competing |
| **transport overhead** | derived | wall total − `total_duration` | Cost of network + JSON serialisation. Should be small; if it rivals generation time, the network is your bottleneck, not the GPU |

## 4. Latency distribution — smoothness, not speed

Averages hide stalls; these don't.

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **ITL min / mean / max** | derived | Inter-token latency bounds | `max` is the worst pause a user sat through |
| **ITL p50** | derived | Median gap between tokens | The typical rhythm. ~1000/p50 = steady-state tok/s |
| **ITL p90 / p95 / p99** | derived | Tail latencies | Where stutter lives. A p99 many times the p50 means the server is time-slicing you against other requests |
| **ITL stdev** | derived | Jitter | Low mean + high stdev reads as *janky* even when average throughput looks fine. The single best "is this server healthy" number |
| **latency histogram** | derived | Distribution of all gaps | Bimodal = two regimes (e.g. GPU vs CPU-offloaded layers, or batch scheduling). One tight peak = clean, dedicated decode |
| **per-token bars** | derived | One bar per decode step | Isolated red spikes pinpoint *which* token stalled — usually batch admission of another request, or a KV-cache resize |

## 5. Model certainty — what the model nearly said

Requires logprobs support in your Ollama build. Empty otherwise.

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **logprob** | logprobs | Log-probability of the chosen token | Raw score for the step |
| **probability** | derived | `exp(logprob)` | Readable confidence. Long low-probability stretches are where hallucination risk concentrates |
| **top-k entropy** | derived | `−Σ p log₂ p` over returned candidates | How *undecided* the model was. Near 0 = one obvious continuation; high = a genuine branch point. Truncated to top-k, so it's a lower bound |
| **margin p1−p2** | derived | Gap between the top two candidates | A small margin means temperature/top_p actually changed the output here. Where sampler settings matter |
| **local perplexity** | derived | `1 / p(chosen)` | Per-step surprise, comparable to training-style perplexity |
| **chosen rank** | derived | Position of the emitted token in the candidate list | Rank 1 = argmax; rank > 1 = the sampler took a non-greedy branch. Set `temperature 0` and this should be 1 everywhere |
| **candidate distribution** | logprobs | Full top-k with probabilities (inspector) | The most direct view of "inner working" — the alternatives the model weighed at that exact position |
| **mean confidence** | derived | Average p(top-1) across the run | A single fluency/certainty score. Compare across prompts, quantizations or temperatures |
| **mean entropy** | derived | Average entropy across the run | Falls as you lower temperature; rises on ambiguous or out-of-distribution prompts |

## 6. Payload and wire

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **wire frames** | socket | Stream frames received | Compare with `eval_count`: more frames than tokens = keepalives; fewer = coalescing |
| **wire bytes** | socket | Total streamed bytes | Real bandwidth cost of streaming |
| **bytes / token** | derived | wire bytes ÷ tokens | JSON envelope overhead per token. ~500 B/token with logprobs on, ~60 B without — worth knowing before you stream to a phone |
| **frame timeline** | socket | Arrival offset + byte size per frame | Wire-level proof of when each chunk landed, independent of any parsing |
| **output tokens / prompt tokens** | derived / engine | Counts | Sanity check against your prompt-size assumptions |
| **text chars** | derived | Characters generated | Char-per-token ratio, a rough tokenizer-efficiency check for your language |

## 7. Capacity and placement

| Metric | Source | What it is | What it's useful for |
|---|---|---|---|
| **context fill** | derived | prompt + output vs `num_ctx` | Shows how close you are to truncation *before* it silently bites |
| **total size** | api `/api/ps` | Model footprint in memory | Sizing |
| **in VRAM / % GPU** | api `/api/ps` | Share resident on the GPU | **The single most important perf number.** Anything under 100% means layers are running on CPU, and eval rate collapses accordingly |
| **context_length (loaded)** | api `/api/ps` | Context the model was actually loaded with | Catches the case where your `num_ctx` request was ignored or clamped |
| **expires_at** | api `/api/ps` | When the model unloads | Explains a surprise `load_duration` on the next call |

## 8. Model internals

From `/api/show` — static facts about the loaded model.

| Metric | What it's useful for |
|---|---|
| **family / parameter_size / quantization / format** | The three things that set your speed/quality tradeoff. Q4_K_M vs Q8_0 shows up directly in eval rate |
| **context_length (max)** | The model's ceiling, as opposed to what it's loaded with |
| **embedding_length / block_count / head_count** | Architecture shape — explains why memory bandwidth, not FLOPs, is the limit |
| **rope.freq_base** | How the model handles long contexts; relevant if you're pushing past its native window |
| **vocab_size** | Tokenizer size; affects bytes-per-token for non-English text |
| **capabilities** | Whether `tools`, `thinking`, `vision` are available before you try to use them |
| **default parameters** | The sampler defaults baked into the model — your Param Lab values override these, and the difference explains behaviour changes |
| **chat template** | The exact Go template wrapping your messages. Essential when a prompt behaves differently than the same text sent raw |
| **baked system prompt** | A system prompt shipped with the model that you may be unknowingly stacking on top of |

## 9. Run-level comparison

| Metric | What it's useful for |
|---|---|
| **run history** | Every run persisted to `runs/*.json` with full token and frame detail |
| **A/B compare** | Side-by-side table with the best value per metric highlighted — the honest way to test a parameter change |
| **tok/s variance (mean ± sd, cv%)** | Set *Repeat* to 5–10 and read this before believing any single-run comparison. A cv above ~10% means your host is noisy and small differences are meaningless |

---

## Reading the whole thing at once

- **Slow first response, fast afterwards** → `load_duration`. Raise `keep_alive`.
- **Slow prefill, fast decode** → prompt is too long; check `prompt_eval_count` and the context bar.
- **Good average, bad feel** → high ITL stdev / p99. Look for spikes in panel 08.
- **Decode rate well below expectation** → check `% GPU` in the Model tab first; partial offload explains most of it.
- **Output stops mid-sentence** → `done_reason: length`. Raise `num_predict`.
- **Wall time ≫ engine total** → network, not the model. See transport overhead.
- **Output wanders or repeats** → look for long low-confidence, high-entropy stretches in panel 09, then lower temperature or raise `repeat_penalty` and compare runs.
