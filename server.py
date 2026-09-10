#!/usr/bin/env python3
"""
Ollama Trace Lab - local proxy + instrumentation server.

Serves the tracer UI and proxies to an Ollama host, capturing wire-level
detail the browser cannot see on its own: TCP connect time, time-to-headers,
per-frame byte counts and arrival timestamps, per-token inter-arrival deltas,
and logprob distributions when the server supports them.

Usage:
    python3 server.py                                   # http://localhost:11434
    python3 server.py --ollama http://ollama.example.com:11434
    python3 server.py --port 8777 --host 127.0.0.1
    OLLAMA_HOST=http://ollama.example.com:11434 python3 server.py

Several hosts can be configured and switched from the UI at runtime — useful
for comparing, say, a laptop against a GPU box. Drop a tracer.local.json next
to this file:

    {
      "port": 8777,
      "hosts": [
        {"label": "laptop",  "url": "http://localhost:11434"},
        {"label": "gpu box", "url": "http://ollama.example.com:11434"}
      ]
    }

The single-host form still works: {"ollama": "http://ollama.example.com:11434"}.
localhost is always offered even when it is not listed. That file is gitignored,
so your internal hostnames never reach the repo.
Resolution order for the default: --ollama > OLLAMA_HOST > tracer.local.json
> localhost.

Stdlib only. No pip installs.
"""

import argparse
import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

DEFAULT_OLLAMA = "http://localhost:11434"
LOCAL_CONFIG = "tracer.local.json"
TARGETS = []            # [{"label": str, "url": str}] offered in the UI
DEFAULT_URL = DEFAULT_OLLAMA
RUN_INDEX = []          # newest-last, in-memory summaries
RUN_LOCK = threading.Lock()
CANCEL = set()          # run ids the client asked to stop
CANCEL_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# upstream plumbing
# ----------------------------------------------------------------------------

def local_config():
    """Optional gitignored tracer.local.json so a private host never lives in tracked code."""
    p = os.path.join(HERE, LOCAL_CONFIG)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f) or {}
    except Exception as e:
        print("  warning: could not read %s (%s)" % (LOCAL_CONFIG, e))
        return {}


def parse_target(url):
    """Accept 'box', 'box:11434' or a full URL (path and all) and normalise it.

    Returns a self-contained dict; nothing global is mutated, so concurrent
    requests to different hosts cannot race each other.
    """
    u = (url or "").strip() or DEFAULT_URL
    if "://" not in u:
        u = "http://" + u
    p = urllib.parse.urlparse(u)
    scheme = p.scheme or "http"
    host = p.hostname or "localhost"
    port = p.port or (443 if scheme == "https" else 11434)
    return {"scheme": scheme, "host": host, "port": port,
            "url": "%s://%s:%d" % (scheme, host, port)}


def new_conn(tgt, timeout=600):
    if tgt["scheme"] == "https":
        return http.client.HTTPSConnection(tgt["host"], tgt["port"], timeout=timeout)
    return http.client.HTTPConnection(tgt["host"], tgt["port"], timeout=timeout)


def upstream_json(tgt, method, path, payload=None, timeout=30):
    """Simple non-streaming proxy call. Returns (status, obj_or_text, elapsed_ms)."""
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    t0 = time.perf_counter()
    conn = new_conn(tgt, timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            return resp.status, json.loads(raw.decode("utf-8", "replace")), ms
        except Exception:
            return resp.status, {"raw": raw.decode("utf-8", "replace")}, ms
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# request building
# ----------------------------------------------------------------------------

NATIVE_OPTION_KEYS = [
    "temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_penalty",
    "repeat_last_n", "presence_penalty", "frequency_penalty", "seed",
    "num_ctx", "num_predict", "num_keep", "num_batch", "num_gpu", "num_thread",
    "mirostat", "mirostat_tau", "mirostat_eta", "stop",
]
OPENAI_KEYS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "num_predict": "max_tokens",
    "stop": "stop",
}


def clean_options(raw):
    out = {}
    for k in NATIVE_OPTION_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if v is None or v == "":
            continue
        if k == "stop":
            if isinstance(v, str):
                v = [s for s in v.split("\n") if s.strip()]
            if not v:
                continue
        out[k] = v
    return out


def build_request(cfg):
    """Return (path, payload) for the chosen endpoint mode."""
    mode = cfg.get("mode", "native")
    model = cfg.get("model") or ""
    messages = cfg.get("messages") or []
    opts = clean_options(cfg.get("options") or {})
    want_lp = bool(cfg.get("logprobs"))
    topk = int(cfg.get("top_logprobs") or 5)

    if mode == "openai":
        payload = {"model": model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}}
        for src, dst in OPENAI_KEYS.items():
            if src in opts:
                payload[dst] = opts[src]
        if want_lp:
            payload["logprobs"] = True
            payload["top_logprobs"] = topk
        return "/v1/chat/completions", payload

    payload = {"model": model, "messages": messages, "stream": True}
    if opts:
        payload["options"] = opts
    if cfg.get("keep_alive") not in (None, ""):
        payload["keep_alive"] = cfg["keep_alive"]
    if cfg.get("think"):
        payload["think"] = True
    if cfg.get("format"):
        payload["format"] = cfg["format"]
    if want_lp:
        payload["logprobs"] = True
        payload["top_logprobs"] = topk
    return "/api/chat", payload


# ----------------------------------------------------------------------------
# frame parsing
# ----------------------------------------------------------------------------

def parse_frame(mode, line):
    """
    Normalise one upstream frame into:
      {done, text, thinking, logprob, top: [{token, logprob}], final: {...}, obj: {...}}
    Returns None for keepalives / [DONE] / unparseable noise.
    """
    s = line.strip()
    if not s:
        return None
    if mode == "openai":
        if s.startswith("data:"):
            s = s[5:].strip()
        if s == "[DONE]":
            return {"sentinel": True}
        if not s.startswith("{"):
            return None
    try:
        obj = json.loads(s)
    except Exception:
        return None

    out = {"obj": obj, "text": "", "thinking": "", "done": False,
           "logprob": None, "top": None, "final": None, "token_str": None}

    if mode == "openai":
        ch = (obj.get("choices") or [{}])
        c0 = ch[0] if ch else {}
        delta = c0.get("delta") or {}
        out["text"] = delta.get("content") or ""
        out["thinking"] = delta.get("reasoning_content") or delta.get("reasoning") or ""
        lp = c0.get("logprobs") or {}
        content = lp.get("content") or []
        if content:
            e = content[0]
            out["logprob"] = e.get("logprob")
            out["token_str"] = e.get("token")
            tops = e.get("top_logprobs") or []
            if tops:
                out["top"] = [{"token": t.get("token"), "logprob": t.get("logprob")} for t in tops]
        if c0.get("finish_reason"):
            out["done"] = True
            out["final"] = {"done_reason": c0.get("finish_reason")}
        if obj.get("usage"):
            out["done"] = True
            u = obj["usage"]
            f = out["final"] or {}
            f.update({
                "prompt_eval_count": u.get("prompt_tokens"),
                "eval_count": u.get("completion_tokens"),
                "total_tokens": u.get("total_tokens"),
            })
            out["final"] = f
        return out

    # native /api/chat
    msg = obj.get("message") or {}
    out["text"] = msg.get("content") or ""
    out["thinking"] = msg.get("thinking") or ""
    lp = obj.get("logprobs")
    if isinstance(lp, list) and lp:
        e = lp[0] if isinstance(lp[0], dict) else {}
        out["logprob"] = e.get("logprob")
        out["token_str"] = e.get("token")
        tops = e.get("top_logprobs") or e.get("top") or []
        if tops:
            out["top"] = [{"token": t.get("token"), "logprob": t.get("logprob")} for t in tops]
    elif isinstance(lp, dict):
        out["logprob"] = lp.get("logprob")
        out["token_str"] = lp.get("token")
        tops = lp.get("top_logprobs") or []
        if tops:
            out["top"] = [{"token": t.get("token"), "logprob": t.get("logprob")} for t in tops]
    if obj.get("done"):
        out["done"] = True
        out["final"] = {k: obj.get(k) for k in (
            "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration",
            "done_reason", "context") if obj.get(k) is not None}
    return out


LOGPROB_ERR = re.compile(r"logprob", re.I)


# ----------------------------------------------------------------------------
# the trace stream
# ----------------------------------------------------------------------------

class Emitter:
    def __init__(self, wfile):
        self.w = wfile
        self.lock = threading.Lock()
        self.dead = False

    def send(self, ev):
        if self.dead:
            return
        try:
            with self.lock:
                self.w.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self.w.flush()
        except Exception:
            self.dead = True


def run_trace(cfg, emit):
    run_id = cfg.get("run_id") or uuid.uuid4().hex[:12]
    tgt = parse_target(cfg.get("host"))
    mode = cfg.get("mode", "native")
    path, payload = build_request(cfg)
    attempts = [(payload, bool(cfg.get("logprobs")))]

    result = None
    for attempt_i, (pl, had_lp) in enumerate(attempts):
        result = _one_attempt(run_id, mode, path, pl, emit, cfg, tgt)
        if result.get("retry_without_logprobs") and had_lp:
            emit({"t": "notice", "level": "warn",
                  "msg": "Server rejected logprobs on this endpoint - retrying without them."})
            pl2 = dict(pl)
            pl2.pop("logprobs", None)
            pl2.pop("top_logprobs", None)
            attempts.append((pl2, False))
            continue
        break
    return result


def _one_attempt(run_id, mode, path, payload, emit, cfg, tgt):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/x-ndjson"}

    emit({"t": "request", "run_id": run_id, "mode": mode,
          "url": tgt["url"] + path, "host": tgt["url"], "headers": headers,
          "payload": payload, "bytes": len(body)})

    T0 = time.perf_counter()

    def ms():
        return (time.perf_counter() - T0) * 1000.0

    conn = new_conn(tgt, timeout=cfg.get("timeout") or 900)
    marks = {}
    frames = []
    steps = []            # every decode step, thinking and content alike
    total_bytes = 0
    final_meta = {}
    last_tok_t = None
    text_buf = []
    think_buf = []
    err = None

    try:
        try:
            conn.connect()
        except socket.gaierror as e:
            emit({"t": "error", "fatal": True,
                  "msg": "DNS failed for '%s' - is that hostname reachable from the machine "
                         "running server.py? (%s)" % (tgt["host"], e)})
            return {"error": "dns"}
        except OSError as e:
            emit({"t": "error", "fatal": True,
                  "msg": "Cannot connect to %s - %s" % (tgt["url"], e)})
            return {"error": "connect"}
        marks["connect_ms"] = ms()
        try:
            peer = conn.sock.getpeername()
            marks["peer"] = "%s:%s" % (peer[0], peer[1])
        except Exception:
            marks["peer"] = None
        emit({"t": "phase", "name": "connect", "at": marks["connect_ms"], "peer": marks["peer"]})

        conn.request("POST", path, body=body, headers=headers)
        marks["sent_ms"] = ms()
        emit({"t": "phase", "name": "sent", "at": marks["sent_ms"]})

        resp = conn.getresponse()
        marks["headers_ms"] = ms()
        hdrs = dict(resp.getheaders())
        emit({"t": "phase", "name": "headers", "at": marks["headers_ms"],
              "status": resp.status, "reason": resp.reason, "headers": hdrs})

        if resp.status >= 400:
            raw = resp.read().decode("utf-8", "replace")
            if LOGPROB_ERR.search(raw) and payload.get("logprobs"):
                return {"retry_without_logprobs": True, "detail": raw}
            emit({"t": "error", "fatal": True,
                  "msg": "Upstream HTTP %d: %s" % (resp.status, raw[:2000])})
            return {"error": "http", "status": resp.status}

        i = 0
        for raw_line in resp:
            if is_cancelled(run_id):
                emit({"t": "notice", "level": "warn", "msg": "Stopped by user."})
                break
            at = ms()
            if marks.get("first_byte_ms") is None:
                marks["first_byte_ms"] = at
                emit({"t": "phase", "name": "first_byte", "at": at})
            nbytes = len(raw_line)
            total_bytes += nbytes
            line = raw_line.decode("utf-8", "replace")
            fr = parse_frame(mode, line)
            frame_rec = {"i": i, "at": round(at, 3), "bytes": nbytes,
                         "raw": line.rstrip("\n")}
            frames.append(frame_rec)
            emit({"t": "frame", **frame_rec})
            i += 1
            if fr is None or fr.get("sentinel"):
                continue

            # A reasoning model spends most of its decode budget in `thinking`.
            # Those are real decode steps and must drive the timing metrics,
            # otherwise the whole dashboard reads "-" while the model works.
            for kind, txt in (("thinking", fr.get("thinking")), ("content", fr.get("text"))):
                if not txt:
                    continue
                if marks.get("ttft_ms") is None:
                    marks["ttft_ms"] = at
                    emit({"t": "phase", "name": "first_token", "at": at})
                if kind == "content" and marks.get("first_content_ms") is None:
                    marks["first_content_ms"] = at
                    emit({"t": "phase", "name": "first_content", "at": at})
                dt = None if last_tok_t is None else at - last_tok_t
                last_tok_t = at
                rec = {"i": len(steps), "kind": kind, "at": round(at, 3),
                       "dt": None if dt is None else round(dt, 3),
                       "text": txt, "bytes": len(txt.encode()),
                       "logprob": fr.get("logprob"), "top": fr.get("top"),
                       "token_str": fr.get("token_str")}
                steps.append(rec)
                (think_buf if kind == "thinking" else text_buf).append(txt)
                emit({"t": "token", **rec})

            if fr.get("final"):
                final_meta.update({k: v for k, v in fr["final"].items() if k != "context"})
                ctx = fr["final"].get("context")
                if ctx:
                    final_meta["context_len"] = len(ctx)
            if fr.get("done") and mode == "native":
                break

        marks["end_ms"] = ms()

    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
        emit({"t": "error", "fatal": True, "msg": err,
              "trace": traceback.format_exc()[-1500:]})
    finally:
        try:
            conn.close()
        except Exception:
            pass
        clear_cancel(run_id)

    summary = summarise(run_id, mode, payload, marks, steps,
                        frames, final_meta, total_bytes, "".join(text_buf),
                        "".join(think_buf), err, tgt)
    emit({"t": "final", **summary})
    persist(summary, steps, frames)
    return summary


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _timer_broken(count, ns):
    """Some runners (notably Ollama's MLX engine) report a near-zero duration.

    Dividing by it yields nonsense like 758,000 tok/s, which is worse than no
    number at all when you are using these figures to choose a runtime.
    """
    return bool(count and count > 1 and (not ns or ns < 1_000_000))   # < 1ms


def _rate(count, ns):
    if not count or not ns or _timer_broken(count, ns):
        return None
    return count / (ns / 1e9)


def summarise(run_id, mode, payload, marks, steps, frames,
              final_meta, total_bytes, text, think_text, err, tgt):
    dts = sorted([t["dt"] for t in steps if t.get("dt") is not None])
    content = [t for t in steps if t.get("kind") != "thinking"]
    thinking = [t for t in steps if t.get("kind") == "thinking"]
    end = marks.get("end_ms") or 0.0
    ttft = marks.get("ttft_ms")
    ntok = len(steps)
    gen_ms = (end - ttft) if (ttft is not None and end > ttft) else None

    # native nanosecond metrics, if present
    def ns(k):
        v = final_meta.get(k)
        return v if isinstance(v, (int, float)) else None

    total_ns, load_ns = ns("total_duration"), ns("load_duration")
    pe_ns, ev_ns = ns("prompt_eval_duration"), ns("eval_duration")
    pe_n, ev_n = ns("prompt_eval_count"), ns("eval_count")

    out = {
        "run_id": run_id,
        "ts": time.time(),
        "mode": mode,
        "host": tgt["url"],
        "model": payload.get("model"),
        "error": err,
        "wall": {
            "connect_ms": marks.get("connect_ms"),
            "sent_ms": marks.get("sent_ms"),
            "headers_ms": marks.get("headers_ms"),
            "first_byte_ms": marks.get("first_byte_ms"),
            "ttft_ms": ttft,
            "first_content_ms": marks.get("first_content_ms"),
            "end_ms": end,
            "peer": marks.get("peer"),
        },
        "counts": {
            "decode_steps": ntok,
            "stream_tokens": len(content),
            "thinking_tokens": len(thinking),
            "frames": len(frames),
            "wire_bytes": total_bytes,
            "text_chars": len(text),
        },
        "itl": {
            "min": dts[0] if dts else None,
            "p50": pct(dts, 50), "p90": pct(dts, 90),
            "p95": pct(dts, 95), "p99": pct(dts, 99),
            "max": dts[-1] if dts else None,
            "mean": (sum(dts) / len(dts)) if dts else None,
            "stdev": _stdev(dts),
        },
        "engine": {
            "total_duration_ms": total_ns / 1e6 if total_ns else None,
            "load_duration_ms": load_ns / 1e6 if load_ns else None,
            "prompt_eval_count": pe_n,
            "prompt_eval_duration_ms": pe_ns / 1e6 if pe_ns else None,
            "prompt_eval_rate": _rate(pe_n, pe_ns),
            "prompt_eval_timer_broken": _timer_broken(pe_n, pe_ns),
            "eval_count": ev_n,
            "eval_duration_ms": ev_ns / 1e6 if ev_ns else None,
            "eval_rate": _rate(ev_n, ev_ns),
            "eval_timer_broken": _timer_broken(ev_n, ev_ns),
            "done_reason": final_meta.get("done_reason"),
            "context_len": final_meta.get("context_len"),
            "total_tokens": final_meta.get("total_tokens"),
        },
        "derived": {
            "wall_tok_per_s": (ntok / (gen_ms / 1000.0)) if (gen_ms and ntok) else None,
            "wall_gen_ms": gen_ms,
            "bytes_per_token": (total_bytes / ntok) if ntok else None,
            "overhead_ms": (end - ((total_ns / 1e6) if total_ns else 0)) if total_ns else None,
        },
        "text": text,
        "thinking": think_text,
        "request": payload,
    }
    return out


def _stdev(vals):
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def persist(summary, tokens, frames):
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        p = os.path.join(RUNS_DIR, "%s.json" % summary["run_id"])
        with open(p, "w") as f:
            json.dump({"summary": summary, "tokens": tokens, "frames": frames}, f)
        with RUN_LOCK:
            RUN_INDEX.append({k: summary[k] for k in
                              ("run_id", "ts", "model", "mode", "host", "counts",
                               "engine", "wall", "itl", "derived")})
            del RUN_INDEX[:-200]
    except Exception:
        pass


def is_cancelled(rid):
    with CANCEL_LOCK:
        return rid in CANCEL


def clear_cancel(rid):
    with CANCEL_LOCK:
        CANCEL.discard(rid)


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OllamaTraceLab/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("TRACER_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ------------------------------------------------------------
    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, name, ctype):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            self._json({"error": "missing %s" % name}, 404)
            return
        with open(p, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except Exception:
            return {}

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        q = urllib.parse.parse_qs(u.query)
        tgt = parse_target((q.get("host") or [None])[0])
        if p == "/api/hosts":
            return self._json({"hosts": TARGETS, "default": DEFAULT_URL})
        if p == "/api/config":
            return self._json({"base": tgt["url"], "host": tgt["host"],
                               "port": tgt["port"], "hosts": TARGETS,
                               "default": DEFAULT_URL})
        if p == "/api/health":
            try:
                st, ver, ms = upstream_json(tgt, "GET", "/api/version", timeout=6)
                return self._json({"ok": st == 200, "status": st, "version": ver,
                                   "rtt_ms": ms, "base": tgt["url"]})
            except Exception as e:
                return self._json({"ok": False, "error": str(e), "base": tgt["url"]})
        if p == "/api/models":
            try:
                st, obj, ms = upstream_json(tgt, "GET", "/api/tags", timeout=15)
                return self._json({"ok": st == 200, "rtt_ms": ms, "data": obj})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})
        if p == "/api/ps":
            try:
                st, obj, ms = upstream_json(tgt, "GET", "/api/ps", timeout=10)
                return self._json({"ok": st == 200, "rtt_ms": ms, "data": obj})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})
        if p == "/api/runs":
            with RUN_LOCK:
                return self._json({"runs": list(reversed(RUN_INDEX))[:100]})
        if p.startswith("/api/run/"):
            rid = p.rsplit("/", 1)[-1]
            fp = os.path.join(RUNS_DIR, "%s.json" % re.sub(r"[^a-z0-9]", "", rid))
            if not os.path.exists(fp):
                return self._json({"error": "not found"}, 404)
            with open(fp) as f:
                return self._json(json.load(f))
        return self._json({"error": "not found", "path": p}, 404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p == "/api/show":
            body = self._body()
            try:
                st, obj, ms = upstream_json(parse_target(body.get("host")), "POST", "/api/show",
                                            {"model": body.get("model"), "verbose": False},
                                            timeout=30)
                return self._json({"ok": st == 200, "rtt_ms": ms, "data": obj})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})
        if p == "/api/stop":
            rid = (self._body() or {}).get("run_id")
            if rid:
                with CANCEL_LOCK:
                    CANCEL.add(rid)
            return self._json({"ok": True})
        if p == "/api/unload":
            body = self._body()
            try:
                st, obj, ms = upstream_json(parse_target(body.get("host")), "POST", "/api/chat",
                                            {"model": body.get("model"), "messages": [],
                                             "keep_alive": 0}, timeout=30)
                return self._json({"ok": st == 200, "data": obj})
            except Exception as e:
                return self._json({"ok": False, "error": str(e)})
        if p == "/api/trace":
            return self._trace()
        return self._json({"error": "not found", "path": p}, 404)

    def _trace(self):
        cfg = self._body()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        emit = Emitter(self.wfile).send
        emit({"t": "hello", "run_id": cfg.get("run_id"),
              "base": parse_target(cfg.get("host"))["url"]})
        try:
            run_trace(cfg, emit)
        except Exception as e:
            emit({"t": "error", "fatal": True, "msg": str(e),
                  "trace": traceback.format_exc()[-1500:]})
        try:
            self.wfile.flush()
        except Exception:
            pass


def build_targets(cli_url, cfg):
    """Assemble the host list offered in the UI, newest intent first.

    localhost is always present: it is the common case (Ollama on this machine)
    and costs nothing when unused.
    """
    out, seen = [], set()

    def add(label, url):
        if not url:
            return
        u = parse_target(url)["url"]
        if u in seen:
            return
        seen.add(u)
        out.append({"label": label, "url": u})

    explicit = cli_url or os.environ.get("OLLAMA_HOST")
    add("default", explicit)
    for h in (cfg.get("hosts") or []):
        if isinstance(h, dict):
            add(h.get("label") or parse_target(h.get("url"))["host"], h.get("url"))
        elif isinstance(h, str):
            add(parse_target(h)["host"], h)
    add("configured", cfg.get("ollama"))
    add("localhost", DEFAULT_OLLAMA)

    default_url = out[0]["url"] if out else DEFAULT_OLLAMA
    return out, default_url


def main():
    ap = argparse.ArgumentParser(description="Ollama Trace Lab")
    cfg = local_config()
    ap.add_argument("--ollama", default=None,
                    help="default Ollama base URL (also: $OLLAMA_HOST, tracer.local.json, "
                         "else %s)" % DEFAULT_OLLAMA)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.port is None:
        a.port = int(os.environ.get("TRACER_PORT") or cfg.get("port") or 8777)

    global TARGETS, DEFAULT_URL
    TARGETS, DEFAULT_URL = build_targets(a.ollama, cfg)
    os.makedirs(RUNS_DIR, exist_ok=True)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    url = "http://%s:%d/" % ("localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host, a.port)
    print("\n  Ollama Trace Lab")
    for t in TARGETS:
        print("  host     : %-9s %s%s" % (t["label"], t["url"],
                                          "   (default)" if t["url"] == DEFAULT_URL else ""))
    print("  ui       : %s" % url)
    print("  runs     : %s" % RUNS_DIR)
    print("\n  Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye.")


if __name__ == "__main__":
    main()
