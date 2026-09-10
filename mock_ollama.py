#!/usr/bin/env python3
"""Mock Ollama for testing the tracer without a real server. Not shipped for real use."""
import json, math, random, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT = ("A transformer scores every token in its vocabulary at each step, using attention "
        "over the whole context to build one hidden vector for the current position.\n"
        "That vector is projected to logits, softmaxed into a distribution, and the sampler "
        "picks from it under temperature and top-p.\nThe chosen token is appended and the "
        "loop runs again.")
TOKENS = [t for t in __import__("re").findall(r"\s*\S+", TEXT)]
VOCAB = ["the", " a", " token", " model", " and", " context", " layer", " which", " that", " step"]


def logprob_set(chosen, k):
    base = -random.random() * 1.4
    out = [{"token": chosen, "logprob": round(base, 4)}]
    for i in range(k - 1):
        out.append({"token": random.choice(VOCAB), "logprob": round(base - 0.4 - i * 0.55 - random.random(), 4)})
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/api/version": return self._j({"version": "0.12.3"})
        if self.path == "/api/tags":
            return self._j({"models": [
                {"name": "qwen3:8b", "size": 5225000000,
                 "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M", "family": "qwen3"}},
                {"name": "llama3.1:8b", "size": 4920000000,
                 "details": {"parameter_size": "8.0B", "quantization_level": "Q4_0", "family": "llama"}}]})
        if self.path == "/api/ps":
            return self._j({"models": [{"name": "qwen3:8b", "model": "qwen3:8b", "size": 6900000000,
                                        "size_vram": 6100000000, "context_length": 8192,
                                        "expires_at": "2026-09-08T21:14:02Z"}]})
        return self._j({"error": "nf"}, 404)

    def do_POST(self):
        if self.path == "/api/show":
            return self._j({
                "details": {"family": "qwen3", "families": ["qwen3"], "parameter_size": "8.2B",
                            "quantization_level": "Q4_K_M", "format": "gguf"},
                "model_info": {"qwen3.context_length": 40960, "qwen3.embedding_length": 4096,
                               "qwen3.block_count": 36, "qwen3.attention.head_count": 32,
                               "qwen3.rope.freq_base": 1000000, "general.vocab_size": 151936},
                "capabilities": ["completion", "tools", "thinking"],
                "parameters": "stop \"<|im_end|>\"\ntemperature 0.6\ntop_p 0.95",
                "template": "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{- end }}",
                "modelfile": "FROM qwen3:8b\nPARAMETER temperature 0.6"})
        if self.path in ("/api/chat", "/v1/chat/completions"):
            return self.stream(self.path == "/v1/chat/completions")
        return self._j({"error": "nf"}, 404)

    def stream(self, openai):
        body = self._body()
        want_lp = bool(body.get("logprobs"))
        k = int(body.get("top_logprobs") or 5)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if openai else "application/x-ndjson")
        self.send_header("Connection", "close"); self.end_headers()
        self.close_connection = True
        time.sleep(0.25)  # simulate prefill
        n = 0
        if body.get("think") or body.get("model", "").startswith("reasoner"):
            for tok in ["Let", " me", " check", " the", " constraint", ":", " three",
                        " sentences", ".", " (Check)", "\n"]:
                time.sleep(0.03)
                n += 1
                d = {"model": body.get("model"),
                     "message": {"role": "assistant", "content": "", "thinking": tok},
                     "done": False}
                if want_lp:
                    d["logprobs"] = [{"token": tok, "logprob": -random.random(),
                                      "top_logprobs": logprob_set(tok, k)}]
                self.wfile.write(json.dumps(d).encode() + b"\n"); self.wfile.flush()
        for i, tok in enumerate(TOKENS):
            time.sleep(max(0.008, random.gauss(0.032, 0.014)) + (0.14 if i in (7, 23) else 0))
            n += 1
            if openai:
                d = {"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
                if want_lp:
                    d["choices"][0]["logprobs"] = {"content": [
                        {"token": tok, "logprob": -random.random(),
                         "top_logprobs": logprob_set(tok, k)}]}
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            else:
                d = {"model": body.get("model"), "message": {"role": "assistant", "content": tok},
                     "done": False}
                if want_lp:
                    d["logprobs"] = [{"token": tok, "logprob": -random.random(),
                                      "top_logprobs": logprob_set(tok, k)}]
                self.wfile.write(json.dumps(d).encode() + b"\n")
            self.wfile.flush()
        if openai:
            fin = {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 34, "completion_tokens": n, "total_tokens": 34 + n}}
            self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\ndata: [DONE]\n\n")
        else:
            fin = {"model": body.get("model"), "message": {"role": "assistant", "content": ""},
                   "done": True, "done_reason": "stop",
                   "total_duration": int(2.9e9), "load_duration": int(0.42e9),
                   "prompt_eval_count": 34, "prompt_eval_duration": int(0.21e9),
                   "eval_count": n, "eval_duration": int(2.1e9)}
            self.wfile.write(json.dumps(fin).encode() + b"\n")
        self.wfile.flush()


if __name__ == "__main__":
    s = ThreadingHTTPServer(("127.0.0.1", 11999), H)
    s.daemon_threads = True
    print("mock ollama on 11999")
    s.serve_forever()
