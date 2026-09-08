#!/usr/bin/env python3
"""
scan_secrets.py — block credentials, real IP addresses and internal hostnames
from being committed.

Usage:
    python3 scripts/scan_secrets.py                 # scan the working tree
    python3 scripts/scan_secrets.py --staged        # scan only staged changes (pre-commit)
    python3 scripts/scan_secrets.py --all           # ignore .gitignore, scan everything
    python3 scripts/scan_secrets.py path/to/file …  # scan specific paths

Exit code 0 = clean, 1 = findings, 2 = usage error.

What it flags
  - credentials: private keys, cloud keys, provider tokens, JWTs, generic
    `password = "…"` style assignments
  - IP addresses: anything that is not loopback or an IETF documentation range.
    Private ranges (10/8, 172.16/12, 192.168/16, 100.64/10, 169.254/16) are
    flagged too — an internal address is exactly the thing you don't want public
  - internal hostnames: URL hosts with no public TLD (`http://buildbox:11434`),  scan:allow
    and *.local / *.internal / *.corp / *.lan names
  - anything listed in .secretscan-denylist (one term per line; that file is
    itself gitignored so your internal names never reach the repo)
  - run captures: any file under runs/ contains real prompts and completions,
    so committing one is refused outright

Escape hatches
  - append `scan:allow` in a comment on the same line for a deliberate example
  - add a path glob to .secretscanignore
"""

import argparse
import ipaddress
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW_MARK = "scan:allow"

# ---------------------------------------------------------------- credentials
CRED_RULES = [
    ("private key",        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY")),
    ("AWS access key id",  re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("AWS secret key",     re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][A-Za-z0-9/+=]{40}['\"]")),
    ("GitHub token",       re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("Slack token",        re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key",     re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Anthropic key",      re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("OpenAI-style key",   re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("Stripe live key",    re.compile(r"\b[rs]k_live_[A-Za-z0-9]{20,}\b")),
    ("HuggingFace token",  re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("JWT",                re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("basic-auth URL",     re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@")),
    ("bearer token",       re.compile(r"(?i)\b(?:authorization|bearer)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{24,}")),
]

PLACEHOLDER = re.compile(
    r"(?i)^(?:x{3,}|\.{3,}|<[^>]*>|\$\{?[a-z_][a-z0-9_]*\}?|changeme|your[-_ ]?\w*|"
    r"placeholder|example|dummy|redacted|none|null|true|false|test|password|secret|"
    r"token|sk-\.\.\.|\*+)$"
)
GENERIC_ASSIGN = re.compile(
    r"(?i)\b(passwd|password|secret|token|api[_-]?key|apikey|access[_-]?key|"
    r"client[_-]?secret|private[_-]?key|credential)s?\b\s*[:=]\s*"
    r"['\"]([^'\"]{6,})['\"]"
)

# ---------------------------------------------------------------- addresses
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6 = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b")

# Reserved for documentation (RFC 5737 / 3849) — safe to publish.
DOC_NETS = [ipaddress.ip_network(n) for n in
            ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")]
# Loopback / unspecified / broadcast — not identifying.
def _ip_ok(ip):
    if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
        return True
    if str(ip) in ("255.255.255.255", "0.0.0.0"):
        return True
    return any(ip in n for n in DOC_NETS)

def _ip_kind(ip):
    if ip.is_private or ip.is_link_local:
        return "private/internal IP"
    return "public IP"

# ---------------------------------------------------------------- hostnames
URL_HOST = re.compile(r"\b[a-z][a-z0-9+.\-]*://(?:[^/@\s]+@)?([A-Za-z0-9_.\-]+)(?::\d+)?", re.I)
INTERNAL_TLD = re.compile(r"(?i)\.(?:local|internal|intranet|lan|corp|home|priv|test)\b")
PUBLIC_HOSTS = {
    "localhost", "example.com", "www.example.com", "example.org", "example.net",
    "github.com", "api.github.com", "raw.githubusercontent.com", "ollama.com",
    "cdnjs.cloudflare.com", "fonts.googleapis.com", "fonts.gstatic.com",
}
# A registrable public name has a dot and a plausible TLD.
PUBLIC_NAME = re.compile(r"^(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,24}$")

BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz",
              ".tar", ".woff", ".woff2", ".ttf", ".ico", ".mp4", ".so", ".dylib"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
# These hold the terms to look for, not leaks; scanning them is pure noise.
ALWAYS_SKIP = {".secretscan-denylist", ".secretscanignore"}
MAX_BYTES = 2_000_000


def load_lines(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(ln)
    return out


def redact(s, keep=4):
    s = s.strip()
    if len(s) <= keep * 2:
        return s[:keep] + "…"
    return s[:keep] + "…" + s[-keep:]


def scan_text(path, text, denylist, findings):
    if path.replace("\\", "/").startswith("runs/"):
        findings.append((path, 0, "run capture",
                         "files under runs/ hold real prompts and completions"))
        return

    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line:
            continue
        stripped = line.strip()

        for label, rx in CRED_RULES:
            m = rx.search(line)
            if m:
                findings.append((path, lineno, label, redact(m.group(0))))

        m = GENERIC_ASSIGN.search(line)
        if m and not PLACEHOLDER.match(m.group(2).strip()):
            findings.append((path, lineno, "hardcoded %s" % m.group(1).lower(),
                             redact(m.group(2))))

        for m in IPV4.finditer(line):
            try:
                ip = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue                      # 999.1.1.1, version strings, etc.
            if not _ip_ok(ip):
                findings.append((path, lineno, _ip_kind(ip), str(ip)))

        for m in IPV6.finditer(line):
            try:
                ip = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue
            if not _ip_ok(ip):
                findings.append((path, lineno, _ip_kind(ip), str(ip)))

        for m in URL_HOST.finditer(line):
            host = m.group(1)
            hl = host.lower()
            if hl in PUBLIC_HOSTS or hl.startswith("127.") or hl == "0.0.0.0":
                continue
            try:
                ipaddress.ip_address(host)
                continue                      # already handled by the IP rules
            except ValueError:
                pass
            if INTERNAL_TLD.search(hl):
                findings.append((path, lineno, "internal hostname", host))
            elif not PUBLIC_NAME.match(host):
                findings.append((path, lineno, "non-public hostname", host))

        for term in denylist:
            if term.lower() in stripped.lower():
                findings.append((path, lineno, "denylisted term", term))


def iter_files(paths, staged, scan_all):
    if paths:
        for p in paths:
            yield os.path.relpath(os.path.abspath(p), ROOT)
        return
    if staged:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                             cwd=ROOT, capture_output=True, text=True)
        for p in out.stdout.split("\n"):
            if p.strip():
                yield p.strip()
        return
    if not scan_all:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            for p in out.stdout.split("\n"):
                if p.strip():
                    yield p.strip()
            return
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield os.path.relpath(os.path.join(dirpath, fn), ROOT)


def main():
    ap = argparse.ArgumentParser(description="Block secrets and real addresses from the repo")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--staged", action="store_true", help="scan staged changes only")
    ap.add_argument("--all", action="store_true", help="walk the tree, ignoring git")
    a = ap.parse_args()

    denylist = load_lines(".secretscan-denylist")
    ignores = load_lines(".secretscanignore")
    findings = []
    scanned = 0

    for rel in iter_files(a.paths, a.staged, a.all):
        if rel in ALWAYS_SKIP:
            continue
        if any(re.fullmatch(g.replace("*", ".*"), rel) for g in ignores):
            continue
        if os.path.splitext(rel)[1].lower() in BINARY_EXT:
            continue
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full) or os.path.getsize(full) > MAX_BYTES:
            continue
        try:
            with open(full, "rb") as f:
                raw = f.read()
            if b"\0" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        scan_text(rel.replace("\\", "/"), text, denylist, findings)

    if not findings:
        print("secret scan: clean (%d files)" % scanned)
        return 0

    seen, uniq = set(), []
    for f in findings:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    print("secret scan: %d finding(s) in %d files\n" % (len(uniq), scanned), file=sys.stderr)
    for path, lineno, kind, detail in uniq:
        loc = "%s:%d" % (path, lineno) if lineno else path
        print("  %-14s %s  → %s" % (kind, loc, detail), file=sys.stderr)
    print("\nFix the finding, or if it is genuinely safe append a comment containing"
          "\n'%s' on that line. Nothing was committed." % ALLOW_MARK, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
