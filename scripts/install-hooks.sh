#!/bin/sh
# One-time setup: route git hooks at .githooks/ so the secret scan runs on commit.
set -e
root="$(git rev-parse --show-toplevel)"
chmod +x "$root/.githooks/pre-commit" "$root/scripts/scan_secrets.py"
git -C "$root" config core.hooksPath .githooks
echo "pre-commit secret scan installed (core.hooksPath = .githooks)"
