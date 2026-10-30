#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v harbor >/dev/null 2>&1; then
  uv tool install harbor
fi

printf '\nSetup complete. Checking tools...\n'
uv --version
harbor --version
docker version
