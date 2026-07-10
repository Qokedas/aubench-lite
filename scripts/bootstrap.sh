#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

echo "Environment ready. Activate with: source .venv/bin/activate"

