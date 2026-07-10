#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f data/benchmark/public/manifest.json || {
  echo "Build data first: .venv/bin/cam1k-eval data build" >&2
  exit 1
}

docker compose build agent runner

# Solvability: the honest DAS baseline must score high.
.venv/bin/cam1k-eval grade baselines/delay_and_sum \
  --private data/benchmark/private \
  --public data/benchmark/public \
  --image cam1k-runner:latest \
  --seed 20260709 \
  --timeout 240 \
  --output reports/integration-delay-and-sum.json

# Floor: the zero baseline must gate to ~0 (only the zero-input case passes).
.venv/bin/cam1k-eval grade baselines/zero \
  --private data/benchmark/private \
  --public data/benchmark/public \
  --image cam1k-runner:latest \
  --seed 20260709 \
  --timeout 240 \
  --output reports/integration-zero.json

python3 - <<'EOF'
import json
das = json.load(open('reports/integration-delay-and-sum.json'))
zero = json.load(open('reports/integration-zero.json'))
print(f"delay_and_sum overall: {das['overall']:.4f}")
print(f"zero          overall: {zero['overall']:.4f}")
assert das['overall'] > 0.90, "solvability regression: DAS baseline should score >0.90"
assert zero['overall'] < 0.10, "floor regression: zero baseline should score <0.10"
print("integration OK")
EOF
