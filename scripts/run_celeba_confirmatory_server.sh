#!/usr/bin/env bash
# Frozen CelebA confirmatory launcher for a single >=18 GiB CUDA GPU.
set -euo pipefail

MODE="${1:-all}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARGS=("$MODE" --config configs/celeba_e2e_confirmatory_seed10_19.yaml)
if [[ -n "${CELEBA_ROOT:-}" ]]; then
  ARGS+=(--data-root "$CELEBA_ROOT")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  ARGS+=(--output-root "$OUTPUT_ROOT")
fi

exec "$PYTHON_BIN" scripts/run_celeba_confirmatory_server.py "${ARGS[@]}" "$@"
