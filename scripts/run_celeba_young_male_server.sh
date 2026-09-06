#!/usr/bin/env bash
# Sealed CelebA Young/Male RepairValue-Q replication (seeds 60--79).
set -euo pipefail

MODE="${1:-all}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARGS=("$MODE" --config configs/celeba_e2e_young_male_rvq_seed60_79.yaml)
if [[ -n "${CELEBA_ROOT:-}" ]]; then
  ARGS+=(--data-root "$CELEBA_ROOT")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  ARGS+=(--output-root "$OUTPUT_ROOT")
fi

exec "$PYTHON_BIN" scripts/run_celeba_young_male_server.py "${ARGS[@]}" "$@"
