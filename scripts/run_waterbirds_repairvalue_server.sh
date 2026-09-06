#!/usr/bin/env bash
# Frozen Waterbirds RepairValue-Q confirmation for one >=18 GiB CUDA GPU.
set -euo pipefail

MODE="${1:-all}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARGS=("$MODE" --config configs/waterbirds_e2e_repairvalue_seed20_39.yaml)
if [[ -n "${WATERBIRDS_ROOT:-}" ]]; then
  ARGS+=(--data-root "$WATERBIRDS_ROOT")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  ARGS+=(--output-root "$OUTPUT_ROOT")
fi

exec "$PYTHON_BIN" scripts/run_waterbirds_repairvalue_server.py "${ARGS[@]}" "$@"

