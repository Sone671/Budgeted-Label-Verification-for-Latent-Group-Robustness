#!/usr/bin/env bash
# Reproduce the frozen innovation-validation pilots on a Linux server.
# Required Stage-1 inputs are documented in SERVER_VALIDATION_README.md.

set -euo pipefail

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_oof() {
  local config="$1"
  local noise="$2"
  local seed="$3"
  "$PYTHON_BIN" scripts/conformal_verification_pilot.py \
    --config "$config" \
    --noise "$noise" --seed "$seed" --budget 0.02 \
    --selection bh_guard --fdr-level 0.20 \
    --min-discovery-fraction 0.90 --fallback cpba \
    --oof-folds 5 --calibration-fraction 0.50 \
    --oof-cache-root outputs/server_validation/oof_cache \
    --output-root outputs/server_validation/oof_guard90
}

run_safe_cap() {
  local config="$1"
  local noise="$2"
  local method="$3"
  local output_root="$4"
  "$PYTHON_BIN" scripts/safe_cap_pilot.py \
    --config "$config" --noise "$noise" --method "$method" \
    --budgets 0.005,0.01,0.02,0.05,0.10 \
    --alphas 0.01,0.02,0.05 --temperature 5.0 \
    --output-root "$output_root"
}

case "$MODE" in
  smoke)
    "$PYTHON_BIN" -m pytest -q \
      tests/test_conformal_verification_pilot.py \
      tests/test_safe_cap_pilot.py
    run_oof configs/ablation_civilcomments_10seeds.yaml uniform20 0
    run_safe_cap configs/ablation_waterbirds_10seeds.yaml uniform20 \
      noise_score_cpba_only outputs/server_validation/safe_cap_cpba
    ;;
  full-oof)
    for config in \
      configs/ablation_waterbirds_10seeds.yaml \
      configs/ablation_celeba_10seeds.yaml \
      configs/ablation_civilcomments_10seeds.yaml; do
      for noise in uniform20 minority_high_40; do
        for seed in {0..9}; do
          run_oof "$config" "$noise" "$seed"
        done
      done
    done
    ;;
  safe-cap)
    for config in \
      configs/ablation_waterbirds_10seeds.yaml \
      configs/ablation_celeba_10seeds.yaml \
      configs/ablation_civilcomments_10seeds.yaml; do
      for noise in uniform20 minority_high_40; do
        run_safe_cap "$config" "$noise" noise_score_cpba_only \
          outputs/server_validation/safe_cap_cpba
        run_safe_cap "$config" "$noise" noise_score_budget_hybrid_v8 \
          outputs/server_validation/safe_cap_ngc
      done
    done
    ;;
  *)
    echo "Usage: $0 {smoke|full-oof|safe-cap}" >&2
    exit 2
    ;;
esac
