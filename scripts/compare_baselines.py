#!/usr/bin/env python
"""Compare baseline query methods on the baseline_test run and emit a go/no-go verdict.

Success criterion (medium): a legal TracIn variant reaches noise_score-level delta-WGA
at 2% budget and does not degrade (delta-WGA > 0) at 5% budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("outputs/baseline_test/stage1/results.csv")

LEGAL_TRACIN = ["tracin_val_uncertainty", "tracin_noise_weighted"]
ORDER = [
    "random",
    "loss",
    "noise_score",
    "noise_score_budget_hybrid",
    "noise_score_budget_hybrid_v8",
    "aum",
    "confident_learning",
    "tracin_val_uncertainty",
    "tracin_noise_weighted",
    "tracin_wga_oracle",
    "oracle_noise_group_balanced",
]

LABELS = {
    "random": "Random", "loss": "Loss", "entropy": "Entropy",
    "noise_score": "NoiseScore",
    "noise_score_budget_hybrid": "BHC",
    "noise_score_budget_hybrid_v8": "NGC",
    "aum": "AUM", "confident_learning": "Cleanlab",
    "tracin_val_uncertainty": "TracIn (val)",
    "tracin_noise_weighted": "TracIn (noise)",
    "tracin_wga_oracle": "TracIn (oracle)",
    "oracle_noise_group_balanced": "Oracle-GB",
}


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main() -> None:
    df = pd.read_csv(RESULTS)
    budgets = sorted(df["budget_fraction"].unique())
    noises = sorted(df["noise_name"].unique())

    for noise in noises:
        print("=" * 92)
        print(f"NOISE: {noise}")
        print("=" * 92)
        for budget in budgets:
            sub = df[(df["noise_name"] == noise) & (df["budget_fraction"] == budget)]
            rows = []
            for method in ORDER:
                m = sub[sub["method"] == method]
                if m.empty:
                    continue
                dw = m["delta_wga"].to_numpy()
                rows.append({
                    "method": method,
                    "delta_wga": f"{dw.mean():+.4f} ± {sem(dw):.4f}",
                    "wga": f"{m['wga'].mean():.3f}",
                    "num_corr": f"{m['num_corrected'].mean():.0f}",
                    "noise_prec": f"{m['noise_precision'].mean():.3f}",
                    "min_q_rate": f"{m['minority_query_rate'].mean():.3f}",
                })
            tbl = pd.DataFrame(rows)
            print(f"\n--- budget = {budget:.0%} ---")
            print(tbl.to_string(index=False))

    # --- Verdict ---
    print("\n" + "=" * 92)
    print("GO / NO-GO VERDICT")
    print("=" * 92)
    verdicts = []
    for noise in noises:
        ns2 = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.02) & (df["method"] == "noise_score")]["delta_wga"].mean()
        ns5 = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.05) & (df["method"] == "noise_score")]["delta_wga"].mean()
        print(f"\n{noise}: noise_score baseline  dWGA@2%={ns2:+.4f}  dWGA@5%={ns5:+.4f}")
        for legal in LEGAL_TRACIN:
            t2 = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.02) & (df["method"] == legal)]["delta_wga"].mean()
            t5 = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.05) & (df["method"] == legal)]["delta_wga"].mean()
            catch_up = t2 >= ns2 - 0.005
            no_degrade = t5 > 0
            tag = "GO" if (catch_up and no_degrade) else "NO-GO"
            verdicts.append(tag == "GO")
            print(f"  {legal:26s} dWGA@2%={t2:+.4f}  dWGA@5%={t5:+.4f}  "
                  f"catch_up={catch_up}  no_degrade={no_degrade}  -> {tag}")
        # oracle headroom
        for oracle in ["tracin_wga_oracle", "oracle_noise_group_balanced"]:
            o5 = df[(df["noise_name"] == noise) & (df["budget_fraction"] == 0.05) & (df["method"] == oracle)]["delta_wga"].mean()
            print(f"  [oracle] {oracle:24s} dWGA@5%={o5:+.4f}")

    print("\n" + "=" * 92)
    if all(verdicts):
        print("OVERALL: GO  -- legal TracIn reaches noise_score level @2% and holds @5%.")
    elif any(verdicts):
        print("OVERALL: PARTIAL -- some (noise, variant) combos pass; investigate failures.")
    else:
        print("OVERALL: NO-GO -- legal TracIn does not catch up @2% or degrades @5%.")
        print("Next: try end-to-end backbone or multi-checkpoint TracIn before abandoning.")


if __name__ == "__main__":
    sys.exit(main())
