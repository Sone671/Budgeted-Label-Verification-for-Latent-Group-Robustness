#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robust_verify.diagnostics import build_diagnostics_table, parse_budgets


DEFAULT_METHODS = [
    "loss",
    "oracle_noise_loss_tiebreak",
    "oracle_noise_random_tiebreak",
    "oracle_noise_group_balanced",
]

DEFAULT_BUDGETS = ["0.5%", "1%", "2%", "5%", "10%"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oracle-noise selection diagnostics for robust verification."
    )
    parser.add_argument("--input", required=True, help="Input CSV with losses, group, and is_noisy.")
    parser.add_argument("--out", default="oracle_diagnostics.csv", help="Output CSV path.")
    parser.add_argument("--loss-col", default="loss", help="Loss column name.")
    parser.add_argument("--group-col", default="group", help="Group column name.")
    parser.add_argument("--noisy-col", default="is_noisy", help="Noisy-label indicator column.")
    parser.add_argument(
        "--minority-values",
        nargs="*",
        default=None,
        help="Group values treated as minority. If omitted, least frequent group is used.",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods to evaluate.")
    parser.add_argument("--budgets", nargs="+", default=DEFAULT_BUDGETS, help="Budgets, e.g. 1%% 0.02 96.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random-tiebreak oracle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    if args.loss_col not in df.columns:
        raise KeyError(
            f"missing loss column {args.loss_col!r}; available={list(df.columns)}"
        )

    losses = df[args.loss_col].to_numpy(dtype=float)
    budgets = parse_budgets(args.budgets, len(df))
    diagnostics = build_diagnostics_table(
        df,
        losses,
        methods=args.methods,
        budgets=budgets,
        group_col=args.group_col,
        noisy_col=args.noisy_col,
        minority_values=args.minority_values,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
