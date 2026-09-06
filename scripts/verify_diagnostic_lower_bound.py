#!/usr/bin/env python
"""Enumerate the strengthened one-action lower-bound instances."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from robust_verify.diagnostic_theory import (
    OneActionInstance,
    enumerate_instance,
    same_xy_marginal,
    verify_logistic_embedding,
)


def _serialize(value):
    if isinstance(value, Fraction):
        return {"fraction": f"{value.numerator}/{value.denominator}", "float": float(value)}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/theory/diagnostic_one_action_lower_bound.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances = [
        OneActionInstance(budget=2, anchor_mass=40),
        OneActionInstance(budget=3, anchor_mass=90),
        OneActionInstance(budget=4, anchor_mass=160),
    ]
    records = []
    for instance in instances:
        record = enumerate_instance(instance)
        if record["max_two_world_utility_sum"] != 1:
            raise RuntimeError("The per-query two-world utility inequality failed")
        if record["balanced_randomized_worst_regret"] != instance.minimax_regret:
            raise RuntimeError("The balanced randomized strategy did not attain the lower bound")
        if not same_xy_marginal(instance):
            raise RuntimeError("The two worlds do not preserve the complete (X,Y) marginal")
        query = instance.balanced_strategy()[0][1]
        if not verify_logistic_embedding(instance, query):
            raise RuntimeError("The orthogonal logistic embedding failed")
        if not verify_logistic_embedding(instance, query, perturbation=1e-3):
            raise RuntimeError("The dense perturbed logistic embedding failed")
        record["logistic_embedding_verified"] = True
        record["dense_perturbation_verified"] = 1e-3
        records.append(record)

    payload = {
        "theorem": "one-action arbitrary-budget observable-equivalence lower bound",
        "same_xy_marginal": True,
        "instances": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(payload), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Diagnostic one-action lower-bound machine check",
        "",
        "Every enumerated query satisfies `U0(Q) + U1(Q) <= 1`; the balanced randomized policy has expected utility 1/2 in both worlds and attains the theorem bound.",
        "",
        "| B | M | Queries | Oracle value | Minimax regret |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['budget']} | {record['anchor_mass']} | "
            f"{record['num_deterministic_queries']} | "
            f"{float(record['oracle_value']):.6f} | "
            f"{float(record['theorem_minimax_regret']):.6f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
