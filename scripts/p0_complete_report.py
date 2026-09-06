#!/usr/bin/env python
"""Render a compact, auditable P0 complete-data Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _dataset_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--results entries must be DATASET=/path/to/results.csv")
        dataset, raw_path = value.split("=", 1)
        if dataset in paths:
            raise ValueError(f"duplicate dataset: {dataset}")
        paths[dataset] = Path(raw_path)
    return paths


def _coverage_table(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for dataset, path in paths.items():
        frame = pd.read_csv(path)
        artifact_columns = {"proxy_before", "proxy_after", "retrain_checkpoint", "query_artifact"}
        root = path.parent.parent
        artifact_rows = 0
        if artifact_columns.issubset(frame.columns):
            for _, row in frame.iterrows():
                checkpoint = root / str(row["retrain_checkpoint"])
                query_artifact = root / str(row["query_artifact"])
                artifact_rows += int(checkpoint.exists() and query_artifact.exists())
        rows.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "seeds": ", ".join(map(str, sorted(frame["seed"].astype(int).unique()))),
                "noise_settings": ", ".join(sorted(frame["noise_name"].unique())),
                "budgets": ", ".join(f"{value:.1%}" for value in sorted(frame["budget_fraction"].unique())),
                "methods": frame["method"].nunique(),
                "artifact_complete_rows": f"{artifact_rows}/{len(frame)}",
                "run_manifest": (root / "stage1" / "run_manifest.json").exists(),
                "artifact_manifest": (root / "stage1" / "artifact_manifest.json").exists(),
            }
        )
    return pd.DataFrame(rows)


def _best_methods(summary: pd.DataFrame) -> pd.DataFrame:
    legal = summary[~summary["method"].str.startswith("oracle_")].copy()
    if legal.empty:
        return legal
    keys = ["dataset", "noise_name", "budget_fraction"]
    indices = legal.groupby(keys)["mean_delta_wga"].idxmax()
    columns = keys + [
        "method", "mean_delta_wga", "paired_bootstrap_ci_low",
        "paired_bootstrap_ci_high", "negative_seed_rate", "worst_seed_delta_wga",
    ]
    return legal.loc[indices, columns].sort_values(keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final P0 complete-data report.")
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _dataset_paths(args.results)
    coverage = _coverage_table(paths)
    statistics_dir = Path(args.statistics_dir)
    summary = pd.read_csv(statistics_dir / "summary.csv")
    best = _best_methods(summary)
    acceptance = json.loads(Path(args.acceptance_report).read_text(encoding="utf-8"))

    lines = [
        "# P0 complete-data report",
        "",
        f"Overall acceptance: **{'PASS' if acceptance.get('passed') else 'REJECT'}**.",
        "",
        "## Raw-data coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Best legal method by predeclared condition",
        "",
        best.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Acceptance checks",
        "",
    ]
    skipped = [dataset for dataset, outcome in acceptance.get("runs", {}).items() if outcome.get("skipped")]
    if skipped:
        lines.extend([
            "## Conditional exclusions",
            "",
            "The following dataset(s) were excluded under the predeclared conditional protocol: " + ", ".join(skipped) + ".",
            "",
        ])
    for dataset, outcome in acceptance.get("runs", {}).items():
        lines.append(f"### {dataset}: {'PASS' if outcome.get('passed') else 'REJECT'}")
        lines.append("")
        for check in outcome.get("checks", []):
            marker = "x" if check.get("passed") else " "
            lines.append(f"- [{marker}] {check.get('check')}: {check.get('detail')}")
        lines.append("")
    lines.extend(
        [
            "## Generated source tables",
            "",
            "- `per_seed_raw.csv`: unaggregated results; use for all traceability checks.",
            "- `summary.csv`: seed-level mean, paired-bootstrap CI, negative rate, and worst seed.",
            "- `paired_vs_reference.csv`, `paired_vs_oracle.csv`: paired comparisons.",
            "- `rank_correlations.csv`, `direction_reversals.csv`, `seed_regret.csv`, and `query_overlap.csv`: mismatch diagnostics.",
            "",
        ]
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
