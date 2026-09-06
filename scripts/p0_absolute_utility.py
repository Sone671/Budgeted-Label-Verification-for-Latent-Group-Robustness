#!/usr/bin/env python
"""Audit practical methods against no correction and Loss.

For each dataset/noise/budget/method condition this command emits two separate
claims:

* absolute utility: ``ΔWGA = WGA(method) - WGA(no-correction)``;
* relative utility: paired ``ΔWGA(method) - ΔWGA(Loss)`` over the same seeds.

Oracle rows are excluded by default.  The command is analysis-only: it never
changes raw result files or retrains a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robust_verify.absolute_utility import AbsoluteUtilityAudit, audit_absolute_utility, prepare_results


def _parse_result_spec(value: str) -> tuple[str | None, Path]:
    """Parse ``DATASET=path`` while also allowing a bare path."""
    if "=" in value:
        dataset, path = value.split("=", 1)
        if not dataset.strip() or not path.strip():
            raise ValueError("--results entries must be DATASET=PATH or PATH")
        return dataset.strip(), Path(path).expanduser()
    return None, Path(value).expanduser()


def _infer_dataset(path: Path, frame: pd.DataFrame) -> str:
    if "dataset" in frame.columns and frame["dataset"].notna().any():
        values = sorted(frame["dataset"].dropna().astype(str).unique().tolist())
        if len(values) == 1:
            return values[0]
    # A conventional path is <output>/<dataset>/stage1/results.csv.  The
    # explicit --results DATASET=PATH syntax remains preferred for audit logs.
    return path.parent.parent.name


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> list[str]:
    if frame.empty:
        return ["_None._"]
    view = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:+.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in record) + " |" for record in view.itertuples(index=False, name=None)]
    suffix = [f"\n_Showing {len(view)} of {len(frame)} rows; see the CSV for the full table._"] if len(view) < len(frame) else []
    return [header, divider, *rows, *suffix]


def _write_report(output_dir: Path, audit: AbsoluteUtilityAudit) -> None:
    absolute = audit.absolute_vs_no_correction
    paired = audit.paired_vs_loss
    absolute_harm = absolute[absolute.get("evidence", pd.Series(dtype=str)) == "absolute_harm_vs_no_correction"]
    relative_harm = paired[paired.get("evidence", pd.Series(dtype=str)) == "relative_harm_vs_loss"]
    lines = [
        "# Practical absolute-utility audit",
        "",
        "This report keeps the two comparisons separate:",
        "",
        "1. **Absolute utility vs no correction** is the reported `delta_wga` for each method.",
        "2. **Relative utility vs Loss** is a seed-paired difference of those absolute effects.",
        "",
        "Do not interpret a relative loss to `Loss` as absolute harm unless the first table also establishes negative `delta_wga`.",
        "",
        "## Audit metadata",
        "",
        "```json",
        json.dumps(audit.metadata, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Conditions with evidence of absolute harm vs no correction",
        "",
        *_markdown_table(
            absolute_harm,
            [
                "dataset", "noise_name", "budget_fraction", "method", "n_paired_seeds",
                "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high", "harmful_seed_count",
                "sign_test_two_sided_p", "evidence",
            ],
        ),
        "",
        "## Conditions with evidence of relative harm vs Loss",
        "",
        *_markdown_table(
            relative_harm,
            [
                "dataset", "noise_name", "budget_fraction", "method", "n_paired_seeds",
                "mean_method_delta_wga", "mean_loss_delta_wga", "mean_effect",
                "bootstrap_ci_low", "bootstrap_ci_high", "sign_test_two_sided_p", "evidence",
            ],
        ),
        "",
        "## Generated files",
        "",
        "- `absolute_vs_no_correction.csv`: condition-level absolute ΔWGA summaries.",
        "- `paired_vs_loss.csv`: condition-level paired ΔWGA differences against Loss.",
        "- `per_seed_absolute_vs_no_correction.csv`: unaggregated absolute effects.",
        "- `per_seed_vs_loss.csv`: unaggregated paired effects against Loss.",
        "- `coverage.csv`: pair availability and any missing seed diagnostics.",
        "- `audit_report.json`: parameters and baseline-integrity metadata.",
        "",
    ]
    (output_dir / "PRACTICAL_ABSOLUTE_UTILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help="Repeatable DATASET=path/to/results.csv input; a bare path is also accepted.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--loss-method", default="loss")
    parser.add_argument("--no-correction-method", default="no_correction")
    parser.add_argument(
        "--method",
        action="append",
        help="Optional practical candidate method; repeat to restrict the report.",
    )
    parser.add_argument("--exclude-method", action="append", default=[])
    parser.add_argument(
        "--effect-epsilon",
        type=float,
        default=0.0,
        help="Effect magnitude treated as a tie for direction counts and evidence labels.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    input_paths: list[str] = []
    for spec in args.results:
        dataset, path = _parse_result_spec(spec)
        path = path.resolve()
        frame = pd.read_csv(path)
        frames.append(prepare_results(frame, dataset or _infer_dataset(path, frame)))
        input_paths.append(str(path))
    results = pd.concat(frames, ignore_index=True)
    # Re-run validation after concatenation to reject accidental duplicate
    # dataset keys supplied through separate input files.
    results = prepare_results(results)
    audit = audit_absolute_utility(
        results,
        loss_method=args.loss_method,
        no_correction_method=args.no_correction_method,
        include_methods=args.method,
        exclude_methods=args.exclude_method,
        epsilon=args.effect_epsilon,
        n_bootstrap=args.bootstrap,
        rng_seed=args.rng_seed,
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.absolute_vs_no_correction.to_csv(output_dir / "absolute_vs_no_correction.csv", index=False)
    audit.paired_vs_loss.to_csv(output_dir / "paired_vs_loss.csv", index=False)
    audit.per_seed_absolute.to_csv(output_dir / "per_seed_absolute_vs_no_correction.csv", index=False)
    audit.per_seed_vs_loss.to_csv(output_dir / "per_seed_vs_loss.csv", index=False)
    audit.coverage.to_csv(output_dir / "coverage.csv", index=False)
    report = {**audit.metadata, "input_results": input_paths}
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, audit)
    print(f"Wrote practical absolute-utility audit to {output_dir}")


if __name__ == "__main__":
    main()
