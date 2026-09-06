"""Summarize end-to-end full-retraining results with seed-paired effects."""

from __future__ import annotations

import argparse

from robust_verify.e2e_analysis import load_end_to_end_results, write_end_to_end_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="results.csv (repeatable)")
    parser.add_argument("--dataset-names", default=None, help="comma-separated names for legacy CSVs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-method", default="loss")
    parser.add_argument("--methods", default=None, help="comma-separated methods to compare")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    names = args.dataset_names.split(",") if args.dataset_names else None
    methods = args.methods.split(",") if args.methods else None
    results = load_end_to_end_results(args.input, dataset_names=names)
    paths = write_end_to_end_summary(
        results,
        args.output_dir,
        reference_method=args.reference_method,
        methods=methods,
        bootstrap_replicates=args.bootstrap_replicates,
        rng_seed=args.seed,
    )
    print(f"Wrote paired summary to {paths['summary']}")


if __name__ == "__main__":
    main()
