from __future__ import annotations

import argparse

from robust_verify.config import load_config
from robust_verify.stage1 import run_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 1 diagnostic experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    results = run_stage1(config)
    print(f"Stage 1 completed with {len(results)} result rows.")
    print(f"Results: {config['project']['output_dir']}/stage1/results.csv")


if __name__ == "__main__":
    main()
