from __future__ import annotations

import argparse

from robust_verify.config import load_config
from robust_verify.stage0 import run_stage0
from robust_verify.stage1 import run_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 0 followed by Stage 1.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_stage0(config)
    results = run_stage1(config)
    print(f"All stages completed with {len(results)} result rows.")
    print(f"Output: {config['project']['output_dir']}")


if __name__ == "__main__":
    main()
