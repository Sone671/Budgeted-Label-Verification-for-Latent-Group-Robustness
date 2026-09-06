from __future__ import annotations

import argparse

from robust_verify.config import load_config
from robust_verify.stage0 import run_stage0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 0 data and feature preparation.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_stage0(config)
    print(f"Stage 0 completed: {config['project']['output_dir']}")


if __name__ == "__main__":
    main()
