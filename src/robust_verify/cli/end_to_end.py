from __future__ import annotations

import argparse

from robust_verify.config import load_config
from robust_verify.end_to_end import run_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end image verification experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    args = parser.parse_args()
    config = load_config(args.config)
    results = run_end_to_end(config)
    print(f"End-to-end run completed with {len(results)} result rows.")


if __name__ == "__main__":
    main()
