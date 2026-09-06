#!/usr/bin/env python
"""Machine-check the finite two-action allocation-regret witness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.two_action_theory import verify_two_action_lower_bound


def main() -> None:
    result = verify_two_action_lower_bound()
    assert result["oracle_world0"] == 1.0
    assert result["oracle_world1"] == 1.0
    assert result["maximum_deterministic_utility_sum"] <= 1.0
    assert result["tight_randomized_worst_regret"] == 0.5
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

