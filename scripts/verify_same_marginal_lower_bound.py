#!/usr/bin/env python
"""Machine-check the same-(X,Y), binary-audit allocation lower bound."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.two_action_theory import verify_same_marginal_lower_bound


def main() -> None:
    result = verify_same_marginal_lower_bound()
    assert result["same_test_xy_marginal"] is True
    assert result["audit_cardinality"] == 2
    assert result["maximum_normalized_utility_sum"] <= 1.0
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
