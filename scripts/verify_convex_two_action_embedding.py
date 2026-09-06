#!/usr/bin/env python
"""Machine-check the strongly convex logistic realization of the witness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.convex_two_action import (
    verify_convex_logistic_embedding,
    verify_nonorthogonal_logistic_stability,
)


def main() -> None:
    result = verify_convex_logistic_embedding()
    result["nonorthogonal_stability"] = verify_nonorthogonal_logistic_stability()
    assert result["all_predictions_match_verified_labels"] is True
    assert result["all_wga_utilities_match_finite_witness"] is True
    assert result["nonorthogonal_stability"]["all_certified_predictions_preserved"] is True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
