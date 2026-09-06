#!/usr/bin/env python
"""Machine-check the finite-tree and capacity-batch three-world theorems."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_verify.three_action_theory import (
    verify_capacity_batch_three_world_extension,
    verify_finite_horizon_bellman_recovery,
    verify_per_label_batch_three_world_extension,
    verify_sharp_three_world_wga_template,
)


def main() -> None:
    sharp = verify_sharp_three_world_wga_template()
    expanded = verify_capacity_batch_three_world_extension()
    per_label = verify_per_label_batch_three_world_extension()
    bellman = verify_finite_horizon_bellman_recovery()
    assert sharp["tight_minimax_regret"] == "2/9"
    assert expanded["tight_minimax_regret"] == "1/5"
    assert expanded["same_test_xy_marginal"] is True
    assert per_label["tight_minimax_regret"] == "16/81"
    assert per_label["same_test_xy_marginal"] is True
    assert per_label["arbitrary_r_g_ordering"] is True
    assert bellman["trajectory_regret_bound"] == "2/5"
    print(
        json.dumps(
            {
                "sharp_finite_menu": sharp,
                "capacity_batch": expanded,
                "per_label_batch": per_label,
                "finite_horizon_bellman_recovery": bellman,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
