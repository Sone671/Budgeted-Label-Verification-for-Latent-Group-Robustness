#!/usr/bin/env python
"""Compatibility entry point for the local Waterbirds gated tau ablation.

The gated seeds 40--59 ablation is a frozen-feature linear-head experiment;
the primary end-to-end result is already archived in the paper. Keep this
historical filename as a thin wrapper so existing launch commands cannot
silently start another full-network run.
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_waterbirds_gated_tau_ablation_frozen.py")),
        run_name="__main__",
    )
