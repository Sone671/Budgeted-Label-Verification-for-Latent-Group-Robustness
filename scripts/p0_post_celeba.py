#!/usr/bin/env python
"""Apply the frozen CelebA consistency decision after background training ends."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from p0_compare_results import compare


def _pid_exists(pid: int) -> bool:
    try:
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for CelebA, validate it, and decide CivilComments.")
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    candidate = root / "outputs/p0_wave_s1/celeba/stage1/results.csv"
    artifact = root / "outputs/p0_wave_s1/celeba/stage1/artifact_manifest.json"
    reference = root / "outputs/p0_30epoch_celeba_10seed/stage1/results.csv"
    control = root / "outputs/p0_control"
    control.mkdir(parents=True, exist_ok=True)
    decision_path = control / "celeba_consistency_decision.json"

    while not (candidate.exists() and artifact.exists()):
        if not _pid_exists(args.pid):
            decision_path.write_text(
                json.dumps({"status": "celeba_incomplete", "civilcomments_started": False}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise SystemExit("CelebA process ended before final artifacts were written.")
        time.sleep(args.poll_seconds)

    comparison = compare(reference, candidate)
    compare_path = control / "celeba_reference_comparison.json"
    compare_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    acceptance_path = control / "celeba_wave_s1_acceptance.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/p0_accept_server_wave.py",
            "--result", "waterbirds=outputs/p0_wave_s1/waterbirds/stage1/results.csv",
            "--result", "celeba=outputs/p0_wave_s1/celeba/stage1/results.csv",
            "--out", str(acceptance_path),
            "--allow-fail",
        ],
        cwd=root,
        check=True,
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance_passed = bool(acceptance["runs"]["celeba"]["passed"])
    status = "consistent" if comparison["passed"] and acceptance_passed else "inconsistent"
    decision = {
        "status": status,
        "comparison_report": str(compare_path),
        "acceptance_report": str(acceptance_path),
        "celeba_artifacts_accepted": acceptance_passed,
        "civilcomments_started": False,
    }
    if not comparison["passed"]:
        stage = root / "outputs/p0_wave_s1/civilcomments/stage1"
        stage.mkdir(parents=True, exist_ok=True)
        stdout = (stage / "training.log").open("a", encoding="utf-8")
        stderr = (stage / "training.err.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/p0_batch.py",
                "--dataset", "civilcomments",
                "--role", "confirmatory",
                "--stage", "stage1",
            ],
            cwd=root,
            stdout=stdout,
            stderr=stderr,
        )
        decision.update({"civilcomments_started": True, "civilcomments_pid": process.pid})
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
