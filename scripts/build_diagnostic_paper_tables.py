from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
T_CRIT_95_DF9 = 2.2621571628540993


RUNS = {
    "waterbirds": ROOT
    / "outputs"
    / "p0_wave_s1"
    / "waterbirds"
    / "stage1"
    / "results.csv",
    "celeba": ROOT
    / "outputs"
    / "p0_wave_s1"
    / "celeba"
    / "stage1"
    / "results.csv",
    "civilcomments": ROOT
    / "outputs"
    / "ablation_civilcomments_10seeds"
    / "stage1"
    / "results.csv",
}


PRIVATE_MANIFEST_DIRS = {
    "waterbirds": ROOT / "outputs" / "p0_30epoch_waterbirds_10seed" / "manifests",
    "celeba": ROOT / "outputs" / "p0_30epoch_celeba_10seed" / "manifests",
}


MAIN_SPECS = [
    ("waterbirds", "uniform20", 0.02, "noise_score"),
    ("waterbirds", "minority_high_40", 0.02, "noise_score_cpba_only"),
    ("celeba", "uniform20", 0.02, "noise_score"),
    ("celeba", "minority_high_40", 0.02, "noise_score"),
    ("civilcomments", "uniform20", 0.02, "noise_score_cpba_only"),
    ("civilcomments", "minority_high_40", 0.02, "noise_score_cpba_only"),
]


MECHANISM_SPECS = [
    ("waterbirds", "minority_high_40", 0.02, "loss"),
    ("waterbirds", "minority_high_40", 0.02, "noise_score_cpba_only"),
    ("celeba", "uniform20", 0.02, "loss"),
    ("celeba", "uniform20", 0.02, "noise_score"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select(
    rows: list[dict[str, str]], noise: str, budget: float, method: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["noise_name"] == noise
        and math.isclose(float(row["budget_fraction"]), budget)
        and row["method"] == method
    ]
    selected.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in selected] != list(range(10)):
        raise ValueError(
            f"Expected seeds 0--9 for {noise}/{budget}/{method}; "
            f"got {[row['seed'] for row in selected]}"
        )
    return selected


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half_width = T_CRIT_95_DF9 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def baseline_weak_group(row: dict[str, str]) -> int:
    return min(
        range(4), key=lambda group: float(row[f"baseline_group_{group}_accuracy"])
    )


def query_group_composition(
    dataset: str, noise: str, row: dict[str, str]
) -> tuple[int, int, int]:
    seed = int(row["seed"])
    manifest_path = (
        PRIVATE_MANIFEST_DIRS[dataset]
        / f"{noise}_seed{seed}_train_private.csv"
    )
    manifest_rows = read_rows(manifest_path)
    manifest = {int(item["sample_id"]): item for item in manifest_rows}

    query_path = RUNS[dataset].parents[1] / row["query_artifact"]
    query = json.loads(query_path.read_text(encoding="utf-8"))
    weak_group = baseline_weak_group(row)
    selected = [manifest[int(sample_id)] for sample_id in query["sample_ids"]]
    weak_queries = sum(int(item["group"]) == weak_group for item in selected)
    weak_corrections = sum(
        int(item["group"]) == weak_group and int(item["is_noisy"]) == 1
        for item in selected
    )
    other_corrections = sum(
        int(item["group"]) != weak_group and int(item["is_noisy"]) == 1
        for item in selected
    )
    return weak_queries, weak_corrections, other_corrections


def build_main_table(cache: dict[str, list[dict[str, str]]]) -> None:
    output: list[dict[str, object]] = []
    for dataset, noise, budget, method in MAIN_SPECS:
        rows = cache[dataset]
        candidate = select(rows, noise, budget, method)
        loss = select(rows, noise, budget, "loss")
        absolute = [float(row["delta_wga"]) for row in candidate]
        relative = [
            float(row["delta_wga"]) - float(loss_row["delta_wga"])
            for row, loss_row in zip(candidate, loss, strict=True)
        ]
        absolute_mean, absolute_low, absolute_high = mean_ci(absolute)
        relative_mean, relative_low, relative_high = mean_ci(relative)
        output.append(
            {
                "dataset": dataset,
                "noise": noise,
                "budget": budget,
                "method": method,
                "delta_wga_vs_no_correction": absolute_mean,
                "delta_wga_vs_no_correction_ci_low": absolute_low,
                "delta_wga_vs_no_correction_ci_high": absolute_high,
                "delta_wga_vs_loss": relative_mean,
                "delta_wga_vs_loss_ci_low": relative_low,
                "delta_wga_vs_loss_ci_high": relative_high,
                "negative_seeds": sum(value < 0 for value in absolute),
                "mean_corrections": statistics.fmean(
                    float(row["num_corrected"]) for row in candidate
                ),
                "mean_loss_corrections": statistics.fmean(
                    float(row["num_corrected"]) for row in loss
                ),
            }
        )

    write_csv(TABLE_DIR / "table_diagnostic_main_absolute.csv", output)


def build_mechanism_table(cache: dict[str, list[dict[str, str]]]) -> None:
    output: list[dict[str, object]] = []
    for dataset, noise, budget, method in MECHANISM_SPECS:
        selected = select(cache[dataset], noise, budget, method)
        compositions = [
            query_group_composition(dataset, noise, row) for row in selected
        ]
        weak_queries = [composition[0] for composition in compositions]
        weak_corrections = [composition[1] for composition in compositions]
        other_corrections = [composition[2] for composition in compositions]
        weak_accuracy_change = []
        for row in selected:
            group = baseline_weak_group(row)
            weak_accuracy_change.append(
                float(row[f"group_{group}_accuracy"])
                - float(row[f"baseline_group_{group}_accuracy"])
            )
        output.append(
            {
                "dataset": dataset,
                "noise": noise,
                "budget": budget,
                "method": method,
                "mean_weak_group_queries": statistics.fmean(weak_queries),
                "mean_weak_group_corrections": statistics.fmean(weak_corrections),
                "mean_other_corrections": statistics.fmean(other_corrections),
                "mean_weak_group_accuracy_change": statistics.fmean(
                    weak_accuracy_change
                ),
                "mean_wga_change": statistics.fmean(
                    float(row["delta_wga"]) for row in selected
                ),
            }
        )

    write_csv(TABLE_DIR / "table_diagnostic_group_mechanism.csv", output)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cache = {dataset: read_rows(path) for dataset, path in RUNS.items()}
    build_main_table(cache)
    build_mechanism_table(cache)


if __name__ == "__main__":
    main()
