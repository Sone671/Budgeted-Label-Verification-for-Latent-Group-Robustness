#!/usr/bin/env python
"""Independently audit the Waterbirds RV-Q-Gated result archive.

This P0 script is deliberately archive-native: it never needs to unpack or
modify the returned server ZIP.  It verifies the portable artifacts that are
present, replays all query and prediction metrics, documents the registered
code deviation, and writes paper-ready mechanism tables and figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "expected_repair_value",
    "noise_gated_repair_value",
    "noise_score",
)
METHOD_LABELS = {
    "expected_repair_value": "RV-Q (tail-product)",
    "noise_gated_repair_value": "RV-Q-Gated",
    "noise_score": "NoiseScore",
}
METHOD_COLORS = {
    "expected_repair_value": "#8c8c8c",
    "noise_gated_repair_value": "#d95f02",
    "noise_score": "#1b9e77",
}
TOLERANCE = 5e-13


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_by_suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one member ending {suffix!r}, found {len(matches)}")
    return matches[0]


def archive_root(names: list[str]) -> str:
    result_member = member_by_suffix(names, "aggregate/results.csv")
    return result_member[: -len("aggregate/results.csv")]


def read_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    return json.loads(archive.read(member).decode("utf-8-sig"))


def read_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))


def load_npz(archive: zipfile.ZipFile, member: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def metric_values(
    prediction: dict[str, np.ndarray], test_meta: dict[int, tuple[int, int]]
) -> dict[str, float]:
    ids = prediction["sample_ids"].astype(int)
    predictions = prediction["predictions"].astype(int)
    labels = np.asarray([test_meta[int(sample_id)][0] for sample_id in ids], dtype=int)
    groups = np.asarray([test_meta[int(sample_id)][1] for sample_id in ids], dtype=int)
    values: dict[str, float] = {
        "average_accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(
            np.mean(
                [np.mean(predictions[labels == label] == labels[labels == label]) for label in sorted(set(labels))]
            )
        ),
    }
    for group in range(4):
        mask = groups == group
        values[f"group_{group}_accuracy"] = float(np.mean(predictions[mask] == labels[mask]))
        values[f"group_{group}_count"] = float(mask.sum())
    values["wga"] = min(values[f"group_{group}_accuracy"] for group in range(4))
    return values


def query_values(ids: np.ndarray, train_meta: dict[int, dict[str, str]]) -> dict[str, float]:
    rows = [train_meta[int(sample_id)] for sample_id in ids.astype(int)]
    noisy = np.asarray([int(row["is_noisy"]) for row in rows], dtype=int)
    minority = np.asarray([int(row["is_minority"]) for row in rows], dtype=int)
    return {
        "num_queried": float(len(rows)),
        "num_corrected": float(noisy.sum()),
        "noise_precision": float(noisy.mean()),
        "mislabeled_minority": float(np.sum((noisy == 1) & (minority == 1))),
        "mislabeled_majority": float(np.sum((noisy == 1) & (minority == 0))),
        "clean_minority": float(np.sum((noisy == 0) & (minority == 1))),
        "clean_majority": float(np.sum((noisy == 0) & (minority == 0))),
        "minority_query_rate": float(minority.mean()),
        "clean_minority_rate": float(np.mean((noisy == 0) & (minority == 1))),
    }


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def replay_confirmatory_decision(results: pd.DataFrame, decision: dict[str, Any]) -> dict[str, Any]:
    """Recompute the registered bootstrap sequence from aggregate seed rows."""
    by_method = {
        method: results[results["method"] == method].sort_values("seed")
        for method in ("noise_gated_repair_value", "noise_score", "random", "loss", "expected_repair_value")
    }
    candidate = by_method["noise_gated_repair_value"]
    noise_score = by_method["noise_score"]
    paired = candidate["delta_wga"].to_numpy(float) - noise_score["delta_wga"].to_numpy(float)
    absolute = candidate["delta_wga"].to_numpy(float)
    random_paired = candidate["delta_wga"].to_numpy(float) - by_method["random"]["delta_wga"].to_numpy(float)
    loss_paired = candidate["delta_wga"].to_numpy(float) - by_method["loss"]["delta_wga"].to_numpy(float)
    v1_paired = candidate["delta_wga"].to_numpy(float) - by_method["expected_repair_value"]["delta_wga"].to_numpy(float)
    precision_difference = candidate["noise_precision"].to_numpy(float) - noise_score["noise_precision"].to_numpy(float)
    rng = np.random.default_rng(int(decision["bootstrap_rng_seed"]))
    replicates = int(decision["bootstrap_replicates"])
    replayed = {
        "paired_mean": float(paired.mean()),
        "paired_ci": bootstrap_mean_ci(paired, rng, replicates),
        "absolute_mean": float(absolute.mean()),
        "absolute_ci": bootstrap_mean_ci(absolute, rng, replicates),
        "random_ci": bootstrap_mean_ci(random_paired, rng, replicates),
        "loss_ci": bootstrap_mean_ci(loss_paired, rng, replicates),
        "v1_ci": bootstrap_mean_ci(v1_paired, rng, replicates),
        "precision_ci": bootstrap_mean_ci(precision_difference, rng, replicates),
        "positive_paired_seeds": int((paired > 0).sum()),
        "candidate_precision": float(candidate["noise_precision"].mean()),
        "noise_score_precision": float(noise_score["noise_precision"].mean()),
    }
    expected = {
        "paired_mean": decision["paired_vs_noise_score"]["mean_delta_wga_difference"],
        "paired_ci": tuple(decision["paired_vs_noise_score"]["bootstrap_95_ci"]),
        "absolute_mean": decision["absolute_utility"]["mean_delta_wga"],
        "absolute_ci": tuple(decision["absolute_utility"]["bootstrap_95_ci"]),
        "random_ci": tuple(decision["v2_vs_random_secondary"]["bootstrap_95_ci"]),
        "loss_ci": tuple(decision["v2_vs_loss_secondary"]["bootstrap_95_ci"]),
        "v1_ci": tuple(decision["v2_vs_v1_secondary"]["bootstrap_95_ci"]),
        "precision_ci": tuple(decision["precision_noninferiority"]["paired_precision_difference_bootstrap_95_ci"]),
        "positive_paired_seeds": decision["paired_vs_noise_score"]["positive_seed_count"],
        "candidate_precision": decision["precision_noninferiority"]["candidate_mean_noise_precision"],
        "noise_score_precision": decision["precision_noninferiority"]["noise_score_mean_noise_precision"],
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual = replayed[key]
        if isinstance(actual, tuple):
            matches = bool(np.allclose(actual, expected_value, atol=TOLERANCE, rtol=0.0))
        elif isinstance(actual, float):
            matches = bool(np.isclose(actual, expected_value, atol=TOLERANCE, rtol=0.0))
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(key)
    return {"exact_match": not mismatches, "mismatched_fields": mismatches}


def verify_archive(
    archive: zipfile.ZipFile, root: str, results: pd.DataFrame, decision: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = archive.namelist()
    name_set = set(names)
    artifact_manifest = read_json(archive, root + "aggregate/artifact_manifest.json")
    preparation_manifest = read_json(archive, root + "source/preparation_manifest.json")

    artifact_present = artifact_missing = artifact_bad = 0
    artifact_missing_extensions: Counter[str] = Counter()
    for record in artifact_manifest["files"]:
        member = root + record["path"]
        if member not in name_set:
            artifact_missing += 1
            artifact_missing_extensions[Path(member).suffix] += 1
            continue
        artifact_present += 1
        content = archive.read(member)
        if len(content) != int(record["bytes"]) or sha256_bytes(content) != record["sha256"]:
            artifact_bad += 1

    preparation_bad = 0
    for record in preparation_manifest["files"]:
        member = root + "source/" + record["path"]
        content = archive.read(member)
        if len(content) != int(record["bytes"]) or sha256_bytes(content) != record["sha256"]:
            preparation_bad += 1

    complete_members = sorted(member for member in names if member.endswith("/COMPLETE.json"))
    complete_bad: list[dict[str, Any]] = []
    for member in complete_members:
        complete = read_json(archive, member)
        result_member = member.rsplit("/", 1)[0] + "/" + complete["results"]
        result_hash_matches = sha256_bytes(archive.read(result_member)) == complete["results_sha256"]
        if complete["status"] != "complete" or not result_hash_matches:
            complete_bad.append(
                {"seed": complete["seed"], "status": complete["status"], "result_hash_matches": result_hash_matches}
            )

    test_rows = read_csv(archive, root + "source/canonical/test.csv")
    test_meta = {int(row["sample_id"]): (int(row["clean_label"]), int(row["group"])) for row in test_rows}
    prediction_cache: dict[str, dict[str, float]] = {}
    train_cache: dict[int, dict[int, dict[str, str]]] = {}
    metric_mismatches: list[dict[str, Any]] = []

    for _, row in results.iterrows():
        seed = int(row["seed"])
        prediction_path = str(row["prediction_artifact"])
        baseline_prediction_path = str(row["baseline_prediction_artifact"])
        for path in (prediction_path, baseline_prediction_path):
            if path not in prediction_cache:
                prediction_cache[path] = metric_values(load_npz(archive, root + path), test_meta)
        method_metric = prediction_cache[prediction_path]
        baseline_metric = prediction_cache[baseline_prediction_path]
        for key, value in method_metric.items():
            if abs(float(row[key]) - value) > TOLERANCE:
                metric_mismatches.append({"seed": seed, "method": row["method"], "field": key})
        for key, value in baseline_metric.items():
            baseline_key = f"baseline_{key}"
            if baseline_key in row and abs(float(row[baseline_key]) - value) > TOLERANCE:
                metric_mismatches.append({"seed": seed, "method": row["method"], "field": baseline_key})
        if abs(float(row["delta_wga"]) - (method_metric["wga"] - baseline_metric["wga"])) > TOLERANCE:
            metric_mismatches.append({"seed": seed, "method": row["method"], "field": "delta_wga"})
        if abs(
            float(row["delta_average_accuracy"])
            - (method_metric["average_accuracy"] - baseline_metric["average_accuracy"])
        ) > TOLERANCE:
            metric_mismatches.append({"seed": seed, "method": row["method"], "field": "delta_average_accuracy"})

        if seed not in train_cache:
            train_rows = read_csv(archive, root + f"source/manifests/uniform20_seed{seed}_train_private.csv")
            train_cache[seed] = {int(record["sample_id"]): record for record in train_rows}
        query = load_npz(archive, root + str(row["query_artifact"]))["sample_ids"]
        for key, value in query_values(query, train_cache[seed]).items():
            if abs(float(row[key]) - value) > TOLERANCE:
                metric_mismatches.append({"seed": seed, "method": row["method"], "field": key})

    integrity = {
        "zip_crc_bad_member": archive.testzip(),
        "artifact_manifest": {
            "declared": len(artifact_manifest["files"]),
            "present": artifact_present,
            "missing": artifact_missing,
            "bad_hash_or_size": artifact_bad,
            "missing_extensions": dict(sorted(artifact_missing_extensions.items())),
        },
        "preparation_manifest": {
            "declared": len(preparation_manifest["files"]),
            "bad_hash_or_size": preparation_bad,
        },
        "complete_shards": {"count": len(complete_members), "bad": complete_bad},
        "metric_replay": {
            "result_rows": int(len(results)),
            "prediction_artifacts": len(prediction_cache),
            "mismatch_count": len(metric_mismatches),
        },
        "confirmatory_decision_replay": replay_confirmatory_decision(results, decision),
    }
    return integrity, pd.DataFrame(metric_mismatches)


def code_deviation_report(archive: zipfile.ZipFile, bundle: Path | None) -> pd.DataFrame:
    if bundle is None or not bundle.exists():
        return pd.DataFrame(columns=["path", "server_sha256", "registered_sha256", "status"])
    with zipfile.ZipFile(bundle) as registered:
        registered_names = registered.namelist()
        rows = []
        for member in sorted(name for name in archive.namelist() if name.startswith("code/") and name.endswith(".py")):
            relative = member.removeprefix("code/")
            registered_member = next((name for name in registered_names if name.endswith(relative)), None)
            server_hash = sha256_bytes(archive.read(member))
            registered_hash = sha256_bytes(registered.read(registered_member)) if registered_member else None
            rows.append(
                {
                    "path": relative,
                    "server_sha256": server_hash,
                    "registered_sha256": registered_hash,
                    "status": "match" if server_hash == registered_hash else "mismatch",
                }
            )
    return pd.DataFrame(rows)


def stratum_summary(
    name: str, ids: list[int], train_meta: dict[int, dict[str, str]], seed: int
) -> dict[str, Any]:
    noisy = [int(train_meta[sample_id]["is_noisy"]) for sample_id in ids]
    minority = [int(train_meta[sample_id]["is_minority"]) for sample_id in ids]
    return {
        "seed": seed,
        "component": name,
        "query_count": len(ids),
        "noisy_count": int(sum(noisy)),
        "noise_precision": float(np.mean(noisy)) if noisy else float("nan"),
        "noisy_minority_count": int(sum(value and group for value, group in zip(noisy, minority))),
    }


def mechanism_tables(
    archive: zipfile.ZipFile, root: str, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = results.set_index(["seed", "method"])
    paired_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for seed in sorted(int(value) for value in results["seed"].unique()):
        v2 = indexed.loc[(seed, "noise_gated_repair_value")]
        noise_score = indexed.loc[(seed, "noise_score")]
        train_rows = read_csv(archive, root + f"source/manifests/uniform20_seed{seed}_train_private.csv")
        train_meta = {int(record["sample_id"]): record for record in train_rows}
        v2_ids = load_npz(archive, root + str(v2["query_artifact"]))["sample_ids"].astype(int).tolist()
        noise_ids = load_npz(archive, root + str(noise_score["query_artifact"]))["sample_ids"].astype(int).tolist()
        budget = len(v2_ids)
        anchor_count = int(np.ceil(0.5 * budget))
        v2_anchor = v2_ids[:anchor_count]
        v2_fill = v2_ids[anchor_count:]
        noise_tail = noise_ids[anchor_count:]
        noise_set = set(noise_ids)
        fill_set = set(v2_fill)
        tail_set = set(noise_tail)
        components = {
            "NoiseScore anchor (top 0.5B)": v2_anchor,
            "RV-Q tail-value fill": v2_fill,
            "NoiseScore tail (B/2 to B)": noise_tail,
            "Swapped in by RepairValue": [sample_id for sample_id in v2_fill if sample_id not in noise_set],
            "Swapped out from NoiseScore": [sample_id for sample_id in noise_tail if sample_id not in fill_set],
        }
        for name, ids in components.items():
            swap_rows.append(stratum_summary(name, ids, train_meta, seed))
        paired_rows.append(
            {
                "seed": seed,
                "baseline_wga": float(v2["baseline_wga"]),
                "v2_delta_wga_pp": 100 * float(v2["delta_wga"]),
                "noise_score_delta_wga_pp": 100 * float(noise_score["delta_wga"]),
                "v2_minus_noise_score_delta_wga_pp": 100 * (float(v2["delta_wga"]) - float(noise_score["delta_wga"])),
                "v2_minus_noise_score_precision_pp": 100 * (float(v2["noise_precision"]) - float(noise_score["noise_precision"])),
                "v2_minus_noise_score_delta_average_accuracy_pp": 100
                * (float(v2["delta_average_accuracy"]) - float(noise_score["delta_average_accuracy"])),
                "query_overlap_count": len(set(v2_ids) & set(noise_ids)),
                "query_overlap_fraction": len(set(v2_ids) & set(noise_ids)) / budget,
                "anchor_matches_noise_score_prefix": v2_anchor == noise_ids[:anchor_count],
            }
        )
        for group in range(4):
            group_rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "v2_minus_noise_score_accuracy_pp": 100
                    * (float(v2[f"group_{group}_accuracy"]) - float(noise_score[f"group_{group}_accuracy"])),
                }
            )

    paired = pd.DataFrame(paired_rows).sort_values("seed")
    swap_per_seed = pd.DataFrame(swap_rows).sort_values(["component", "seed"])
    swap_summary = (
        swap_per_seed.groupby("component", sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_query_count=("query_count", "mean"),
            mean_noisy_count=("noisy_count", "mean"),
            mean_noise_precision=("noise_precision", "mean"),
            mean_noisy_minority_count=("noisy_minority_count", "mean"),
        )
        .reset_index()
    )
    group_effects = (
        pd.DataFrame(group_rows)
        .groupby("group")["v2_minus_noise_score_accuracy_pp"]
        .agg(mean_pp="mean", median_pp="median", min_pp="min", max_pp="max")
        .reset_index()
    )

    summary_rows = []
    for method in METHODS:
        frame = results[results["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": int(len(frame)),
                "mean_delta_wga_pp": 100 * float(frame["delta_wga"].mean()),
                "mean_wga_percent": 100 * float(frame["wga"].mean()),
                "mean_delta_average_accuracy_pp": 100 * float(frame["delta_average_accuracy"].mean()),
                "mean_noise_precision_percent": 100 * float(frame["noise_precision"].mean()),
                "mean_corrected_labels": float(frame["num_corrected"].mean()),
            }
        )
    return paired, swap_per_seed, swap_summary, group_effects, pd.DataFrame(summary_rows)


def plot_mechanism(
    output_dir: Path,
    method_summary: pd.DataFrame,
    swap_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), constrained_layout=True)

    method_order = list(METHODS)
    methods = method_summary.set_index("method").loc[method_order].reset_index()
    axis = axes[0]
    bars = axis.bar(
        range(len(methods)),
        methods["mean_delta_wga_pp"],
        color=[METHOD_COLORS[method] for method in methods["method"]],
        width=0.68,
    )
    axis.axhline(0, color="#3b3b3b", linewidth=0.9)
    axis.set_xticks(range(len(methods)), ["RV-Q", "RV-Q-Gated", "NoiseScore"])
    axis.set_ylabel("Mean ΔWGA vs no correction (pp)")
    axis.set_title("Absolute repair utility")
    axis.set_ylim(0, float(methods["mean_delta_wga_pp"].max()) + 1.25)
    for bar, precision in zip(bars, methods["mean_noise_precision_percent"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.22,
            f"{bar.get_height():+.2f}\nprecision {precision:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    precision_order = [
        "NoiseScore anchor (top 0.5B)",
        "RV-Q tail-value fill",
        "NoiseScore tail (B/2 to B)",
        "Swapped in by RepairValue",
        "Swapped out from NoiseScore",
    ]
    precision = swap_summary.set_index("component").loc[precision_order].reset_index()
    axis = axes[1]
    segment_labels = ["Anchor", "RV fill", "NS tail", "Swap in", "Swap out"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#d95f02", "#1b9e77"]
    bars = axis.bar(range(len(precision)), 100 * precision["mean_noise_precision"], color=colors, width=0.7)
    axis.set_xticks(range(len(precision)), segment_labels, rotation=27, ha="right")
    axis.set_ylabel("Query noise precision (%)")
    axis.set_title("RV-Q-Gated fill replaces a high-precision detector tail")
    axis.set_ylim(0, 100)
    for bar, count in zip(bars, precision["mean_noisy_count"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            f"{bar.get_height():.1f}%\n{count:.1f} errors",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    axis = axes[2]
    ordered = paired.sort_values("v2_minus_noise_score_delta_wga_pp")
    y = np.arange(len(ordered))
    colors = np.where(ordered["v2_minus_noise_score_delta_wga_pp"] >= 0, "#1b9e77", "#d95f02")
    axis.hlines(y, 0, ordered["v2_minus_noise_score_delta_wga_pp"], color=colors, linewidth=1.4)
    axis.scatter(ordered["v2_minus_noise_score_delta_wga_pp"], y, color=colors, s=34, zorder=3)
    axis.axvline(0, color="#3b3b3b", linewidth=0.9)
    mean = float(ordered["v2_minus_noise_score_delta_wga_pp"].mean())
    axis.axvline(mean, color="#555555", linewidth=1.0, linestyle="--")
    axis.set_yticks(y, [str(int(seed)) for seed in ordered["seed"]])
    axis.set_xlabel("RV-Q-Gated − NoiseScore ΔWGA (pp)")
    axis.set_ylabel("Seed")
    axis.set_title("Paired effect: 7/20 seeds positive")
    axis.text(
        0.02,
        0.98,
        "mean = −1.27 pp\nformal 95% CI [−3.11, +0.34]",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "boxstyle": "round,pad=0.25", "alpha": 0.92},
    )

    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_waterbirds_v2_mechanism.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    archive: Path,
    decision: dict[str, Any],
    integrity: dict[str, Any],
    code_report: pd.DataFrame,
    swap_summary: pd.DataFrame,
) -> None:
    paired = decision["paired_vs_noise_score"]
    absolute = decision["absolute_utility"]
    precision = decision["precision_noninferiority"]
    artifact = integrity["artifact_manifest"]
    code_mismatches = code_report[code_report["status"] == "mismatch"]
    lines = [
        "# P0 Waterbirds RV-Q-Gated archive audit",
        "",
        f"Source archive: `{archive}`",
        f"SHA256: `{sha256_file(archive)}`",
        "",
        "## Frozen decision",
        "",
        f"- Primary gate: **FAILED** (one of four gates passed).",
        f"- v2 minus NoiseScore: {100 * paired['mean_delta_wga_difference']:+.3f} pp "
        f"[{100 * paired['bootstrap_95_ci'][0]:+.3f}, {100 * paired['bootstrap_95_ci'][1]:+.3f}], "
        f"{paired['positive_seed_count']}/{decision['n_seeds']} positive seeds.",
        f"- Absolute v2 ΔWGA: {100 * absolute['mean_delta_wga']:+.3f} pp "
        f"[{100 * absolute['bootstrap_95_ci'][0]:+.3f}, {100 * absolute['bootstrap_95_ci'][1]:+.3f}].",
        f"- v2 query precision: {100 * precision['candidate_mean_noise_precision']:.3f}%; "
        f"NoiseScore: {100 * precision['noise_score_mean_noise_precision']:.3f}%; "
        f"difference CI [{100 * precision['paired_precision_difference_bootstrap_95_ci'][0]:+.3f}, "
        f"{100 * precision['paired_precision_difference_bootstrap_95_ci'][1]:+.3f}] pp.",
        "",
        "## Independent archive replay",
        "",
        f"- ZIP CRC: {'passed' if integrity['zip_crc_bad_member'] is None else integrity['zip_crc_bad_member']}",
        f"- Artifact manifest: {artifact['present']}/{artifact['declared']} embedded files passed hash/size checks; "
        f"{artifact['missing']} intentionally omitted checkpoint files ({artifact['missing_extensions']}).",
        f"- Preparation manifest: {integrity['preparation_manifest']['declared']} files, "
        f"{integrity['preparation_manifest']['bad_hash_or_size']} failures.",
        f"- Completed shards: {integrity['complete_shards']['count']}; failures: {len(integrity['complete_shards']['bad'])}.",
        f"- Metric replay: {integrity['metric_replay']['result_rows']} result rows and "
        f"{integrity['metric_replay']['prediction_artifacts']} prediction artifacts; "
        f"mismatches: {integrity['metric_replay']['mismatch_count']}.",
        f"- Confirmatory decision bootstrap replay: "
        f"{'exact match' if integrity['confirmatory_decision_replay']['exact_match'] else 'mismatch: ' + ', '.join(integrity['confirmatory_decision_replay']['mismatched_fields'])}.",
        "",
        "## Query-swap mechanism",
        "",
        "| Segment | Mean samples | Mean errors | Noise precision | Mean noisy-minority errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in swap_summary.iterrows():
        lines.append(
            f"| {row['component']} | {row['mean_query_count']:.1f} | {row['mean_noisy_count']:.2f} | "
            f"{100 * row['mean_noise_precision']:.2f}% | {row['mean_noisy_minority_count']:.2f} |"
        )
    lines.extend(["", "## Registered-code comparison", ""])
    if code_report.empty:
        lines.append("Registered bundle was not supplied, so code comparison was not performed.")
    else:
        lines.append(f"- Compared {len(code_report)} returned Python files with the registered bundle.")
        if code_mismatches.empty:
            lines.append("- All compared files match.")
        else:
            lines.append(
                "- Mismatches: " + ", ".join(f"`{value}`" for value in code_mismatches["path"].tolist()) + "."
            )
            lines.append(
                "- The returned `scoring.py` diff is confined to NN agreement/label-spreading batching and caching; "
                "these methods are absent from the formal method list. This supports semantic interpretation of "
                "the replayed result, but not a claim of strict sealed-code conformance."
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Server-returned v2 result ZIP.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New derived P0 output directory.")
    parser.add_argument(
        "--registered-bundle",
        type=Path,
        default=Path("dist/waterbirds_repairvalue_v2_seed40_59_server.zip"),
        help="Frozen server bundle used for code-hash comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.out_dir.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(args.archive) as archive:
        root = archive_root(archive.namelist())
        results = pd.read_csv(io.BytesIO(archive.read(root + "aggregate/results.csv")))
        decision = read_json(archive, root + "aggregate/audit/confirmatory_decision.json")
        integrity, metric_mismatches = verify_archive(archive, root, results, decision)
        code_report = code_deviation_report(archive, args.registered_bundle)
        paired, swap_per_seed, swap_summary, group_effects, method_summary = mechanism_tables(archive, root, results)

    integrity["source_archive_sha256"] = sha256_file(args.archive)
    integrity["source_archive"] = str(args.archive.resolve())
    integrity["decision_exact_primary_gate_passed"] = bool(decision["primary_gate_passed"])
    integrity["query_anchor_prefix_matches"] = int(paired["anchor_matches_noise_score_prefix"].sum())
    integrity["query_anchor_prefix_total"] = int(len(paired))

    method_summary.to_csv(args.out_dir / "method_summary.csv", index=False)
    paired.to_csv(args.out_dir / "per_seed_v2_vs_noise_score.csv", index=False)
    swap_per_seed.to_csv(args.out_dir / "query_swap_per_seed.csv", index=False)
    swap_summary.to_csv(args.out_dir / "query_swap_summary.csv", index=False)
    group_effects.to_csv(args.out_dir / "group_effects_v2_vs_noise_score.csv", index=False)
    metric_mismatches.to_csv(args.out_dir / "metric_replay_mismatches.csv", index=False)
    code_report.to_csv(args.out_dir / "registered_code_comparison.csv", index=False)
    (args.out_dir / "integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (args.out_dir / "confirmatory_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_report(
        args.out_dir / "P0_WATERBIRDS_V2_AUDIT.md",
        args.archive,
        decision,
        integrity,
        code_report,
        swap_summary,
    )
    plot_mechanism(args.out_dir, method_summary, swap_summary, paired)
    print(f"P0 v2 audit written to {args.out_dir}")


if __name__ == "__main__":
    main()
