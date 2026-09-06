from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from robust_verify.scoring import score_by_method


@dataclass(frozen=True)
class Budget:
    label: str
    fraction: float
    count: int


def parse_budget(value: str, n: int) -> Budget:
    raw = value.strip()
    if raw.endswith("%"):
        fraction = float(raw[:-1]) / 100.0
        count = max(1, int(round(n * fraction)))
        return Budget(label=raw, fraction=fraction, count=count)

    numeric = float(raw)
    if 0 < numeric < 1:
        count = max(1, int(round(n * numeric)))
        return Budget(label=f"{100 * numeric:g}%", fraction=numeric, count=count)

    count = int(round(numeric))
    if count <= 0:
        raise ValueError(f"budget must be positive, got {value!r}")
    return Budget(label=str(count), fraction=count / n, count=min(count, n))


def parse_budgets(values: Iterable[str], n: int) -> list[Budget]:
    return [parse_budget(value, n) for value in values]


def minority_mask_from_values(
    series: pd.Series, values: Sequence[str] | None
) -> np.ndarray:
    if values is None or len(values) == 0:
        counts = series.value_counts()
        if counts.empty:
            return np.zeros(len(series), dtype=bool)
        minority_value = counts.idxmin()
        return (series == minority_value).to_numpy()

    normalized = {str(value) for value in values}
    return series.astype(str).isin(normalized).to_numpy()


def first_true_rank(
    ranking: Sequence[int] | np.ndarray, mask: Sequence[bool] | np.ndarray
) -> int | None:
    ranked = np.asarray(ranking, dtype=int)
    mask_arr = np.asarray(mask, dtype=bool)
    hits = np.flatnonzero(mask_arr[ranked])
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def selection_metrics(
    *,
    method: str,
    budget: Budget,
    selected: np.ndarray,
    loss_selected: np.ndarray,
    noisy_mask: np.ndarray,
    minority_mask: np.ndarray,
    first_noisy_minority_loss_rank: int | None,
) -> dict:
    selected = np.asarray(selected, dtype=int)
    noisy_selected = noisy_mask[selected]
    minority_selected = minority_mask[selected]
    noisy_minority_selected = noisy_selected & minority_selected
    clean_minority_selected = (~noisy_selected) & minority_selected
    overlap = len(set(selected.tolist()) & set(loss_selected.tolist()))

    return {
        "method": method,
        "budget": budget.label,
        "budget_fraction": budget.fraction,
        "budget_count": budget.count,
        "selected_count": len(selected),
        "noisy_count": int(noisy_selected.sum()),
        "noisy_precision": float(noisy_selected.mean()) if len(selected) else 0.0,
        "minority_count": int(minority_selected.sum()),
        "minority_query_rate": float(minority_selected.mean()) if len(selected) else 0.0,
        "noisy_minority_count": int(noisy_minority_selected.sum()),
        "clean_minority_count": int(clean_minority_selected.sum()),
        "overlap_with_loss_count": int(overlap),
        "overlap_with_loss_rate": float(overlap / len(selected)) if len(selected) else 0.0,
        "first_noisy_minority_loss_rank": first_noisy_minority_loss_rank,
    }


def build_diagnostics_table(
    private_frame: pd.DataFrame,
    losses: Sequence[float] | np.ndarray,
    *,
    methods: Sequence[str],
    budgets: Sequence[Budget],
    group_col: str = "group",
    noisy_col: str = "is_noisy",
    minority_values: Sequence[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a diagnostics table comparing oracle-noise variants across budgets."""
    if group_col not in private_frame.columns:
        raise KeyError(f"missing group column: {group_col}")
    if noisy_col not in private_frame.columns:
        raise KeyError(f"missing noisy column: {noisy_col}")

    losses_arr = np.asarray(losses, dtype=float)
    noisy_mask = private_frame[noisy_col].astype(bool).to_numpy()
    minority_mask = minority_mask_from_values(private_frame[group_col], minority_values)
    noisy_minority_mask = noisy_mask & minority_mask

    loss_ranked = np.argsort(-losses_arr, kind="stable")
    first_noisy_minority_loss_rank = first_true_rank(loss_ranked, noisy_minority_mask)

    rows: list[dict] = []
    for method in methods:
        scores = score_by_method(
            method,
            private_frame,
            losses_arr,
            group_col=group_col,
            noisy_col=noisy_col,
            seed=seed,
        )
        ranking = np.argsort(-scores, kind="stable")

        for budget in budgets:
            selected = ranking[: budget.count]
            loss_selected = loss_ranked[: budget.count]
            rows.append(
                selection_metrics(
                    method=method,
                    budget=budget,
                    selected=selected,
                    loss_selected=loss_selected,
                    noisy_mask=noisy_mask,
                    minority_mask=minority_mask,
                    first_noisy_minority_loss_rank=first_noisy_minority_loss_rank,
                )
            )

    return pd.DataFrame(rows)
