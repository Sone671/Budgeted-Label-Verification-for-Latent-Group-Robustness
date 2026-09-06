# Waterbirds robust-set RV-Q exploratory result

## Method

`robust_set_repair_value_ranking` is a set-level upgrade of RV-Q. It builds
5-fold cross-fitted validation scenarios over tail fractions `{0.10, 0.20,
0.50}`, ranks the signed repair effect in each scenario, multiplies by the
legal NoiseScore proxy, and greedily selects a budget prefix using an
aggregate mean-minus-uncertainty objective plus a random-projection cosine
redundancy penalty. The method is exploratory and is not registered as a
production scoring method.

## Protocol

- Waterbirds cached frozen features: `outputs/ablation_waterbirds_20seeds`.
- Seeds: 0--19; noise: `uniform20`, `minority_high_40`.
- Budgets: 2% and 5%.
- Retraining: the same frozen linear-head protocol and initial states as
  `outputs/local_waterbirds_repairvalue_comparison_20seeds`.
- Reference methods: Loss, NoiseScore, and fixed RV-Q (`tau=.20`, `.50`).

## Paired results

Values below are percentage points; each comparison is paired by seed and
bootstrap intervals are 10,000-replicate intervals.

| noise | budget | set-RV-Q ΔWGA | vs Loss | vs NoiseScore | vs RV-Q τ=.20 | vs RV-Q τ=.50 | query precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| minority_high_40 | 2% | +0.35 | +5.68 [2.13, 9.29] | **−5.44 [−9.19, −2.10]** | +0.90 [−0.28, 2.45] | −3.53 [−7.73, 0.55] | 8.65% |
| minority_high_40 | 5% | +1.60 | +9.80 [7.42, 12.20] | **−8.12 [−11.36, −5.16]** | +0.41 [−1.78, 2.09] | **−6.91 [−11.04, −3.54]** | 6.48% |
| uniform20 | 2% | +3.06 | +1.95 [−0.11, 3.88] | +1.46 [−1.16, 3.91] | +0.30 [−0.49, 1.06] | **−3.66 [−6.77, −0.02]** | 8.59% |
| uniform20 | 5% | +5.81 | +8.32 [5.48, 11.17] | **+8.61 [5.67, 11.51]** | **+4.00 [1.00, 7.69]** | −1.24 [−3.95, 1.52] | 8.19% |

The full per-seed rows and summary are in
`outputs/local_waterbirds_set_rvq_v2/per_seed_adaptive.csv` and
`outputs/local_waterbirds_set_rvq_v2/comparison_summary.csv`.

## Decision

Do **not** spend the next cycle on full end-to-end retraining or paper
promotion of this version. The method wins on `uniform20` at 5%, but loses
to NoiseScore on both minority-shift cells and loses to fixed RV-Q `tau=.50`
on three of four cells (significantly on minority 5% and uniform20 2%). The
low query precision (6--9%) also shows that the set objective is trading away
the detector signal too aggressively.

If continuing, restrict it to a small score-only ablation: impose a NoiseScore
floor/top-quantile candidate pool and tune the uncertainty/redundancy weights
on a held-out Waterbirds split. Require non-inferiority to NoiseScore on both
noise regimes and to fixed `tau=.50` before any new full retraining.
