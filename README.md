# Budgeted Label Verification for Latent-Group Robustness

This repository implements the **Stage 0–1 MVP**:

- Waterbirds metadata ingestion;
- public/private information separation;
- fixed uniform or group-dependent label corruption;
- frozen ImageNet ResNet-50 feature extraction;
- a shared linear probe with per-example training dynamics;
- random, loss, entropy, forgetting, and noise-likelihood querying;
- analysis-only noise, minority-aware, and group-balanced oracles;
- correction-only retraining from an identical initialization;
- average accuracy, group accuracies, WGA, query composition, and budget curves.

The central diagnostic is whether

\[
\arg\max_Q \#\{\text{corrected labels}\}
\neq
\arg\max_Q \Delta \mathrm{WGA}.
\]

This code intentionally does **not** use group labels in model training, score calculation,
query selection for legal methods, validation checkpoint selection, or hyperparameter tuning.
Private group labels are available only to corruption generation, analysis-only oracles, and
the final evaluator.

## Reviewer Quick Path

If you are reviewing the paper or preparing a clean reproduction, these are the
only paths you need first:

| Question | Open this file or directory |
| --- | --- |
| What is the paper's claim and experimental protocol? | [`paper/main_iclr2027.tex`](paper/main_iclr2027.tex) |
| How is the ICLR source built and which assets are included? | [`paper/ICLR2027_BUILD.md`](paper/ICLR2027_BUILD.md) |
| Where is the maintained Python implementation? | [`src/robust_verify/`](src/robust_verify/) |
| Where is the core module-by-module map? | [`docs/CORE_CODE_MAP.md`](docs/CORE_CODE_MAP.md) |
| How do I verify the code without downloading data? | [`docs/REVIEWER_REPRODUCTION.md`](docs/REVIEWER_REPRODUCTION.md) and [`tests/`](tests/) |
| Where is the RV-Q definition and implementation? | Paper RV-Q section in [`paper/main_iclr2027.tex`](paper/main_iclr2027.tex), [`src/robust_verify/modern_baselines.py`](src/robust_verify/modern_baselines.py), and [`src/robust_verify/scoring.py`](src/robust_verify/scoring.py) |
| Which configs are the small and full Waterbirds references? | [`configs/waterbirds_quick.yaml`](configs/waterbirds_quick.yaml) and [`configs/waterbirds_full.yaml`](configs/waterbirds_full.yaml) |
| Where are claim/result and experiment notes? | [`claim_evidence_ledger.md`](claim_evidence_ledger.md), [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md), and [`docs/`](docs/) |

The maintained source of truth is `src/robust_verify`.  The root package keeps
the public/private data boundary, probe training, legal and oracle scoring,
RV-Q value scoring, budget allocation, evaluation, and command-line runners in
one importable package.  Use [`docs/CORE_CODE_MAP.md`](docs/CORE_CODE_MAP.md)
when you need a more detailed reading order.

The following local material is **not required** for reviewer installation or
the no-dataset test suite: `data/`, `outputs/`, `output/`, `tmp/`, `dist/`,
root-level archives, and historical experiment snapshots such as
`budget_adaptive/`, `budget_hybrid/`, `gated_adaptive/`, and `train/`.  They
remain available locally for provenance and server hand-off.

## 1. Expected Waterbirds layout

Point `data.root` to a Waterbirds directory containing `metadata.csv` and the image paths
listed in its `img_filename` column.

Typical metadata columns are:

```text
img_filename, y, place, split
```

The loader also accepts `a` or `background` instead of `place`.

The code does not download Waterbirds automatically. Use the official group_DRO/WILDS
instructions and comply with the source dataset licenses.

## 2. Installation

```bash
cd latent_group_verification_mvp
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Run a dependency and core-logic smoke test:

```bash
python -m robust_verify.smoke_test
python -m pytest -q
```

On Windows, use `.venv\Scripts\python.exe -m ...` if the environment is not
activated.  The same commands are listed with the expected data prerequisites
in [`docs/REVIEWER_REPRODUCTION.md`](docs/REVIEWER_REPRODUCTION.md).

## 3. Configure the data path

Copy the quick config and edit:

```bash
cp configs/waterbirds_quick.yaml configs/local.yaml
```

Set:

```yaml
data:
  root: /absolute/path/to/waterbirds
  metadata_csv: metadata.csv
```

## 4. Run Stage 0

Stage 0 creates canonical metadata, frozen features, and deterministic public/private
corruption manifests.

```bash
rv-stage0 --config configs/local.yaml
```

Outputs include:

```text
outputs/waterbirds_mvp/
  canonical/
  features/
  manifests/
```

Public training manifests contain only image paths, sample IDs, and noisy/current labels.
Private manifests contain clean labels, groups, minority status, and corruption status.

## 5. Run Stage 1

```bash
rv-stage1 --config configs/local.yaml
```

Stage 1 trains one shared probe per experimental seed, builds all query rankings, performs
correction-only verification, retrains every method from the same initialization, and writes:

```text
outputs/waterbirds_mvp/stage1/
  probes/
  dynamics/
  queries/
  results.csv
  score_correlations.csv
  plots/
```

Run both stages:

```bash
rv-run-all --config configs/local.yaml
```

## 6. Quick and full configurations

`configs/waterbirds_quick.yaml`:

- one seed;
- five probe/retraining epochs;
- budgets 0.5%, 1%, and 2%;
- intended for checking the pipeline.

`configs/waterbirds_full.yaml`:

- five seeds;
- twenty epochs;
- budgets 0.5%, 1%, 2%, and 5%;
- intended for the Stage 1 diagnostic experiment.

## 7. Legal and oracle query methods

Legal methods:

```text
random
loss
entropy
forgetting
noise_score
```

Analysis-only methods:

```text
oracle_noise
oracle_minority
oracle_group_balanced
```

Oracle methods read the private manifest and must never be presented as deployable methods.

RV-Q/value-aligned methods:

```text
expected_repair_value
expected_repair_value_multiclass
expected_repair_value_adaptive_tau
expected_repair_value_capture_tau
expected_repair_value_calibrated_tau
noise_gated_repair_value
reliability_gated_repair_value
```

The fixed-tail RV-Q score is the product of a clipped legal noise proxy and a
normalized first-order validation-tail repair-value rank.  Its implementation
and all adaptive variants are in
[`src/robust_verify/modern_baselines.py`](src/robust_verify/modern_baselines.py);
registration and per-budget dispatch are in
[`src/robust_verify/scoring.py`](src/robust_verify/scoring.py).

## 8. Key output columns

`results.csv` includes:

```text
noise_name
seed
method
budget_fraction
budget_count
num_corrected
noise_precision
minority_query_rate
clean_minority_rate
group_query_entropy
average_accuracy
balanced_accuracy
wga
delta_average_accuracy
delta_wga
```

The most important checks are:

1. Does `noise_score` correct more labels than a failure-oriented or oracle-minority policy?
2. Does correcting more labels always imply a larger WGA gain?
3. Is there a meaningful gap between `oracle_noise` and `oracle_minority`?
4. Is there enough oracle headroom to justify developing latent failure slices?

For reviewer-facing result interpretation, preserve the full per-seed
`results.csv`, query IDs under `stage1/queries/`, and the matching corruption
manifest.  Do not rely on an aggregate table without its seed-level records.

## 9. Reproducibility

Each experiment seed deterministically controls:

- corruption generation;
- linear-head initialization;
- minibatch shuffling;
- random query ranking.

All methods at a given seed use the same frozen features, corruption manifest, initial
linear-head state, optimizer settings, and validation checkpoint rule.

## 10. Important limitations

This is a diagnostic MVP, not the final proposed method.

- It uses a frozen backbone and a linear head.
- Stage 1 uses correction-only retraining.
- Noise probability is a rank-normalized likelihood score, not a calibrated probability.
- The oracle methods expose private information only to quantify headroom.
- The project does not yet implement latent clustering, certification-aware upweighting,
  multi-round active querying, or influence functions.

Those components should be added only after Stage 1 establishes the target mismatch.

## 11. Practical absolute-utility audit

For a practical query method, being worse than `loss` is not the same claim as
causing absolute harm relative to doing no correction.  Run the audited
comparison after a frozen matrix completes:

```bash
python scripts/p0_absolute_utility.py \
  --results waterbirds=outputs/p0_wave_s1/waterbirds/stage1/results.csv \
  --results celeba=outputs/p0_wave_s1/celeba/stage1/results.csv \
  --out-dir outputs/p0_control/practical_absolute_utility
```

The command excludes oracle rows, preserves seed pairing against `loss`, and
writes both condition-level and unaggregated tables.  See
[`docs/PRACTICAL_ABSOLUTE_UTILITY.md`](docs/PRACTICAL_ABSOLUTE_UTILITY.md) for
the output schema and paper-claim interpretation rules.
