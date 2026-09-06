# CelebA uniform20 / 10% end-to-end TracIn-CP (val)

## Purpose

This configuration tests whether a practical TracIn-CP (val) query set still
causes the frozen-matrix CelebA uniform20 / 10% contrast after full ResNet-50
retraining.  It reports each method's absolute `delta_wga` against the
per-seed no-correction model and its seed-paired difference against Loss.

The run is a training-mechanism transfer test, not a fresh-seed confirmatory
trial: it consumes the existing `ablation_celeba_10seeds` manifests for outer
seeds 0--9.  For an independent confirmatory claim, create an otherwise
identical source directory with held-out corruption seeds (for example 10--19)
and update `project.input_dir` and `experiment.seeds` before running.

## What the code does

`tracin_val_uncertainty` is now supported by `rv-end-to-end`.  The runner:

1. trains the ResNet-50 probe using only the public noisy training manifest and
   public validation labels;
2. reloads its best validation checkpoint, extracts its penultimate train and
   validation embeddings, and computes TracIn-CP (val) with validation
   uncertainty weights;
3. saves query IDs before opening the private verification manifest;
4. corrects only those IDs, fully retrains from the same initialization, and
   evaluates test WGA through the private evaluator.

This remains the repository's single-best-checkpoint TracIn-CP proxy.  It is
not a multi-checkpoint canonical TracIn implementation and must be named
`TracIn-CP (val)` in the paper.

## Server command

```bash
cd /path/to/latent_group_verification_mvp
python -m robust_verify.cli.end_to_end \
  --config configs/celeba_e2e_tracin_uniform20_10pct.yaml
```

Expected output: 50 result rows (10 seeds x 5 methods), each with
`budget_count = 16275`.  Query IDs and retraining checkpoints are saved below
`outputs/celeba_e2e_tracin_uniform20_10pct/end_to_end/`.

## Required report

Run the absolute-utility audit after the experiment:

```bash
PYTHONPATH=src python scripts/p0_absolute_utility.py \
  --results celeba=outputs/celeba_e2e_tracin_uniform20_10pct/end_to_end/results.csv \
  --method tracin_val_uncertainty \
  --method noise_score \
  --method noise_score_cpba_only \
  --out-dir outputs/celeba_e2e_tracin_uniform20_10pct/audit
```

Report absolute Delta-WGA versus `baseline_wga`, paired Delta-WGA versus Loss,
negative seed counts, correction count, query noise precision, and offline
minority-side query composition.  Do not treat Random as no correction.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_end_to_end_smoke.py -q
```
