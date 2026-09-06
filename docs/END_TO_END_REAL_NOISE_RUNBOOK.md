# End-to-End and CIFAR-N Verification Runbook

## Scope

This bundle adds two confirmation paths without changing the frozen-feature
diagnostic matrix:

1. CIFAR-10N with real human annotation noise and clean-label verification
   replay. Its outcome is worst-class accuracy, not a claim about latent
   spurious groups.
2. Waterbirds with all ResNet parameters trainable after each query-set label
   repair. Its outcome remains WGA.

The standard Stage-1 implementation remains the frozen-feature protocol. Use
`rv-end-to-end` only for the three supplied confirmation configurations.

## Privacy Boundary

For CIFAR-10N, the official `clean_label` array is written only to private
manifests. Public training manifests contain `sample_id`, image path, and the
chosen human noisy label. `rv-end-to-end` rejects public manifests containing
`clean_label`, rejects all `oracle_*` acquisition methods, and hands clean
labels to the learner only after a ranking prefix has been frozen and saved as
a query-ID artifact.

The canonical CSV is an internal preparation artifact, not an input to scoring
or model selection. Do not inspect its train `clean_label` column while
developing or selecting a practical method.

## Installation

From the repository root, use the existing virtual environment or install the
package in editable mode:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Confirm CUDA visibility before a real run:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## CIFAR-10N Preparation and Run

1. Download the official `CIFAR-10_human.pt` annotation artifact from the
   CIFAR-N release and place it at `data/cifar10n/CIFAR-10_human.pt`.
2. Put the torchvision CIFAR-10 files under `data/cifar10n`, or set
   `data.download: true` for the first Stage-0 run.
3. Run Stage 0 once. It materializes 32x32 PNGs and writes public/private
   manifests. It intentionally skips frozen feature extraction.

```powershell
rv-stage0 --config configs/cifar10n_end_to_end_confirmatory.yaml
rv-end-to-end --config configs/cifar10n_end_to_end_confirmatory.yaml
```

The result file is
`outputs/cifar10n_end_to_end_confirmatory/end_to_end/results.csv`.
Report `delta_worst_class_accuracy` (identical to `delta_wga` for this dataset)
as **delta worst-class accuracy**, because the evaluation group is the clean
CIFAR class. Do not call it a latent-group or spurious-correlation result.

## Waterbirds End-to-End Run

Run the 2% confirmation before the high-budget stress configuration:

```powershell
rv-stage0 --config configs/waterbirds_end_to_end_confirmatory.yaml
rv-end-to-end --config configs/waterbirds_end_to_end_confirmatory.yaml

rv-stage0 --config configs/waterbirds_end_to_end_stress.yaml
rv-end-to-end --config configs/waterbirds_end_to_end_stress.yaml
```

All methods within a seed reuse the probe initialization. Each result writes a
query-ID artifact and the exact final model state under `end_to_end/`.

## Frozen Matrix and Promotion Rule

Do not substitute these runs for the mechanism confirmations. Execute the
existing frozen-representation matrix in this order:

1. matched-pair: three datasets, two noise mechanisms, ten data seeds;
2. spectral, posterior-times-influence, and cluster closed-loop confirmations;
3. CIFAR-10N real-noise replay;
4. Waterbirds end-to-end confirmation;
5. Waterbirds 10% stress test.

Use the initial two seeds only to test data loading, GPU memory, artifacts, and
clean-label isolation. Freeze every configuration before extending to ten
seeds. The statistical unit is the outer seed, never budgets, methods, or
matched pairs within a seed.

## Local Test Commands

```powershell
pytest -q
python -m robust_verify.smoke_test
```

The added tests use generated tiny images and `tiny_cnn`; they require neither
CIFAR-N files nor a GPU. A passing smoke test confirms the public/private
boundary, end-to-end one-shot repair loop, query artifacts, and result schema.
