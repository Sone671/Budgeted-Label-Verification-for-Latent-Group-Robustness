# CelebA Young/Male RepairValue-Q replication

This package defines a new, task-level end-to-end replication of RepairValue-Q.
It is disjoint from the existing CelebA Eyeglasses/Male protocols.

## Frozen protocol

- target: `Young`
- spurious attribute: `Male`
- corruption: `uniform20` (20% independent binary flips)
- query budget: 10% (`16,277` labels for the standard CelebA train split)
- seeds: `60--79` (20 fresh corruption/training seeds)
- backbone: pretrained ResNet-50, full-parameter retraining
- checkpoint selection: public validation balanced accuracy
- RepairValue-Q tail fraction: `0.20`
- methods: Random, Loss, NoiseScore, OOF Cleanlab, RepairValue-Q

The primary estimand is the seed-paired contrast

```text
delta_WGA(RepairValue-Q) - delta_WGA(Loss)
```

The registered gate requires a positive paired-bootstrap lower endpoint, at
least 15 of 20 positive paired seeds, and a positive absolute RepairValue-Q
delta-WGA bootstrap lower endpoint.  A failed gate is retained as a null result.

## OOF Cleanlab definition

`oof_cleanlab` uses five deterministic stratified linear probes over the
baseline probe's penultimate features. Each training example is scored by the
probe whose fit fold excludes that example. Only observed noisy labels and
public probe features are used; validation, test, clean-training, and group
labels are not used for scoring. The OOF probabilities and fold IDs are saved
under `end_to_end/oof/` for replay.

This is an OOF linear-probe baseline, not five full-network OOF models. The
paper-facing label should therefore be `Cleanlab (OOF linear-probe score)`.

## Running on a CUDA server

Run from the repository root. Use a fresh output root for every formal run.

```powershell
python scripts/run_celeba_young_male_server.py preflight
python scripts/run_celeba_young_male_server.py prepare
python scripts/run_celeba_young_male_server.py run
python scripts/run_celeba_young_male_server.py aggregate
```

The shell wrapper accepts `CELEBA_ROOT` and `OUTPUT_ROOT` overrides on a
Unix-like server:

```bash
CELEBA_ROOT=/path/to/celebA_v1.0 \
OUTPUT_ROOT=/path/to/results/celeba_young_male_seed60_79 \
./scripts/run_celeba_young_male_server.sh all
```

The runner seals Stage-0 public/private manifests, preserves failed attempts,
validates every query and prediction artifact, and writes an aggregate
`artifact_manifest.json`. It does not replace old protocol files.

## Expected outputs

The aggregate contains 100 rows (20 seeds x 5 methods). Each seed has one
shared no-correction baseline and five corrected full-retraining outcomes.
The aggregate audit includes paired effects versus Loss, absolute utility,
query composition, baseline-worst-group coverage, and the registered decision.

## Scope of the baseline matrix

The frozen confirmatory matrix contains exactly the five methods listed above.
`AUTO-D3M-Q` is available in the repository as the explicit adaptation
`auto_d3m_query`, but it is intentionally excluded from this protocol: the
published method uses a different action space (removing or downweighting
examples), whereas this experiment only permits querying and correcting labels.
Adding it here would change the registered method set and require a separate
exploratory protocol/version; it must not be appended after seeing the
confirmatory outcomes.
