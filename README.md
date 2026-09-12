# Budgeted Label Verification for Latent-Group Robustness

This repository contains a self-contained Stage 0-1 implementation for studying
budgeted label verification when group membership is not available to the deployed
selection rule.

The pipeline separates public information from private evaluation information and
measures whether selecting the most suspicious labels is the same as selecting the
labels whose correction most improves worst-group accuracy (WGA). In general, these
objectives need not agree:

~~~
maximise the number of corrected labels != maximise the improvement in WGA
~~~

The repository is intended to be used without author-specific metadata. It does not
include the manuscript or documentation directories; the code, configurations, tests,
and runnable experiment scripts are the relevant artifacts.

## Repository map

~~~
src/robust_verify/       Core Python package and command-line entry points
configs/                 Example experiment configurations
scripts/                 Extended analyses and server-side runners
tests/                   Unit and end-to-end smoke tests
pyproject.toml           Package metadata and dependencies
requirements.txt         Pinned/explicit runtime requirements
LICENSE                  MIT license
~~~

The primary implementation is under src/robust_verify. The package includes data
loaders, public/private manifest handling, feature extraction, probe training, legal
query scoring, private oracles, budget allocation, evaluation, and result plotting.

## What is implemented

- Waterbirds metadata loading and canonical split generation.
- Synthetic uniform and group-dependent label corruption.
- Public training manifests separated from private clean-label/group manifests.
- Frozen ImageNet ResNet-50 features with a shared linear probe.
- Per-example training dynamics for loss, entropy, forgetting, and noise-likelihood
  ranking.
- Correction-only retraining from a matched initialization.
- Analysis-only noise, minority, and group-balanced oracle policies.
- Average accuracy, balanced accuracy, group accuracy, WGA, query composition, and
  budget-response curves.

Legal query rules do not read group labels. Private group information is used only for
corruption generation, oracle analyses, and final evaluation.

## Requirements and installation

Python 3.10 or newer is required. A CUDA-capable PyTorch installation is recommended
for ResNet-50 feature extraction, although the small smoke tests can run on CPU.

~~~
python -m venv .venv
~~~

Activate the environment using the command for your shell:

~~~
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

Install the package and development dependencies:

~~~
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Run the dependency-light smoke test and the test suite:

~~~
python -m robust_verify.smoke_test
python -m pytest -q
~~~

## Data layout

The example configurations expect a Waterbirds directory containing metadata.csv
and the image files referenced by its img_filename column:

~~~
waterbirds/
  metadata.csv
  images/
    ... image files ...
~~~

The metadata normally contains:

~~~
img_filename, y, place, split
~~~

The loader also accepts a or background as the environment/place column. The
dataset is not downloaded automatically. Obtain it from the official dataset source
and follow its license and usage conditions.

## Configure an experiment

Start from one of the two example configurations:

- configs/waterbirds_quick.yaml: one seed, five probe epochs, and budgets of
  0.5%, 1%, and 2%; suitable for a pipeline check.
- configs/waterbirds_full.yaml: five seeds, twenty probe epochs, and budgets of
  0.5%, 1%, 2%, and 5%; suitable for the reference Stage 0-1 experiment.

Copy either file to a local configuration and set data.root to the Waterbirds
directory. For example:

~~~
project:
  output_dir: outputs/waterbirds_mvp

data:
  root: /path/to/waterbirds
  metadata_csv: metadata.csv
~~~

Use a repository-relative or user-local path in the configuration. Do not commit
machine-specific absolute paths.

## Run the pipeline

Stage 0 prepares canonical metadata, frozen features, and deterministic public/private
corruption manifests:

~~~
rv-stage0 --config configs/waterbirds_full.yaml
~~~

Stage 1 trains the shared probe, builds query rankings, performs correction-only
verification, retrains each policy from the same initialization, and evaluates the
result:

~~~
rv-stage1 --config configs/waterbirds_full.yaml
~~~

Run both stages in sequence:

~~~
rv-run-all --config configs/waterbirds_full.yaml
~~~

The main output directory contains the following artifacts:

~~~
outputs/waterbirds_mvp/
  canonical/       Canonical train/validation/test tables
  features/        Cached feature arrays
  manifests/       Public and private corruption manifests
  probes/          Probe checkpoints and training dynamics
  queries/         Ranked and budget-truncated query IDs
  checkpoints/     Correction-only retraining checkpoints
  stage1/          results.csv and score correlations
  plots/            Generated result figures
~~~

## Query policies

The example configurations include these policies:

~~~
random
loss
entropy
forgetting
noise_score
oracle_noise
oracle_minority
oracle_group_balanced
~~~

The oracle_* policies use private information and are analysis baselines only; they
must not be interpreted as deployable selection rules. Additional registered policies
and adaptive variants are implemented in src/robust_verify/scoring.py and
src/robust_verify/modern_baselines.py.

## Reading the results

The Stage 1 result table reports, among other fields:

~~~
noise_name
seed
method
budget_fraction
budget_count
num_corrected
noise_precision
minority_query_rate
average_accuracy
balanced_accuracy
wga
delta_average_accuracy
delta_wga
~~~

The most important comparisons are:

1. Whether a legal policy improves WGA relative to no correction.
2. Whether correcting more labels also improves WGA.
3. The gap between noise-oriented and minority/group-oriented oracle policies.
4. Whether the result is stable across seeds and corruption settings.

Keep the per-seed results.csv, query ID arrays, and the matching private manifest
together when interpreting or reproducing a run.

## Reproducibility notes

For a fixed configuration and seed, the pipeline controls corruption generation,
linear-head initialization, minibatch shuffling, and query ranking. All policies at a
given seed use the same frozen features, corruption manifest, initialization,
optimizer settings, and validation checkpoint rule.

The implementation is a diagnostic MVP. It uses a frozen backbone and a linear head,
and its corruption process is synthetic. It does not claim to implement every possible
latent-group discovery, certification, or multi-round intervention strategy.

## License

The code is released under the MIT License. Dataset licenses and access conditions are
separate and remain the responsibility of the user.
