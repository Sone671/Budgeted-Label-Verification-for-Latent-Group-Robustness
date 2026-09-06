# Reviewer Reproduction

This document is the short path for inspecting and running the maintained
implementation.  It deliberately separates the executable core from local
datasets, generated results, and historical snapshots.

## 1. Install

From the repository root:

```text
python -m venv .venv
```

On Windows:

```text
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On POSIX systems, activate `.venv` and run the equivalent `python -m pip`
commands.  The package declares the runtime dependencies in `pyproject.toml`;
the `dev` extra adds pytest and Ruff.

## 2. Fast verification

These commands do not require Waterbirds or any other dataset:

```text
python -m robust_verify.smoke_test
python -m pytest -q
python -m pytest tests/test_core.py tests/test_method_registration.py -q
```

The smoke test exercises deterministic probe training, legal and oracle
rankings, private verification, and WGA evaluation on synthetic data.  The
full suite covers the scoring, selection, theory, data adapters, and the
end-to-end runner.  A clean run should report all tests passed; warning lines
from scikit-learn about convergence or deprecated arguments do not indicate a
test failure.

## 3. Data-backed run

The code never downloads datasets automatically.  Prepare a licensed
Waterbirds directory containing `metadata.csv` and the referenced images, then
copy and edit `configs/waterbirds_quick.yaml` (or create your own local config
copy):

```yaml
data:
  root: /absolute/path/to/waterbirds
  metadata_csv: metadata.csv
```

Run the two stages from the repository root:

```text
python -m robust_verify.cli.stage0 --config configs/waterbirds_quick.yaml
python -m robust_verify.cli.stage1 --config configs/waterbirds_quick.yaml
```

Stage 0 writes canonical metadata, frozen features, and public/private
corruption manifests.  Stage 1 trains a shared linear probe, builds rankings,
corrects only queried labels, retrains from the same initialization, and writes
per-seed metrics and query records under the configured output directory.

For a small local check, keep the quick configuration's one seed and five
epochs.  Use `configs/waterbirds_full.yaml` only when the cached features and
the required compute are available.

## 4. Information boundary

Legal query rules may use inputs or frozen representations, noisy labels,
model outputs, training dynamics, and the clean validation split.  Clean
training labels and group labels are available only to the private manifest,
offline oracle diagnostics, and final evaluation.  The `oracle_*` methods in
`scoring.py` quantify headroom and are not deployable baselines.

## 5. Paper build

The current ICLR source and the official style files are under `paper/`.
Build instructions, page-limit notes, and included assets are maintained in
[`paper/ICLR2027_BUILD.md`](../paper/ICLR2027_BUILD.md).  Generated `.aux`,
`.log`, PDFs, and temporary extraction files are not needed to run the Python
tests.

## 6. Scope and reproducibility notes

- The maintained implementation is `src/robust_verify`; do not mix it with
  historical copies under experiment directories.
- Config paths are intentionally explicit and relative to the repository root;
  record the exact config, seed list, feature cache, and corruption manifest
  with any result bundle.
- Results under `outputs/` are local artifacts and are ignored by Git.  Keep
  the raw per-seed CSV/JSON and query IDs when preparing a reviewer package.
- The maintained-core lint gate is:
  `python -m ruff check src/robust_verify tests --select E4,E7,E9,F`.
- `ruff check src tests scripts` currently includes many exploratory scripts
  and reports legacy style debt.  Ruff output is therefore informational for
  this repository; pytest and the smoke test are the executable correctness
  gates for the maintained core.
