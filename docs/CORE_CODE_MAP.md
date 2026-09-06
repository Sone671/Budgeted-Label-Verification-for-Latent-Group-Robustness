# Core Code Map

This map defines the maintained source of truth for review and future reuse.
The repository also contains historical experiment bundles; those are retained
for provenance but are not alternate implementations of the core package.

## Maintained source

| Area | Source of truth | Role |
| --- | --- | --- |
| Public/private data boundary | `src/robust_verify/data/` | Dataset loaders, manifests, corruption generation, and private evaluation |
| Feature preparation | `src/robust_verify/features.py` | Frozen representation extraction and cache validation |
| Probe training | `src/robust_verify/training.py` | Shared initialization, linear-head training, checkpoints, and dynamics |
| Query scoring | `src/robust_verify/scoring.py` | Legal baselines, oracle diagnostics, AUM/Cleanlab-style scores, and adaptive rankings |
| RV-Q value model | `src/robust_verify/modern_baselines.py` | First-order validation-tail repair value, multiclass extension, and reliability/adaptive gates |
| Budget allocation | `src/robust_verify/selection.py` | Coverage, hybrid, gated, and budget-aware ranking primitives |
| Experiment runners | `src/robust_verify/stage0.py`, `stage1.py`, `end_to_end.py` | Reproducible preparation, ranking, correction, retraining, and result writing |
| Metrics and audits | `src/robust_verify/analysis.py`, `diagnostics.py`, `absolute_utility.py` | Query composition, WGA, paired utility, and audit summaries |
| Theory and auxiliary analyses | `src/robust_verify/*_theory.py`, `action_value.py`, `joint_budget.py` | Formal witnesses and value decompositions used by the manuscript |
| Command-line entry points | `src/robust_verify/cli/` | Stage 0/1, end-to-end, diagnostics, and smoke-test commands |

The `src/` package is the only implementation path used by the root
`pyproject.toml`.  Directories such as `budget_adaptive/`, `budget_hybrid/`,
`gated_adaptive/`, and `train/` are historical snapshots or standalone
prototypes and should not be imported by new experiments.

## Method registry

`src/robust_verify/scoring.py` is the registry for query methods:

- `METHOD_NAMES` is the complete accepted method vocabulary.
- `ADAPTIVE_METHODS` identifies methods whose ranking is built per budget.
- `noise_score`, `noise_score_budget_hybrid`, and
  `noise_score_budget_hybrid_v8` are the v1/v7/v8 NoiseScore family.
- `expected_repair_value` is the fixed-tail RV-Q definition; the related
  entries are `expected_repair_value_multiclass`,
  `expected_repair_value_adaptive_tau`,
  `expected_repair_value_capture_tau`,
  `expected_repair_value_calibrated_tau`, `noise_gated_repair_value`, and
  `reliability_gated_repair_value`.
- `NOISE_RISK_LOSS_RATIO_THRESHOLD = 4.0` is the current v8 risk gate.

The registration and dispatch contract is covered by
`tests/test_method_registration.py`.  Keep the test and registry change in the
same review when adding or renaming a method.

## Review path

1. Start with `README.md` and `docs/REVIEWER_REPRODUCTION.md`.
2. Follow `configs/waterbirds_quick.yaml` for a data-backed smoke configuration;
   `configs/waterbirds_full.yaml` is the larger frozen-feature reference.
3. Read `stage0.py` and `stage1.py` to see the public/private information
   boundary and the correction-only evaluation protocol.
4. Read `scoring.py` and `modern_baselines.py` for detector/value definitions,
   then `selection.py` for budget behavior and `training.py` for shared
   initialization and checkpoint selection.
5. Use `tests/` as executable specifications before running any expensive
   dataset experiment.
6. Build the manuscript from `paper/main_iclr2027.tex`; use the official style
   files and figures under `paper/` rather than generated PDFs in `output/`.

## Retained but out of scope for the core package

The following local material is intentionally preserved for provenance or
server hand-off, but is not required for reviewer installation:

- `data/`, `outputs/`, `output/`, `tmp/`, and `dist/` contain local data or
  generated artifacts;
- root-level archives (`*.zip`, `*.rar`, `*.tar.gz`) are snapshots and result
  bundles;
- `server/`, `baseline_supplement_server/`, and the phase-specific experiment
  directories contain workload-specific runners;
- `joint_group_label_budget/` contains a separate theory/manuscript line;
- `paper/output/` and `paper/tmp/` contain LaTeX intermediates.

None of these directories is deleted by the cleanup.  They are excluded from
the reviewer-facing source boundary so a fresh checkout remains understandable
and reproducible.
