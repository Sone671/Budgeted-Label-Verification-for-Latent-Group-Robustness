# Practical absolute-utility audit

This audit answers two different questions for each practical query method,
always within a fixed `(dataset, noise, budget, outer corruption seed)` setup.

1. **Does the method help or harm WGA relative to no correction?**
   The answer is the method's `delta_wga`, defined as
   `WGA(method) - WGA(no_correction)` after labels are verified and the model
   is retrained.
2. **Does the method help or harm WGA relative to Loss?**
   The answer is the seed-paired difference
   `delta_wga(method) - delta_wga(loss)`.

These are deliberately not interchangeable.  A negative comparison to Loss
does **not** establish absolute harm if the method's own `delta_wga` remains
positive.  Conversely, a method can cause absolute harm while still looking
less harmful than another method.

## Run the audit

Use explicitly named inputs so the output records the dataset identity:

```powershell
python scripts/p0_absolute_utility.py `
  --results waterbirds=outputs/p0_wave_s1/waterbirds/stage1/results.csv `
  --results celeba=outputs/p0_wave_s1/celeba/stage1/results.csv `
  --out-dir outputs/p0_control/practical_absolute_utility
```

To audit only a predeclared practical method, repeat `--method`:

```powershell
python scripts/p0_absolute_utility.py `
  --results civilcomments=outputs/ablation_civilcomments_10seeds/stage1/results.csv `
  --method noise_score_cpba_tc_v8_fallback `
  --out-dir outputs/p0_control/civilcomments_absolute_utility
```

The command is read-only with respect to raw result files.  It creates only
the requested audit directory.

## Inputs and integrity checks

Each CSV requires:

```text
noise_name, seed, method, budget_fraction, delta_wga
```

When `wga` and `baseline_wga` are also available, the audit rejects any row
where:

```text
delta_wga != wga - baseline_wga
```

This is a guard against mixing a method result with the wrong no-correction
baseline.  Duplicate `(dataset, noise_name, seed, method, budget_fraction)`
rows are also rejected rather than pooled.

By default, all methods whose names contain `oracle` are excluded, as are
`loss` and `no_correction` themselves.  Random, entropy and other legal query
baselines remain in the output unless explicitly excluded with
`--exclude-method`.

## Outputs

| File | Contents |
| --- | --- |
| `absolute_vs_no_correction.csv` | Condition-level absolute ΔWGA summaries. |
| `paired_vs_loss.csv` | Seed-paired method-minus-Loss ΔWGA summaries. |
| `per_seed_absolute_vs_no_correction.csv` | Unaggregated absolute effect for every candidate/seed. |
| `per_seed_vs_loss.csv` | Unaggregated, matched method-minus-Loss differences. |
| `coverage.csv` | Candidate/Loss seed counts and unmatched-seed diagnostics. |
| `audit_report.json` | Inputs, methods, bootstrap parameters and baseline integrity metadata. |
| `PRACTICAL_ABSOLUTE_UTILITY_REPORT.md` | Compact human-readable summary. |

Each condition reports a percentile bootstrap 95% interval of the seed mean,
direction counts, and an exact two-sided sign test with ties removed.  The
script does not pool budgets or treat multiple methods from the same seed as
independent replicates.

## Interpretation protocol

Use the following language in the paper:

| Absolute vs no correction | Paired vs Loss | Permitted conclusion |
| --- | --- | --- |
| Negative with interval below zero | Any direction | Practical method shows absolute WGA harm in that predeclared condition. |
| Positive with interval above zero | Negative with interval below zero | Practical method helps relative to no correction but underperforms Loss. This is relative ranking mismatch, not absolute harm. |
| Interval crosses zero | Any direction | No stable absolute-effect claim. |
| Any direction | Interval crosses zero | No stable relative-to-Loss claim. |

The default evidence labels are mechanical summaries of the unadjusted
condition-level intervals.  They do not correct for choosing a result after
examining a large matrix.  For a confirmatory paper claim, predeclare the
condition and method, or use new held-out corruption seeds.
