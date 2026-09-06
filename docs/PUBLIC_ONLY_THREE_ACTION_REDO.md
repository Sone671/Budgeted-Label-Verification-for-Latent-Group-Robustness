# Public-Only Three-Action Redo

This pilot is a leakage-controlled rerun of the R/G/Stop protocol. The
selector reads public features, noisy labels, public validation labels,
purchased group responses, and frozen Bernoulli propensities. Private manifests
are used only by the outer driver to return a purchased response and to score
the final retrained model.

## Protocol

- Dataset caches: `ablation_celeba_10seeds` and `ablation_waterbirds_10seeds`.
- Conditions: `uniform20` and `minority_high_40`, seeds 0--2.
- Budget: 2% of the training population.
- Audit design: independent Bernoulli inclusion with `pi=0.5`.
- A fixed public-only scout audit precedes the certificate decision.
- Certificate states: `continue`, `stop`, or `unresolved`; unresolved states use
  a conservative fallback and are never relabeled as certified Stop.
- Coverage audit: 1,000 Monte Carlo audit draws using simultaneous group
  Hoeffding intervals under the equal-propensity design.

## Results

| Dataset | Audit cost ratio | Episodes | Simultaneous coverage | Mean max interval width | Mean selector utility | Mean delta vs Verify-only | Unresolved rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| CelebA | 0.1 | 6 | 1.000 | 0.308 | -0.0315 | +0.0491 | 1.000 |
| Waterbirds | 0.1 | 6 | 1.000 | 0.417 | -0.1335 | +0.2011 | 1.000 |

The selector abstained from unverified label repair in all reported episodes.
The relative improvement over Verify-only is caused by avoiding harmful repair
branches, but the absolute utility remains negative because the scout/audit
cost is not yet offset by a certified positive continuation value.

## Reproduction

```powershell
python scripts/run_public_only_three_action.py `
  --root outputs/ablation_celeba_10seeds `
  --output-dir outputs/public_only_three_action_celeba_cost01 `
  --seeds 0 1 2 `
  --noise uniform20 minority_high_40 `
  --coverage-draws 1000 `
  --audit-cost-ratio 0.1

python scripts/run_public_only_three_action.py `
  --root outputs/ablation_waterbirds_10seeds `
  --output-dir outputs/public_only_three_action_waterbirds_cost01 `
  --seeds 0 1 2 `
  --noise uniform20 minority_high_40 `
  --coverage-draws 1000 `
  --task-batch-size 32 `
  --audit-batch-size 64 `
  --audit-cost-ratio 0.1
```

These are pilot results, not confirmatory paper claims. The next method step is
to reduce interval width for rare groups and to estimate action-after-retraining
value with a separately calibrated, cross-fitted public model.
