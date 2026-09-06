# Public-Only Three-Action: Next Round

This round fixes the audit-design mismatch in the pilot and evaluates a
deployable safety-first protocol. The private validation manifest is used only
to return responses for units that were independently sampled by the frozen
audit design, and for the final blind test evaluation.

## Protocol

- Datasets: CelebA and Waterbirds.
- Noise settings: `uniform20` and `minority_high_40`.
- Seeds: 0, 1, 2 for each setting (6 episodes per dataset).
- Training verification budget: 2% of the training population.
- Group audit: independent Bernoulli inclusion with uniform `pi=0.5`.
- Every included audit unit is purchased; no included prefix is discarded.
- Uniform designs use simultaneous conditional group Hoeffding intervals.
- The selector defaults to `allow_uncalibrated_verify=False`: an unresolved
  value proxy cannot certify label repair, so the fallback action is Stop.
- Coverage uses 500 Monte Carlo draws on the uniform20 validation population.

## Confirmatory Safety-Side Results

| Dataset | Episodes | Coverage | Mean max interval width | Mean net utility | Mean audit cost | False Continue | Stop action | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CelebA | 6 | 1.000 | 0.308 | -0.0305 | 99.4 | 0.000 | 1.000 | 1.000 |
| Waterbirds | 6 | 1.000 | 0.417 | -0.0651 | 6.25 | 0.000 | 1.000 | 1.000 |

The negative utility is expected under this safety-first round: the selector
pays for the group audit but does not claim a positive retraining value without
a calibrated action-after-retraining estimator. The result is evidence for
coverage and abstention, not evidence of a useful positive policy.

## Propensity Diagnostic

We also tested a public-only non-uniform propensity proportional to baseline
prediction uncertainty, calibrated to the same mean inclusion probability. The
IPW intervals were vacuous on these rare-group validation populations (maximum
width approximately 1.0), despite empirical coverage of 1.0. This design is
therefore not promoted as the next main method; it identifies the need for a
cross-fitted DR nuisance model or a sharper finite-population bound before
using unequal propensities.

Diagnostic outputs:

- `outputs/public_only_three_action_celeba_next_uncertainty_p05`
- `outputs/public_only_three_action_waterbirds_next_uncertainty_p05`

## Reproduction

```powershell
python scripts/run_public_only_three_action.py `
  --root outputs/ablation_celeba_10seeds `
  --output-dir outputs/public_only_three_action_celeba_next_safe_uniform_p05 `
  --seeds 0 1 2 `
  --noise uniform20 minority_high_40 `
  --coverage-draws 500 `
  --audit-probability 0.50 `
  --audit-design uniform `
  --task-batch-size 256 `
  --audit-batch-size 512 `
  --audit-cost-ratio 0.01

python scripts/run_public_only_three_action.py `
  --root outputs/ablation_waterbirds_10seeds `
  --output-dir outputs/public_only_three_action_waterbirds_next_safe_uniform_p05 `
  --seeds 0 1 2 `
  --noise uniform20 minority_high_40 `
  --coverage-draws 500 `
  --audit-probability 0.50 `
  --audit-design uniform `
  --task-batch-size 32 `
  --audit-batch-size 64 `
  --audit-cost-ratio 0.01
```

## Decision

This round supports retaining the public-only audit and explicit unresolved
state in the paper. It does not support an Oral-level claim yet. The next
required experiment is a pre-fit or cross-fitted estimator for the value of
retraining after verification, with false-Continue, false-Stop, regret, and
coverage reported on held-out seeds.
