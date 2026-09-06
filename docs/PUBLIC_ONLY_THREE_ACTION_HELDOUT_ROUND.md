# Public-Only Three-Action Held-Out Round

This is a fixed-rule replication of the exploratory 30% budget upper-bound
diagnostic.  It is intended to test whether the small positive mean observed
on seeds 0--2 generalizes to held-out corruption seeds; it is not a
confirmatory safety claim because the selector still runs with
`allow_uncalibrated_verify=True`.

## Locked protocol

- Dataset: CelebA, using `outputs/ablation_celeba_10seeds`.
- Held-out seeds: 3, 4, 5, 6, 7, 8, 9.
- Noise settings: `uniform20` and `minority_high_40`.
- Verification score: `public_repair_value`.
- Budget fraction: `0.30`.
- Task batch: 256; audit batch: 512.
- Audit design: independent uniform Bernoulli, `pi=0.50`.
- Audit cost ratio: `0.0001`.
- Coverage draws: 200.
- No threshold, batch size, score, or seed selection will be changed after
  inspecting the held-out outcomes.

The private manifests are used only by the purchased-response simulator and
the final blind evaluator.  Results are exploratory and will be reported with
mean utility, per-seed utility, false Continue, coverage, and the number of
positive episodes.  A positive mean without low false Continue is not treated
as evidence for a deployable policy.

## Reproduction

```powershell
python scripts/run_public_only_three_action.py `
  --root outputs/ablation_celeba_10seeds `
  --output-dir outputs/public_only_three_action_celeba_rvq_budget30_heldout_exploratory `
  --seeds 3 4 5 6 7 8 9 `
  --noise uniform20 minority_high_40 `
  --coverage-draws 200 `
  --budget-fraction 0.30 `
  --audit-probability 0.50 `
  --audit-design uniform `
  --task-batch-size 256 `
  --audit-batch-size 512 `
  --audit-cost-ratio 0.0001 `
  --verify-score-mode public_repair_value `
  --allow-uncalibrated-verify
```

## Held-out result

The locked run completed 14 episodes.  Coverage remained 1.000 and the mean
maximum interval width was 0.308.  Mean net utility was `+0.00742`, with 9/14
episodes positive and 5/14 negative.  Four episodes had an actual WGA drop
after Continue, giving a false-Continue rate of `4/14 = 0.286`; the selector
continued in all 14 episodes because this is the explicitly exploratory
uncalibrated mode.  A normal-approximation interval for the episode mean is
approximately `[-0.0115, 0.0264]`, so the aggregate effect is not separated
from zero.

| Noise setting | Mean net utility | Positive net utility | WGA harm |
|---|---:|---:|---:|
| `uniform20` | +0.00478 | 4/7 | 2/7 |
| `minority_high_40` | +0.01006 | 5/7 | 2/7 |

This held-out replication strengthens the case that verification can have a
positive expected value when the verification cost is made extremely small,
but it does not establish a reliable public Continue certificate.  The
production-safe default remains to abstain on an uncalibrated value proxy.
The next experiment should fit a cross-fitted action-value calibrator on a
development set and freeze it before evaluating a new held-out set, reporting
false Continue, false Stop, regret, coverage, and utility intervals.
