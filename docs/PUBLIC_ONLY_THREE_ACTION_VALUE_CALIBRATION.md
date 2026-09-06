# Public-Only Value Calibration Round

This round compares the original public noise score with a public-only
reliability-gated repair-value score. The latter is built from training
features, noisy public labels, public validation features/labels, and public
validation probabilities. It uses validation-fold rank stability and noise
retention as a gate; it never reads clean training labels, latent groups, or
private test outcomes during selection.

The selector exposes this through `--verify-score-mode`:

- `noise`: original predicted-error ranking;
- `public_repair_value`: reliability-gated public repair-value ranking.

The production-safe default remains `--allow-uncalibrated-verify` **off**.
The exploratory comparison below enables it explicitly so that false Continue
can be measured against blind private WGA.

## Controlled Exploratory Comparison

Both CelebA runs use the same six episodes (two noise settings, seeds 0--2),
uniform independent Bernoulli audit with `pi=0.5`, audit cost ratio `0.01`,
500-sized verification budget, and 200 coverage draws.

| Verify score | Coverage | Mean max width | Mean net utility | False Continue | Stop action |
|---|---:|---:|---:|---:|---:|
| Noise | 1.000 | 0.308 | -0.0494 | 1/6 = 0.167 | 0.833 |
| Public repair value | 1.000 | 0.308 | -0.0305 | 0/6 = 0.000 | 1.000 |

The noise run's single Continue was harmful: final WGA fell from 0.7192 to
0.6845 on the corresponding CelebA episode. The public repair-value run did
not Continue in any episode, so it avoided that error but did not yet produce
positive absolute utility.

Waterbirds produced no Continue under either score in the six-episode
exploratory run; mean net utility was -0.0651 and coverage was 1.000.

## Interpretation

This is evidence that the public repair-value gate is a safer ranking signal
than raw noise score in this pilot, not evidence that it is a calibrated
retraining-value estimator. The conservative behavior is currently driven by
the stability/retention gate and the wide action intervals. It must be tested
on held-out seeds and additional datasets before being promoted into the main
paper claim.

Outputs:

- `outputs/public_only_three_action_celeba_noise_exploratory`
- `outputs/public_only_three_action_celeba_rvq_exploratory`
- `outputs/public_only_three_action_waterbirds_noise_exploratory`
- `outputs/public_only_three_action_waterbirds_rvq_exploratory`

The next required step is an explicitly pre-registered cross-fitted action
value calibration study: fit the public value model on one set of episodes,
freeze it, and evaluate Continue/Stop decisions on held-out seeds with false
Continue, false Stop, regret, coverage, and utility confidence intervals.

## Budget and Batch Upper-Bound Diagnostics

The following runs deliberately enable `allow_uncalibrated_verify` and are
therefore exploratory only.  They use the public repair-value score on CelebA,
uniform Bernoulli auditing, six episodes (two noise settings and seeds 0--2),
and should not be read as a calibrated deployment result.

| Budget fraction | Mean net utility | False Continue | Stop | Mean audit+task cost |
|---:|---:|---:|---:|---:|
| 0.10 | -0.00498 | 2/6 = 0.333 | 0/6 | 256.994 |
| 0.30 | **+0.00554** | 2/6 = 0.333 | 0/6 | 256.994 |

The 30% setting is a useful cost-sensitivity diagnostic: four of six
episodes have positive net utility, but the two seed-1 episodes still lose
WGA after Continue.  Thus the positive mean is not evidence for a safe
policy, and it does not justify changing the production default.

For a fixed CelebA `uniform20`, seed-2 diagnostic at the 2% budget, changing
the task batch size did not stabilize the decision:

| Task batch | Baseline WGA | Final WGA | Net utility | False Continue |
|---:|---:|---:|---:|---:|
| 32 | 0.7192 | 0.6877 | -0.0445 | yes |
| 64 | 0.7192 | 0.6940 | -0.0480 | yes |
| 256 | 0.7192 | 0.7413 | -0.0596 | no |

The larger batch can recover WGA on this episode but pays more verification
cost; the smaller batches are harmful.  These results reinforce the need for
cross-fitted action-value calibration and held-out seeds before selecting a
batch or promoting Continue.
