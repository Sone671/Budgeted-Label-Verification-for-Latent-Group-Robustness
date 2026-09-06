# Baseline supplement plan for main_iclr2027

Priority: P0 = required before submission; P1 = strongly recommended; P2 = optional diagnostic.

## 1. Baseline list

| Priority | Baseline | Category | Information used | Role |
|---|---|---|---|---|
| P0 | ERM (noisy labels, zero query) | Model reference | Noisy train labels | Absolute no-query reference |
| P0 | Oracle GroupDRO | Model reference | Private train/val groups | Oracle group-training upper bound |
| P0 | ActiveClean-style expected model change | Active cleaning | Noisy labels + current model | Mainstream active-cleaning system |
| P0 | Active Label Cleaning (Bernhardt et al.) | Active cleaning | CE loss + prediction entropy | Mainstream active-cleaning system |
| P0 | OOF Cleanlab | Noise detection | Out-of-fold predicted probabilities | Replaces in-sample Cleanlab |
| P1 | AUTO-D3M query-action adaptation | Attribution/group selection | Frozen features + public val labels | Strong data-centric subgroup baseline |
| P1 | SELF-style last-layer selection | Subgroup data selection | Public val labels; oracle variant if val groups are used | Strong data-centric subgroup baseline |
| P1 | TRAK-style attribution query | Attribution | Checkpoint gradients | Strong modern attribution baseline |
| P1 | Full BADGE | Active learning | Gradient embeddings | Removes BADGE-lite caveat |
| P1 | JTT error-set query | Group-robust/noisy hybrid | First-stage JTT errors | Tests JTT errors as acquisition scores |
| P2 | Oracle repair-value query | Oracle diagnostic | Private clean labels + retraining | Upper bound for repair-value alignment |
| P2 | JTT training baseline | Failure diagnostic | Noisy labels, first-stage errors | Shows naive JTT failure under label noise |
| P2 | Full fine-tuning ERM / GroupDRO | End-to-end reference | Raw images / full backbone | Stronger external validity |
| P2 | KAIROS data valuation | Data valuation | Validation loss trajectories | Broad data-utility reference |

Existing baselines that stay unchanged: Random, Loss, Entropy, Forgetting, AUM, in-sample Cleanlab, Margin, Core-set-lite, Uncertainty-Diversity, BADGE-lite, TracIn variants, NoiseScore and CPBA variants, oracle noise/group queries, CIFAR-10N replay methods.

## 2. Datasets, noise, seeds, budgets

Common frozen protocol: Waterbirds (WB), CelebA (CE), CivilComments (CC); noise = uniform20 and minority-high-40; outer corruption seeds = 0-9; budgets = 0.5%, 1%, 2%, 5%, 10%; frozen features; linear-head retraining.

### 2.1 P0 mandatory runs

| Baseline | Datasets | Noise | Seeds | Budgets | Runs |
|---|---|---|---|---|---|
| ERM no-query | WB, CE, CC | 2 | 0-9 | none | 60 |
| Oracle GroupDRO | WB, CE, CC | 2 | 0-9 | none | 60 |
| ActiveClean-style | WB, CE, CC | 2 | 0-9 | 5 | 300 |
| Active Label Cleaning | WB, CE, CC | 2 | 0-9 | 5 | 300 |
| OOF Cleanlab | WB, CE, CC | 2 | 0-9 | 5 | 300 |

OOF Cleanlab needs one out-of-fold probability vector per corruption seed. Use the same outer seeds 0-9 and a fixed five-fold partition per dataset; do not mix folds across seeds.

### 2.2 P1 strong baselines

| Baseline | Datasets | Noise | Seeds | Budgets | Runs |
|---|---|---|---|---|---|
| AUTO-D3M adaptation | WB, CE full; CC minimum | 2 | 0-9 | WB/CE all 5; CC at least 2% | 200-220 |
| SELF-style | WB, CE full; CC minimum | 2 | 0-9 | WB/CE all 5; CC at least 2% | 200-220 |
| TRAK-style | WB, CE full; CC minimum | 2 | 0-9 | WB/CE all 5; CC at least 2% | 200-220 |
| Full BADGE | WB, CE full; CC minimum | 2 | 0-9 | WB/CE all 5; CC at least 2% | 200-220 |
| JTT error-set query | WB, CE, CC | 2 | 0-9 | all 5 | 300 |

For CC, the minimum accepted condition is uniform20 and minority-high-40 at 2% budget (20 runs per method). The full CC five-budget grid is recommended if compute allows.

### 2.3 P2 optional runs

| Baseline | Datasets | Noise | Seeds | Budgets | Runs |
|---|---|---|---|---|---|
| Oracle repair-value query | CE and WB | uniform20 | 0-9 | 2% or 10% | 20 |
| JTT training baseline | WB, CE, CC | 2 | 0-9 | none | 60 |
| Full fine-tuning ERM/GroupDRO | WB, CE | uniform20 | 10-19 and 20-29 | 10% | 40 |
| KAIROS | WB or CE | uniform20 | 0-9 | 2% | 10 |

Oracle repair-value is analysis-only and must not be pooled with legal baselines.

## 3. End-to-end full-retraining placement

New P0 query baselines should also appear in the existing full-retraining blocks so the frozen result is not the only evidence.

| End-to-end block | Condition | Seeds | Add these new methods |
|---|---|---|---|
| CelebA RepairValue block | CE uniform20, 10% | 20-29 | ActiveClean-style, Active Label Cleaning, OOF Cleanlab |
| CelebA TracIn block | CE uniform20, 10% | 10-19 | optional: Active Label Cleaning only |
| Waterbirds v1 transfer block | WB uniform20, 10% | 20-39 | ActiveClean-style, Active Label Cleaning, OOF Cleanlab |
| Waterbirds v2 gated block | WB uniform20, 10% | 40-59 | optional: report only if already generated |
| Waterbirds mechanism block | WB uniform20, 10% | 10-19 | no new methods required |

Do not add new end-to-end seed blocks. Reuse the existing seed partitions above.

## 4. What not to run

- No new baselines on CIFAR-10N; keep it as a fixed-realization exploratory scope check.
- No new datasets.
- Do not reuse frozen seeds 0-9 as confirmatory evidence for a new method; any new method must pass the same disjoint end-to-end block as RepairValue-Q before being called confirmatory.
