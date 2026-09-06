# Claim and evidence ledger

This ledger covers two explicitly separated manuscripts. For the original
diagnostic paper (C1--C7), a result is *confirmatory* only when it comes from
the locked Wave S1 protocol and passes `scripts/p0_accept_server_wave.py`.
For the joint-budget paper (C8 onward), theory is supported by proof plus exact
machine checks, while empirical confirmation is limited to the separately
frozen Phase-10/11 protocols. Older result directories remain exploratory and
must not be silently upgraded.

| ID | Permitted claim | Required evidence | Current status | Prohibited interpretation |
|---|---|---|---|---|
| C1 | A query-time observable proxy does not by itself identify post-retraining worst-group repair. | Wave S1 seed-level WGA, proxy before/after, rank correlations, reversal rate, query IDs, and paired CI. | Pending Wave S1 acceptance. | A mean improvement or a noise-precision result alone establishes the claim. |
| C2 | The mismatch can appear as direction reversal, ranking failure, or semantic mismatch. | Respectively proxy-up/WGA-down rows; Spearman/Kendall across methods; matched latent-cluster witness. | Direction/ranking: pending Wave S1. Semantic witness: exploratory until a fixed construction is accepted. | All three mechanisms occur on every dataset or budget. |
| C3 | Noise precision and WGA repair are dataset and budget dependent. | Three-dataset raw seed table, five budgets, both corruptions, plus the P0 statistics report. | Existing runs are exploratory; confirmation pending. | Noise detection is universally useless for robustness. |
| C4 | Waterbirds can fail at high verification budgets under the locked protocol. | Ten seeds, predeclared method/budget list, negative-seed rate, worst seed, paired CI, and complete artifacts. | Pending Wave S1 acceptance. | A five-seed or selected-budget pilot proves a global safety cap. |
| C5 | CPBA/TC/NGC/Guarded diagnose allocation mechanisms. | Named ablations, exact query IDs, per-seed WGA, and diagnostic metadata. | Exploratory/diagnostic. | Any variant is a universally low-regret or safe policy. |
| C6 | Oracle-GB, Oracle-Disag, and oracle posterior quantify headroom. | Private-manifest analysis with clear oracle labeling. | Diagnostic only. | Oracle performance is deployable without group/private information. |
| C7 | Training-dynamics baseline comparisons are informative. | Baseline integrity report showing epoch support, effective forgetting rate, probability source, and completed seed count. | Pending audit per incoming run. | A five-epoch forgetting or in-sample Cleanlab result is a definitive baseline failure. |
| C8 | Two-action allocation is not uniformly identifiable even with the same test `(X,Y)`, one binary audit, unequal costs, and sequential randomized policies. | General theorem, exact normalized-utility proof, common-marginal support check, and 905-policy finite enumeration. | Complete; `SAME_MARGINAL_BINARY_AUDIT_THEOREM_ZH.md`. | Group audits are always useless, or real datasets attain the lower bound. |
| C9 | The C8 lower bound holds for a uniquely solved strongly convex linear learner and a certified non-orthogonal neighborhood. | L2-logistic embedding, coordinate proof, explicit spectral radius, 64-query/256-WGA exact check, and 256 dense perturbed full-retraining fits. | Complete; `CONVEX_LOGISTIC_EMBEDDING_ZH.md` and `NONORTHOGONAL_STABILITY_ZH.md`. | The result covers arbitrary highly correlated real representations or proves real features lie in the certified ball. |
| C10 | Joint auditing has a reproducible but conditional cost-benefit region. | Frozen Phase-10 180-cell primary grid, ten paired seeds, exact cost audit, full cell table, and Holm report. | Complete; 24 raw common beneficial cells, narrower multiplicity-stable uniform-noise core. | A universal break-even cost, globally optimal 25%/50% share, or a deployable selector. |
| C11 | The development-frozen 2%/0.1/50% cell partially transfers to an unopened tabular dataset. | Frozen ACS source/config/preflight hashes, 20 primary cells, exact action counts, paired intervals, and query artifacts. | Complete: uniform20 +0.405 pp [0.177, 0.632]; minority-high-40 -0.219 pp [-0.484, 0.047]. | Strong cross-noise confirmation or permission to search another ACS cell/dataset. |
| C12 | Better attribute identification and more correct label repairs do not suffice for better WGA. | ACS minority-high-40: weak group identified 10/10, precision 0.760 vs 0.360, +776.7 corrections, but nonpositive strict WGA contrast. | Complete diagnostic supporting the recovery-bound decomposition. | These diagnostics cause the WGA loss or universally predict failure. |
| C13 | Uniform legal-value approximation is sufficient for recoverable allocation: plug-in regret is at most `2E`, and a gap larger than `2E` identifies the true optimum. | Formal assumption, plug-in proof, cost-aware component decomposition, exact-gap and label-only safety corollaries. | Complete theorem statement; `RECOVERY_GUARANTEE_ZH.md`. | The current heuristic consistently estimates every error term or already has a distribution-free safety guarantee. |
| C14 | In the predeclared CelebA uniform20 / 10% condition, a practical high-precision TracIn-CP (val) query set can have negative absolute WGA utility after full ResNet-50 retraining. | Exactly seeds 10--19, same-seed no-correction baselines, saved query IDs/predictions, reproducible training configuration, 20-row accepted aggregate, and all three registered primary gates. | Registered null: only the precision gate passed; the absolute-harm gates failed. Retained as a null in the diagnostic manuscript. | Pooling exploratory seeds 0--9, replacing a failed seed, searching another budget, or using the relative-to-Loss contrast as evidence of absolute harm. |
| C15 | With label verification as the only action, observational equivalence can force arbitrary-budget randomized WGA regret arbitrarily close to 1/2, even under unique strongly convex logistic retraining. | General two-block proof, identical complete test `(X,Y)` marginal, identical sequential label feedback, L2-logistic realization, dense perturbation check, and complete enumeration for B=2,3,4. | Complete; `docs/DIAGNOSTIC_ONE_ACTION_LOWER_BOUND_ZH.md` and `outputs/theory/diagnostic_one_action_lower_bound.json`. | Real representations attain the bound, hidden groups can be changed in deployment, or the result covers the separate group-audit/label-audit allocation problem. |
| C16 | A uniform pairwise set-value stability condition makes the recovery theorem's batch remainder explicit at order B-squared over N-squared. | Telescoping discrete-second-difference proof and machine-checked bound formula. | Complete proposition in the diagnostic manuscript and `robust_verify.diagnostic_theory`. | Current influence scores establish the pairwise stability constant or worst-group switching is negligible. |
| C17 | On the reused CelebA uniform20 / 10% frozen-feature condition, explicitly value-aligned query adaptations can outperform high-precision noise detection, while additional TracIn snapshots do not repair the mismatch. | Complete 50-row seed-by-method grid, saved query IDs, reproducible training configuration, manifest hashes, seed-bootstrap intervals, paired Loss contrasts, and exact sign diagnostics. | Complete but strictly exploratory/post-hoc; `outputs/local_celeba_modern_baselines_uniform20_10pct/audit/REPORT_ZH.md`. | AUTO-D3M-Q is an exact reproduction of AUTO-D3M; the selected methods are confirmatory; seed 0--9 are independent of the previously inspected matrix; or frozen-head gains imply end-to-end gains. |
| C18 | In a prospectively frozen follow-up, RV-Q (tail-product) can improve full-retraining WGA over Loss on fresh CelebA seeds 20--29. | Exactly seeds 20--29, paired RV-Q-minus-Loss Delta-WGA, lower 95% seed-bootstrap endpoint above zero, at least 8/10 positive paired seeds, saved query IDs/predictions, reproducible training configuration, and a verified 40-row aggregate. | Passed in its locked CelebA condition: paired effect +0.986 pp, 9/10 positive seeds. This is condition-specific evidence, not a transfer claim. | The seed 0--9 pilot is confirmation; RV-Q is a universal WGA certificate; absolute utility or group diagnostics substitute for the paired gate; or a failed gate permits another tail fraction, method, or seed search. |
| C19 | RV-Q (tail-product) does not establish a transferable Waterbirds improvement over Loss. | Development-only Waterbirds seeds 20--39, complete per-seed archive, query/prediction replay, and paired comparison against Loss. | Failed: mean RV-Q-minus-Loss Delta WGA is -2.642 pp, with 6/20 positive seeds. The server `scoring.py` differs from the registered archive revision, so this block is diagnostic rather than strict registered confirmation. | Pooling seeds 20--39 with CelebA seeds; treating the RV-Q failure as proof that no value proxy can help; or retuning RV-Q on these seeds. |
| C20 | RV-Q-Gated does not outperform NoiseScore on locked Waterbirds seeds 40--59, despite positive absolute utility. | Frozen 20-seed protocol, same-seed no-correction baselines, exact query IDs/predictions, seed-bootstrap decision, and independent archive replay. | Primary gate failed: RV-Q-Gated-minus-NoiseScore is -1.266 pp [-3.110, +0.336], 7/20 positive; absolute RV-Q-Gated Delta WGA is +4.440 pp [+1.286, +7.647]. Precision noninferiority also failed by 19.688 pp. All prediction/query replays pass. The server `scoring.py` has an inactive NN-cache deviation from the registered package, documented separately. | Claiming RV-Q-Gated is superior, pooling seeds 40--59 with development seeds, retuning the gate on these seeds, or claiming strict sealed-code conformance without the deviation note. |

## Evidence classes

- **Confirmatory:** frozen configuration, all predeclared seeds/noises/budgets,
  raw per-seed records, query IDs, proxy values, reproducible training entry points,
  input/output hashes, and a passing acceptance report.
- **Pilot:** an isolated `p0_pilots/` subset.  It may validate implementation
  and numerical reproducibility but cannot change the locked method list,
  threshold, or main-paper claim.
- **Exploratory:** legacy outputs, post-hoc slices, method searches, or any run
  missing a required raw artifact.  Use only as labeled diagnostics.

## Canonical naming

Code IDs are immutable CSV identifiers.  The corresponding paper/figure names
come exclusively from `robust_verify.scoring.METHOD_LABELS`; regenerate the
registry with:

```bash
python scripts/p0_validate_protocol.py
```

The canonical names relevant to the main diagnostic are `NoiseScore`, `BHC`,
`NGC`, `CPBA-TC`, `CPBA-only`, `TC-only`, `CPBA-TC-noFB`,
`CPBA-TC-v8FB`, `TracIn-CP (multi)`, `RV-Q (tail-product)`,
`RV-Q-Gated`,
`AUTO-D3M-Q (adapt.)`, `Oracle-GB`, and `Oracle-Disag`.  Do not use informal aliases
such as “v7”, “v8 fallback”, “Safe”, or “Guarded” in a table without an
explicit mapping to one of these IDs.

The stable mapping for the two repair-value policies is:

| Paper name | Internal ID | Role |
|---|---|---|
| `RV-Q (tail-product)` | `expected_repair_value` | NoiseScore proxy times normalized tail repair-value rank |
| `RV-Q-Gated` | `noise_gated_repair_value` | NoiseScore anchors plus gated tail repair-value fill |

`CPBA-only`, `CPBA-TC`, `BHC`, and `NGC` are separate allocation/coverage
families, not aliases for either repair-value policy.  Protocol suffixes such
as `-v1` identify a frozen archive revision and are not method names.

## Frozen decisions

- The v8 loss-ratio threshold remains `4.0`; P0 does not tune it.
- Validation checkpoint selection remains the legal `balanced_accuracy`
  setting unless an explicitly labelled oracle diagnostic says otherwise.
- Same-seed budgets are condition labels, not independent samples.  All
  inference uses seed-level paired/bootstrap calculations.
- No test WGA, held-out dataset result, or intermediate server result may be
  used to select a proxy, method, threshold, or budget.
- The local modern-attribution screen reuses CelebA seeds 0--9 and was designed
  after inspecting the main matrix.  It remains exploratory regardless of its
  interval or sign-test values and is never pooled with the held-out server run.
- The independent CelebA end-to-end confirmation is protocol
  `celeba-e2e-tracin-heldout-seeds-10-19-v1`; its server bundle SHA256 is
  `d92a7aa14c3e649e1c1c01527cec56d7c54f71459a460b0dd220ff52cd32a10d`.
  A null result is retained, seeds 0--9 are not pooled, and no formal server
  outcome had been inspected when the registration was written.
- The RepairValue prospective follow-up is protocol
  `celeba-e2e-repairvalue-followup-seeds-20-29-v1`; it discloses the seed 0--9
  frozen-feature result as its hypothesis-generating pilot and uses only fresh
  seeds 20--29.  Its server bundle SHA256 is
  `7a7b86c097b415720e73fe4d3696f6ca3da9aec6c91933b19324cb7cd7bcf71e`.
  It is never pooled with seeds 0--19, and a failed paired gate is retained.
