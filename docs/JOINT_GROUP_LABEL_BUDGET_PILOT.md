# Joint group-audit and label-verification budget pilot

## Question

With a fixed human-action budget, can a small number of group-attribute audits
improve which noisy training labels are verified enough to offset the labels
that were not verified?

## Information boundary

- A group audit is performed on the public clean validation pool and reveals
  only the binary `place` attribute.
- A label-verification action is performed on the noisy training pool and
  reveals only the clean class label.
- The audit set is selected class-balanced at random before `place` is opened.
- Test labels, test groups, clean training labels, and unaudited validation
  attributes are not used by the sparse method.
- Full private attributes are used only after query selection for diagnostics
  and in the explicitly named `oracle_attribute` comparator.  This comparator
  keeps the same acquisition formula; it is not a global performance upper
  bound.

For Waterbirds, CelebA, and CivilComments, `place` respectively represents the
spurious background, the configured spurious facial attribute, and the
configured identity indicator.  The evaluated group remains
`group = 2 * clean_label + place`.

## Frozen pilot

- Dataset: Waterbirds frozen ResNet-50 features.
- Total human budget: 2% of the training-set size.
- Group-audit shares: 50% and 75%.
- Integer budgets first preserve the canonical remaining label-budget
  fraction; any one-action rounding remainder is assigned to group auditing.
- Initial seeds: 0, 1, 2 for both `uniform20` and `minority_high_40`.
- Sparse attribute model: class-balanced logistic regression on L2-normalized
  validation features, with a one-class constant-probability fallback.
- Joint query score:

  ```text
  NoiseScore(i) * E[estimated validation group error | x_i]
  ```

No threshold or score parameter is selected using WGA.

## Comparators

1. `label_only_total_budget`: all human actions verify labels.
2. `label_only_same_label_budget`: the same number of label verifications as
   the joint method, isolating the value of sparse group information.
3. `joint_sparse_attribute`: the deployable pilot under the declared boundary.
4. `oracle_attribute`: full train/validation `place` information under the
   same fixed scoring rule.  It diagnoses attribute-estimation error separately
   from value-model error, but is not claimed to be an optimal query policy.

## Interpretation

The primary paired contrast is `joint_sparse_attribute` minus
`label_only_same_label_budget`.  It asks whether group audits add acquisition
value once the number of corrected labels is held to the same cap.

The stricter cost contrast is `joint_sparse_attribute` minus
`label_only_total_budget`.  A positive value means the group audits were useful
enough to compensate for the label verifications they displaced.

Three seeds are a feasibility pilot, not confirmatory evidence.  Expansion is
warranted only if the sparse method improves the primary contrast and improves
the strict cost contrast in a reproducible subset of conditions.  A gap to the
full-attribute comparator diagnoses attribute estimation; failure of that
comparator diagnoses the fixed value model itself.

## Command

```bash
python scripts/joint_group_label_budget_pilot.py \
  --seeds 0,1,2 \
  --noises uniform20,minority_high_40 \
  --total-budget 0.02 \
  --group-shares 0.50,0.75 \
  --include-oracle-attribute
```
