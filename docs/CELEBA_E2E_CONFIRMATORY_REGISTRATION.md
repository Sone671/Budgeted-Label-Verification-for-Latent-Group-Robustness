# CelebA held-out end-to-end confirmation registration

Registered on 2026-07-31 before any formal seed 10--19 end-to-end result was
generated or inspected. The machine-readable source of truth is
`configs/celeba_e2e_confirmatory_seed10_19.registration.json`.

## Frozen question

For CelebA `Eyeglasses` under uniform 20% label corruption and a 10% label
verification budget, does the fully practical single-best-checkpoint
TracIn-CP (val) query rule have negative absolute WGA utility after full
ResNet-50 retraining, despite at least 99% mean query noise precision?

The independent outer corruption seeds are exactly 10--19. Loss is the
seed-paired query baseline, while the absolute comparator is a separately
exported same-seed no-correction model. Results from exploratory seeds 0--9
are not pooled with this run.

## Primary gate

All three conditions must hold:

1. the upper endpoint of the deterministic 10,000-replicate seed-bootstrap
   95% interval for mean absolute Delta-WGA is below zero;
2. at least 8 of 10 seed effects are negative;
3. mean TracIn query noise precision is at least 99%.

The paired TracIn-minus-Loss contrast is secondary. Failure of the primary
gate is a valid null result; it does not permit seed replacement, budget
search, epoch changes, or a new candidate under this protocol version.

## Frozen artifacts

| Artifact | SHA256 |
| --- | --- |
| Protocol payload | `45d903535a23475c4de61be2d5f466d526ab73c46ab2f9d68d58407ae71c849f` |
| YAML config | `a23aa125211203740c171bac06b260de8082a9264a33692d9c147334507101d4` |
| Server bundle | `d92a7aa14c3e649e1c1c01527cec56d7c54f71459a460b0dd220ff52cd32a10d` |

The server launcher validates the scientific fields, creates one immutable
attempt directory per seed, saves query IDs before opening the verification
oracle, persists test predictions and replay checkpoints, and recomputes all
reported metrics during aggregation.

## Local-only work boundary

Local work may test code, generate deterministic Stage-0 manifests, prove or
machine-check theory, implement diagnostic frozen-feature baselines, and
prepare analysis templates. It may not fabricate, impute, or preview the
missing end-to-end seed 10--19 outcomes. Only a fully accepted 20-row server
aggregate may populate the confirmatory paper table.

