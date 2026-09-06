# Pre-existing baseline results audit

Generated 2026-08-21. Scope: identify local baseline results that were run before the current locked p0 protocol, check whether they are usable for main_iclr2027.tex.

## 1. Asset inventory

| Asset | Path | Status |
|---|---|---|
| Old aggregated model baselines (ERM / Oracle GroupDRO / JTT) | paper/tables/appendix_model_baselines.csv | Complete, 18 cells, 10 seeds each |
| Old raw model baselines | outputs/baselines_{waterbirds,celeba,civilcomments}*/stage1/model_baselines.csv | Complete |
| Old query-baseline comparison | outputs/baseline_comparison/query_baselines_full.csv | Superseded by final p0 matrix |
| Old 3-dataset model summary | paper/tables/table_model_baselines_3datasets.csv | Older pooled summary, do not use |
| Current locked frozen audit (WB/CE) | outputs/p0_wave_s1/{waterbirds,celeba}/stage1/results.csv | Matches Table tab:frozen exactly at 2% |
| Current CivilComments frozen audit | outputs/p0_30epoch_civilcomments_10seed/stage1/results_partial.csv | Incomplete locally (seed 0 only) |
| Final 580-cell matrix | paper/tables/appendix_s2_complete_matrix.csv | Authoritative paper table |

## 2. Old model-baseline numbers (10 seeds, frozen linear head, noisy labels, zero query)

WGA mean (sem):

| Dataset | Noise | ERM | Oracle GroupDRO | JTT |
|---|---|---|---|---|
| Waterbirds | uniform20 | 0.6167 (0.0102) | 0.8251 (0.0082) | 0.0573 (0.0191) |
| Waterbirds | minority-high-40 | 0.4862 (0.0189) | 0.6166 (0.0189) | 0.3809 (0.0129) |
| CelebA | uniform20 | 0.6964 (0.0065) | 0.9096 (0.0025) | 0.0004 (0.0001) |
| CelebA | minority-high-40 | 0.6211 (0.0026) | 0.7981 (0.0105) | 0.2186 (0.0272) |
| CivilComments | uniform20 | 0.0736 (0.0015) | 0.6721 (0.0026) | 0.0001 (0.0000) |
| CivilComments | minority-high-40 | 0.0514 (0.0009) | 0.6493 (0.0043) | 0.0141 (0.0004) |

JTT is not a competitive baseline under this protocol: for uniform20 CelebA/CivilComments its best epoch is frequently 0, and it upweights 35k-72k first-stage errors.

## 3. Consistency check against the current locked baseline

The old ERM values do not match the current no-correction WGA baseline used by p0.

| Dataset | Noise | Current no-correction WGA | Old ERM WGA | Gap |
|---|---|---|---|---|
| Waterbirds | minority-high-40 | 0.4860 | 0.4862 | +0.0002 |
| Waterbirds | uniform20 | 0.6115 | 0.6167 | +0.0052 |
| CelebA | minority-high-40 | 0.6991 | 0.6211 | -0.0780 |
| CelebA | uniform20 | 0.7628 | 0.6964 | -0.0664 |
| CivilComments | uniform20 (seed 0) | 0.1165 | 0.0736 | -0.0429 |

Conclusion: the old model-baseline runs use an earlier probe-selection/code path and are not interchangeable with the current no-correction reference. The same applies to the old query_baselines_full.csv; its Waterbirds values differ from the final matrix by up to 0.13 pp (e.g., uniform20 10% NoiseScore: old -0.1225 vs p0 +0.0036).

## 4. Pilot rerun under current p0 code (Waterbirds seed 0)

A one-seed rerun using current `model_baselines.py` and the current locked p0 Waterbirds probes gives:

| Noise | Method | Current pilot WGA | Old WGA |
|---|---|---|---|
| uniform20 | ERM | 0.5000 | 0.5826 |
| uniform20 | Oracle GroupDRO | 0.7858 | 0.7849 |
| uniform20 | JTT | 0.0745 | 0.0745 |
| minority-high-40 | ERM | 0.4891 | 0.4891 |
| minority-high-40 | Oracle GroupDRO | 0.5607 | 0.5607 |
| minority-high-40 | JTT | 0.4065 | 0.4081 |

This confirms that ERM must be re-anchored to the current probe; Oracle GroupDRO and JTT are closer but should still be rerun under the locked protocol before publication.

## 5. Recommendation

1. Do NOT merge `baseline_comparison/query_baselines_*.csv` or `table_model_baselines_3datasets.csv` into the current paper.
2. Keep `appendix_s2_complete_matrix.csv` and the p0 wave-S1 results as the only frozen query audit.
3. The old model-baseline table can be cited only as an exploratory appendix with an explicit "earlier probe selection" caveat, or better, rerun all three datasets with the current locked p0 config.
4. If added to the main paper, report the current no-correction WGA from p0, not the old ERM column, and label Oracle GroupDRO as an oracle upper bound. Report JTT only as a diagnostic that naive group-robust retraining collapses under uniform label noise, not as a competitive method.
5. Full CivilComments current baselines are not complete locally; the final matrix was built from the server archive, so any new absolute-performance table for CivilComments should be sourced from that archive rather than old local runs.
