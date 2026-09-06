# Paper Tables — Data Summary

## Main Tables

| File | Paper Ref | Content |
|------|-----------|---------|
| `table1_main_results_overall.csv` | Table 1 | Main results: all budgets + both noises pooled, per dataset + overall |
| `table2_stress_test.csv` | Table 2 | Stress test: uniform20 b=5%, per-method dWGA + noise_precision |
| `table3_disagreement.csv` | Table 3 | CivilComments real-noise disagreement: dWGA + disagreement_lift @ b=5% |
| `table3_noise_stats.csv` | — | Noise statistics: actual noise rates for 3 noise types |
| `table4_mechanism.csv` | Table 4 | Mechanism analysis: Spearman rho of 3 features vs dWGA |
| `table4_prong1_precision.csv` | — | Mechanism prong 1: noise_precision vs dWGA (all datasets) |
| `table4_prong2_disagreement.csv` | — | Mechanism prong 2: disagreement_lift vs dWGA (CC real-noise) |
| `table4_prong3_coverage.csv` | — | Mechanism prong 3: minority_q / group_entropy vs dWGA |

## Appendix Tables

| File | Content |
|------|---------|
| `appendix_ablation_attribution.csv` | Ablation: CPBA component attribution (Waterbirds + CivilComments) |
| `appendix_model_baselines.csv` | Model-training baselines: ERM / GroupDRO / JTT WGA |
| `appendix_paired_tests.csv` | Full paired significance tests (all datasets, noises, budgets) |
| `appendix_query_baselines_full.csv` | Full query method comparison: per-budget, per-noise, per-dataset |
| `appendix_query_baselines_summary.csv` | Budget-pooled summary: per-noise, per-dataset |

## Completed server-results audit (2026-07-24 archive)

These assets are generated from the supplied `completed_results_final.zip`, not
from budget-pooled aggregates. The source SHA256, input filenames, and output
counts are stored in `completed_results_matrix_manifest.json`.

```powershell
python scripts/build_completed_results_paper_assets.py \
  --archive /path/to/completed_results_final.zip \
  --paper-dir paper
```

| File | Content |
|------|---------|
| `table_practical_absolute_counterexample.{csv,tex}` | CelebA uniform20 / 10% practical TracIn-CP(val) absolute-harm contrast. |
| `appendix_s2_complete_matrix.csv` | All 580 S2 raw method/noise/budget cells, with absolute and paired-vs-Loss summaries. |
| `appendix_s2_{dataset}_{noise}.tex` | Six complete S2 matrices used by the Chinese diagnostic appendix. |
| `appendix_s2_reversal_coverage.csv` | Matrix-coverage counts for fewer-corrections/higher-WGA reversals. |
| `appendix_s3_multibudget.{csv,tex}` | CivilComments S3 spectral and feedback results for 2%, 5%, and 10%. |
| `appendix_celeba_e2e_audit.tex` | Complete four-method CelebA ResNet-50 end-to-end audit at uniform20 / 2% / 10 seeds. |

The CelebA end-to-end table is transcribed from
`completed_results_final_new.zip` entry
`results/13_CelebA_e2e_confirmatory.csv` and cross-checked with
`audit_results.zip/audit_results/celeba_e2e/`. Its absolute comparator is the
per-seed exported `baseline_wga`; `Random` is a query method, not no correction.

## Figures

| File | Paper Ref | Content |
|------|-----------|---------|
| `../figures/fig1_noise_detection_vs_wga.png` | Figure 1 | Hook: noise precision vs dWGA |
| `../figures/fig2_budget_curves.png` | Figure 2 | Budget curves (uniform20) |
| `../figures/fig3_ablation_attribution.png` | Figure 3 | Ablation component attribution |
| `../figures/fig4_fallback_analysis.png` | Figure 4 | Fallback failure analysis |
| `../figures_diagnostic/fig2_practical_absolute_mismatch.pdf` | Diagnostic Figure 2 | Fully practical detection/coverage/WGA mismatch. |
| `../figures_diagnostic/fig3_s2_reversal_coverage.pdf` | Appendix Figure | Full S2 frozen-matrix coverage heatmap. |
