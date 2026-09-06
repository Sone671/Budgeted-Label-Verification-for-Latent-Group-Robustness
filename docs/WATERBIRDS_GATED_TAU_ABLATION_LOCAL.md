# Waterbirds RV-Q-Gated frozen-feature tail ablation

This local run covers the disjoint seeds `40--59` under uniform20 noise at a
10% query budget. The gated rule keeps a `0.50` NoiseScore anchor and a `2B`
candidate pool; only the validation-tail fraction varies:

```text
tau = 0.20, 0.30, 0.35, 0.40, 0.50
```

The fixed ResNet-50 feature cache, linear-head optimizer, validation selection
metric, and tail settings follow the completed end-to-end block. This local
ablation does not retrain the backbone. The frozen script writes after every
arm and caches each probe, so it can be interrupted and resumed.

The workstation uses `num_workers=0` because Windows worker startup is much
slower for this image block. The output is kept separate from the archived
end-to-end block and is labeled as a local frozen-feature ablation.

Run the new arms:

```text
.venv\\Scripts\\python.exe scripts\\run_waterbirds_gated_tau_ablation_frozen.py `
  --num-workers 0
```

Generate the appendix table and figure only after all 100 rows are present:

```text
.venv\\Scripts\\python.exe scripts\\analyze_waterbirds_gated_tau_ablation.py
```

The analysis selects the main-text tau by the highest mean frozen-feature
linear-head validation balanced accuracy. Test WGA is used only for the final
sensitivity report, not for tau selection.
