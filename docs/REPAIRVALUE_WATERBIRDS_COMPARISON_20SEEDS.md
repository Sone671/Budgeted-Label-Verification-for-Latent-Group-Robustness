# Waterbirds RepairValue-Q：20-seed 扩展结果

这是 Waterbirds RV-Q 冻结特征比较的 20-seed 扩展结果记录。完整的数据来源、
字段字典、算法定义、信息边界和统计方法见
[REPAIRVALUE_WATERBIRDS_COMPARISON.md](REPAIRVALUE_WATERBIRDS_COMPARISON.md)；
本文件只记录本次 0–19 seed 扩展的运行状态和新统计结果。

## 1. 证据定位

本扩展仍属于**探索性 frozen-feature linear-head comparison**：

- 使用 ImageNet 预训练 ResNet-50 的缓存 2048 维特征；
- 不执行 full-network ResNet-50 training；
- 不修改已注册的 Waterbirds/CelebA end-to-end protocol；
- `tau=0.50` 是此前探索网格中的候选值，不是本次扩展后重新选择的参数；
- 新 seed 与旧 seed 合并后可增强探索性精度，但不能单独宣称为预注册确认性结果。

扩展的科学问题是：在固定相同协议下，RV-Q 的条件性收益是否在更多 corruption
seed 上保持。新增的 seed 为 10–19；结果包同时重算并保留 0–9，因此所有汇总
均为完整 20-seed 配对统计。

## 2. 运行矩阵与完整性

| 维度 | 取值 |
|---|---|
| seeds | 0–19，共 20 个 |
| 噪声 | `uniform20`, `minority_high_40` |
| 预算 | 2%（96 个样本）、5%（240 个样本） |
| RV-Q tail fraction | `tau=0.20`, `tau=0.50` |
| validation fraction | 1.0 |
| validation noise | 0.0，mode=`symmetric` |
| 方法 | `loss`, `noise_score`, `entropy`, `expected_repair_value` |
| probe training | 5 epochs，来自新建 20-seed Stage 1 输出 |
| comparison retraining | 20 epochs，batch 4096，lr 0.001，wd 0.0001，patience 5 |
| checkpoint metric | validation balanced accuracy |
| bootstrap | 10,000 次，以 seed 为重采样单位 |

完整性检查：

- Stage 1 原始结果：240 行（40 个 `(noise, seed)` 组合 × 3 方法 × 2 预算）；
- comparison `per_seed_comparison.csv`：400 行；
- comparison `comparison_summary.csv`：8 行；
- 方法计数：RV-Q 160 行，Loss/NoiseScore/Entropy 各 80 行；
- seed 集合恰为 `0,1,...,19`；
- 与原 10-seed 结果逐字段对齐的 200 行完全一致（最大绝对差为 0）。

## 3. 20-seed 主结果（`tau=0.50`）

数值直接来自 `comparison_summary.csv`。均值和区间单位为百分点（pp）；
绝对列是 RV-Q 相对于无修正基线，paired 列是 RV-Q 相对于对应 reference。

| 噪声 | 预算 | RV-Q vs 无修正 | RV-Q vs Loss | RV-Q vs NoiseScore | RV-Q vs Entropy | 查询噪声精度 | minority 查询率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minority_high_40 | 2% | +3.88 [+0.33, +7.20] (+16/20) | +9.21 [+6.36, +12.07] (+19/20, p=.00004) | -1.92 [-5.81, +1.10] (+7/20, p=.481) | — | 25.26% | 16.20% |
| minority_high_40 | 5% | +8.51 [+5.96, +11.20] (+18/20) | +16.71 [+13.45, +20.03] (+20/20, p=.000002) | -1.21 [-3.92, +1.40] (+8/20, p=.648) | — | 16.40% | 8.58% |
| uniform20 | 2% | +6.72 [+4.63, +9.13] (+19/20) | +5.62 [+2.14, +8.61] (+18/20, p=.00040) | +5.12 [+1.74, +8.00] (+17/20, p=.00258) | — | 35.52% | 10.31% |
| uniform20 | 5% | +7.06 [+4.50, +9.63] (+18/20) | +9.56 [+6.76, +12.13] (+18/20, p=.00040) | +9.85 [+7.24, +12.37] (+17/20, p=.00258) | — | 28.69% | 6.15% |

`comparison_summary.csv` 也计算了 RV-Q vs Entropy；为保持主表紧凑，本表省略该列。
该列以及所有未四舍五入值均应以 CSV 为准。

核心解读：

- `tau=0.50` 下，RV-Q 相对无修正基线在四个条件均为正，且 95% bootstrap CI
  均高于 0；这是 10-seed 结果扩展到 20 seed 后最稳定的结论。
- RV-Q 相对 Loss 在四个条件均为正，20 seed 下区间均高于 0。
- RV-Q 相对 NoiseScore 的优势只出现在 `uniform20`；在
  `minority_high_40` 下点估计仍为负，但 20 seed 区间跨 0，因此应写成
  “不稳定/不支配 NoiseScore”，而不是“显著劣于 NoiseScore”。

## 4. `tau=0.20` 敏感性结果

| 噪声 | 预算 | RV-Q vs 无修正 | RV-Q vs Loss | RV-Q vs NoiseScore |
|---|---:|---:|---:|---:|
| minority_high_40 | 2% | -0.55 [-3.07, +1.50] (+15/20) | +4.78 [+1.29, +8.32] (+18/20, p=.00040) | -6.34 [-9.96, -3.14] (+3/20, p=.00258) |
| minority_high_40 | 5% | +1.19 [-1.75, +3.64] (+16/20) | +9.38 [+7.09, +11.80] (+19/20, p=.00004) | -8.53 [-11.24, -5.96] (+2/20, p=.00040) |
| uniform20 | 2% | +2.76 [+1.30, +4.89] (+20/20) | +1.65 [+0.19, +3.08] (+14/20, p=.115) | +1.16 [-1.27, +3.44] (+12/20, p=.503) |
| uniform20 | 5% | +1.81 [-1.04, +4.88] (+15/20) | +4.32 [+0.56, +7.68] (+16/20, p=.012) | +4.61 [+0.15, +8.80] (+15/20, p=.041) |

这组结果说明 `tau=0.50` 的收益不是“任意 tail fraction 都成立”：在
`minority_high_40` 中，`tau=0.20` 明显更容易落后 NoiseScore；因此主文应把
tail fraction 作为条件变量和局限性，而不是隐藏该敏感性。

## 5. 噪声生成统计

| 噪声设置 | 20 seed 平均 noisy 数 | noisy 数 SD | 平均 noisy-minority 数 | noisy-minority 数 SD |
|---|---:|---:|---:|---:|
| uniform20 | 959.7 / 4,795 | 31.90 | 47.8 | 6.19 |
| minority_high_40 | 551.6 / 4,795 | 25.48 | 95.6 | 6.77 |

这些是 private manifest 中的噪声实现统计；不是查询精度，也没有被用于排序。

## 6. 结果文件与哈希

结果目录：`outputs/local_waterbirds_repairvalue_comparison_20seeds/`

| 文件 | 作用 | SHA-256 |
|---|---|---|
| `per_seed_comparison.csv` | 400 条逐 seed/method/condition 记录 | `e523dabf8a4d94d4be48154289cfd1ce1a39c7822fe7fb58bc991b4ac8c1f817` |
| `comparison_summary.csv` | 8 条 20-seed 汇总记录 | `6cd7bddf8bed57b158b7b373e299843cba931a963f7973c03e84fa4dbeef5761` |
| `protocol.json` | 20-seed 协议与 122 个输入 fingerprint | `60fbbc3f1d5f8c10f12b73b894ce57a44424a815f04b47aa74a753569f077eea` |
| `STATUS.md` | 探索性证据边界 | `8d39b1250537ccd55f100b8494aefbb572b9abef370fc4b32b0a23fbaa73a12b` |

输入特征和配置：

- train features SHA-256：`879afca289ab0e609d906bdf490dc8afd76d8ce00002606c90f53fdb55314edf`；
- val features SHA-256：`c012bc704adabce86609f47b29d65ed88b1b09b8ce671aaf81db79e6e88b3faf`；
- test features SHA-256：`16898a9d15e862f2bfcf2c0ac2391a6d8f89dcadeaca7a42c23e2cb79a92c69f`；
- 20-seed config SHA-256：`68138f2f7233e89a544b5b702fa4fbc459de7e3a2edba49ba82f180e3f1b388b`。

注意：`protocol.json` 的 122-file fingerprint 包含 train/val features、40 组
public/private train manifest 和 40 个 probe dynamics，但 runner 当前没有把
`test.npz` 放入 JSON fingerprint；上面单独记录了 test feature hash。正式确认性
artifact 应在 runner 中纳入该文件后重新生成协议。

## 7. 复现命令

### 7.1 生成 20-seed manifests（特征缓存可复用）

```powershell
python -m robust_verify.cli.stage0 --config configs/ablation_waterbirds_20seeds_rvq.yaml
Copy-Item outputs/ablation_waterbirds_10seeds/features/train.npz outputs/ablation_waterbirds_20seeds/features/train.npz
Copy-Item outputs/ablation_waterbirds_10seeds/features/val.npz outputs/ablation_waterbirds_20seeds/features/val.npz
Copy-Item outputs/ablation_waterbirds_10seeds/features/test.npz outputs/ablation_waterbirds_20seeds/features/test.npz
```

### 7.2 生成 probe dynamics

```powershell
python -m robust_verify.cli.stage1 --config configs/ablation_waterbirds_20seeds_rvq.yaml
```

### 7.3 运行四方法比较

```powershell
python scripts/run_waterbirds_repairvalue_comparison.py `
  --input-dir outputs/ablation_waterbirds_20seeds `
  --probe-dir outputs/ablation_waterbirds_20seeds/stage1/probes `
  --output-dir outputs/local_waterbirds_repairvalue_comparison_20seeds `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --noise-names uniform20 minority_high_40 `
  --budgets 0.02 0.05 `
  --taus 0.20 0.50 `
  --validation-fractions 1.0 `
  --validation-noise-rates 0.0 `
  --methods loss noise_score entropy expected_repair_value `
  --epochs 20 --batch-size 4096 --device auto
```

## 8. 论文使用建议

20 seed 结果足以把主文表述从“10-seed exploratory feasibility”提升为：

> 在固定 Waterbirds frozen-feature linear-head protocol 的 20 个 corruption
> seeds 中，RV-Q 在 `tau=0.50` 下相对无修正基线和 Loss 呈现一致的正向 WGA
> 改善；该改善依赖噪声结构和 tail fraction，并不保证优于 NoiseScore。

仍不建议写成：

- RV-Q 普遍优于所有基线；
- RV-Q 已完成 Waterbirds full-network transfer；
- 20 seed bootstrap 等同于跨数据集泛化证明；
- `tau=0.50` 是预注册或全局最优参数。

因此，20-seed 包适合放在主文的“conditional feasibility / sensitivity”结果
以及补充材料的完整数据表中；若要升级为确认性主张，仍需预先注册独立协议，
并进行 full-network 或新的 held-out dataset 验证。
