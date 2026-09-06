# Waterbirds RepairValue-Q 完整数据文档

本文件是 `outputs/local_waterbirds_repairvalue_comparison/` 结果包的
数据卡（data card）和复现实验记录。它描述数据来源、信息边界、方法、
训练协议、统计量、逐行字段和可以写入论文的结论。结果包对应的是
**探索性的 Waterbirds 冻结特征线性头比较**，不是新的确认性实验，也不是
端到端 ResNet-50 迁移成功的证明。

20-seed 扩展结果（seeds 0–19）已单独生成，见
[REPAIRVALUE_WATERBIRDS_COMPARISON_20SEEDS.md](REPAIRVALUE_WATERBIRDS_COMPARISON_20SEEDS.md)
及 `outputs/local_waterbirds_repairvalue_comparison_20seeds/`。本文件保留原始
10-seed 结果包的字段和审计记录，不应与 20-seed 汇总混用。

## 1. 结论先行与证据状态

在相同的查询预算、相同的重训练流程和 10 个外层噪声 seed 下，RV-Q
(`expected_repair_value`) 有一个清晰但有条件的可行区域：

- `tau=0.50` 时，RV-Q 相对于无修正基线在四个
  `(noise, budget)` 条件的平均 ΔWGA 分别为 `+3.57/+6.81 pp`
  （`minority_high_40`, 2%/5%）和 `+8.17/+9.30 pp`
  （`uniform20`, 2%/5%）。
- 在四个 `tau=0.50` 条件下，RV-Q 相对于 `Loss` 的均值差异为
  `+8.24/+14.42/+5.04/+9.25 pp`（顺序同上），其中三个条件的
  95% seed bootstrap 区间完全高于 0；`uniform20, 2%` 的区间跨 0。
- RV-Q **不支配** `NoiseScore`：在 `minority_high_40` 下，RV-Q 相对于
  `NoiseScore` 为 `-2.31 pp`（2%）和 `-3.58 pp`（5%），区间均低于 0。

因此本数据包支持的定位是：

> 一个利用验证集尾部损失和噪声后验代理的、面向预算约束标签修复的
> allocation-aware / conditional RepairValue-Q 实例。

它不支持“RV-Q 在 Waterbirds 上普遍优于所有方法”“已完成端到端迁移”或
“替代已注册 Waterbirds transfer block”的表述。`tau=0.50` 来自本次探索性
敏感性网格，不是预注册的超参数选择；不能把这些行写成确认性结果。

## 2. 数据来源与版本

| 项目 | 记录 |
|---|---|
| 数据集 | Waterbirds，WILDS v1.0 组织方式 |
| 本地数据根目录 | `data/waterbirds_v1.0` |
| 原始获取方式 | `wilds.get_dataset("waterbirds", download=True)`；本包只读取已生成的 manifest 和特征缓存 |
| split 编码 | train=0，val=1，test=2（见 `configs/ablation_waterbirds_10seeds.yaml`） |
| group 定义 | `group = 2 * label + place`；minority 为 `label != place` |
| 训练集 | 4,795；minority 240（5.0%） |
| 验证集 | 1,199；公开标签，敏感性主条件使用全部验证集 |
| 测试集 | 5,794；group 计数为 2,255、2,255、642、642；minority 合计 2,897（50%） |
| 特征 | ImageNet 预训练 ResNet-50 的冻结表征，2048 维；train/val/test 分别为 `(4795,2048)`, `(1199,2048)`, `(5794,2048)` |
| 代码配置 | `configs/ablation_waterbirds_10seeds.yaml` |

训练集噪声由 seed-specific private manifest 记录。配置目标为：

- `uniform20`：所有训练样本以 0.20 概率翻转标签；10 个 seed 的实际噪声数
  平均为 961.3/4795（约 20.0%）。
- `minority_high_40`：majority 目标噪声率 0.10，minority 目标噪声率
  0.40；10 个 seed 的实际噪声数平均为 555.5/4795（约 11.6%），
  minority 噪声数平均为 97.1/240（约 40.5%）。

这些百分比是生成噪声的目标/实现统计，不应与结果表中的查询精度
(`query_noise_precision`) 混淆。

## 3. 实验矩阵与完整性

| 维度 | 取值 |
|---|---|
| 外层 seed | 0–9，共 10 个 |
| 噪声设置 | `uniform20`, `minority_high_40` |
| 查询预算 | 2%, 5%（对应 96、240 个训练样本；`budget_to_count` 规则） |
| RV-Q `tail_fraction` (`tau`) | 0.20、0.50 |
| 验证子集比例 | 1.0 |
| 验证标签噪声 | 0.0；模式字段为 `symmetric`，因此主结果使用干净公开验证标签 |
| 方法 | `loss`, `noise_score`, `entropy`, `expected_repair_value`（RV-Q） |
| 重训练 | 同一冻结特征线性头协议；每个方法和 seed 独立训练 |
| bootstrap | 10,000 次；以 seed 为重采样单位 |

行数核对：每个 `(noise, seed, budget)` 有 3 条 reference 行和 2 条 RV-Q
条件行，故 `2 × 10 × 2 × (3+2) = 200` 条
`per_seed_comparison.csv` 数据行；8 个 `(noise, budget, tau)` 组合对应
`comparison_summary.csv` 的 8 条数据行。当前结果包满足该计数。

## 4. 方法定义与排序规则

所有方法的排序在查询前完成，并使用稳定的降序排序；主分数相同时用最终
loss 降序作为确定性 tie-break。

### 4.1 Reference 方法

- `Loss`：探针在训练 noisy label 上得到的最终 per-sample cross-entropy loss。
- `Entropy`：探针预测分布的 Shannon entropy。
- `NoiseScore`：四个类内归一化 rank 的均值：
  `rank(loss)`、`rank(1 - p(noisy_label))`、`rank(entropy)` 和
  `rank(disagreement)`。其中 disagreement 是预测 argmax 是否等于 noisy
  label。类内 rank 避免多数类的尺度直接压制少数类。

### 4.2 RepairValue-Q（`expected_repair_value`）

1. 用验证集标签和冻结特征计算 class-conditional validation-tail 权重；
   每个类别选择其最高 CE loss 的 `ceil(tau × class_count)` 个样本，类别
   总权重相等。
2. 计算训练样本标签翻转对验证 tail loss 的一阶梯度修复价值，得到
   `repair_value_rank`。
3. 将值 rank 与 `NoiseScore` 作为噪声后验代理相乘：
   `score = clip(noise_score, 0, 1) × repair_value_rank`。
4. 按该 score 降序、最终 loss 降序排序，再取预算前缀。

这是当前仓库中 v1 expected repair value 的明确实现；它不是对外部论文中
“交互式/多轮”算法的逐字复现。主文若使用该名字，应说明这是
`RepairValue-Q (frozen-feature, tail-loss instantiation)`。

## 5. 信息边界与标签使用审计

| 信息 | 排序前可用？ | 用途 |
|---|---:|---|
| 冻结 train/val/test features | 是 | 探针、RV-Q 一阶价值和线性头训练 |
| train public `noisy_label` | 是 | Loss、NoiseScore、Entropy、RV-Q 的观测标签 |
| probe train probabilities / correctness history | 是 | 合法分数构造；来自缓存 probe dynamics |
| val public label / val probabilities | 是 | RV-Q 验证 tail 的价值估计；主条件验证噪声率为 0 |
| train private `clean_label` | 否 | 排序完成后，仅用于被查询样本的实际标签替换 |
| train private `is_noisy`, `is_minority`, `group` | 否 | 仅用于查询精度和事后群体诊断 |
| test private clean/group fields | 否 | 仅用于最终 WGA、平均准确率和 balanced accuracy 评估 |

实现层面，排序函数只接收 `SeedData` 中的 public features、public labels 和
probe outputs。加载器会预先打开 private manifest 来对齐 `is_noisy` 数组和
生成事后诊断，但 `clean_label`、`is_minority` 和 `group` 不会作为排序分数
输入；实际的 clean-label 替换发生在 ranking 完成之后。脚本没有执行
full-network training，也没有修改任何已注册的 CelebA/Waterbirds end-to-end
配置。

## 6. 重训练与评估协议

每个 `(noise, seed, budget, method)` 的流程如下：

1. 读取同一 seed 的缓存 frozen features、public train labels、probe dynamics。
2. 按方法冻结 ranking，取前 `budget_count` 个样本。
3. 读取 private `clean_label`，只替换被查询位置；未查询样本保留 noisy label。
4. 从相同的线性头初始状态开始训练 2-class `LinearHead`：
   `epochs=20`、`batch_size=4096`、`learning_rate=0.001`、
   `weight_decay=0.0001`、`patience=5`，以 validation balanced accuracy
   选择最佳 checkpoint。
5. 在固定 test features 上预测，使用 test private manifest 计算
   average accuracy、balanced accuracy 和 WGA。

`delta_wga = wga_after_retraining - baseline_wga`，其中 baseline 是同一
   seed、同一噪声和同一预算下“不进行任何修正”的线性头结果。相对于
   reference 的差异是 seed-paired 的 `delta_wga_RVQ - delta_wga_reference`。

## 7. 统计量定义

- `mean_delta_wga_pp`：10 个 seed 的绝对 ΔWGA 均值，乘 100 转为百分点。
- `delta_wga_ci_low/high_pp`：对 10 个 seed 做 10,000 次 percentile bootstrap，
  取 2.5% 和 97.5% 分位数。seed 是重采样单位，不把预算或样本当作独立观测。
- `positive_absolute_seeds`：绝对 ΔWGA > 0 的 seed 数。
- `mean_vs_<method>_pp` 及对应 CI：同一 `(noise, budget, tau)` 下，逐 seed
  与 reference 配对后的百分点差。
- `positive_vs_<method>_seeds`：paired 差异 > 0 的 seed 数。
- `sign_test_vs_<method>_p`：去除零差异后的 exact two-sided sign test；它是
  描述性配对检验，不替代跨数据集确认性统计。

## 8. 完整结果（主关注 `tau=0.50`）

下表数值直接来自 `comparison_summary.csv`；格式为
`均值 [95% bootstrap CI]`，单位均为 pp。括号后的 `+k/10` 是正向 seed 数，
`p` 是 exact sign-test p-value。

### 8.1 `tau=0.50`

| 噪声 | 预算 | RV-Q vs 无修正 | RV-Q vs Loss | RV-Q vs NoiseScore | RV-Q vs Entropy | 查询噪声精度 | minority 查询率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minority_high_40 | 2% | +3.57 [-0.79, +8.43] (+7/10) | +8.24 [+3.77, +13.05] (+9/10, p=.021) | -2.31 [-4.64, -0.42] (+1/10, p=.039) | +1.74 [-1.99, +5.67] (+4/10, p=.754) | 25.00% | 16.56% |
| minority_high_40 | 5% | +6.81 [+3.12, +11.04] (+8/10) | +14.42 [+10.08, +18.85] (+10/10, p=.002) | -3.58 [-7.14, -0.36] (+3/10, p=.508) | +2.57 [-0.48, +5.95] (+7/10, p=.344) | 16.21% | 8.79% |
| uniform20 | 2% | +8.17 [+4.27, +12.56] (+9/10) | +5.04 [-1.29, +10.25] (+9/10, p=.021) | +4.63 [-1.39, +9.06] (+9/10, p=.021) | +5.64 [-1.93, +12.09] (+9/10, p=.021) | 33.75% | 10.94% |
| uniform20 | 5% | +9.30 [+6.02, +12.85] (+10/10) | +9.25 [+5.49, +12.90] (+9/10, p=.021) | +8.62 [+4.20, +12.96] (+7/10, p=.344) | +6.31 [+3.41, +9.33] (+8/10, p=.109) | 25.75% | 6.33% |

### 8.2 `tau=0.20` 敏感性对照

| 噪声 | 预算 | RV-Q vs 无修正 | RV-Q vs Loss | RV-Q vs NoiseScore | RV-Q vs Entropy | 查询噪声精度 | minority 查询率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minority_high_40 | 2% | -0.84 [-4.49, +1.99] (+8/10) | +3.83 [+0.87, +7.73] (+9/10, p=.021) | -6.71 [-12.41, -1.96] (+2/10, p=.109) | -2.66 [-7.02, +2.15] (+4/10, p=.754) | 8.54% | 7.50% |
| minority_high_40 | 5% | +2.88 [+0.17, +6.09] (+8/10) | +10.50 [+7.40, +14.21] (+10/10, p=.002) | -7.50 [-10.07, -4.85] (+1/10, p=.021) | -1.36 [-4.16, +2.01] (+4/10, p=.754) | 5.21% | 4.67% |
| uniform20 | 2% | +4.09 [+1.32, +8.03] (+10/10) | +0.97 [-1.01, +2.61] (+7/10, p=.344) | +0.56 [-2.66, +3.29] (+6/10, p=.754) | +1.56 [-0.56, +3.69] (+7/10, p=.344) | 6.77% | 4.27% |
| uniform20 | 5% | +2.50 [-2.73, +8.19] (+7/10) | +2.45 [-3.66, +8.22] (+7/10, p=.344) | +1.82 [-5.29, +8.22] (+6/10, p=.754) | -0.50 [-8.54, +6.45] (+8/10, p=.109) | 4.38% | 2.92% |

`query_noise_precision` 与 `minority 查询率` 是事后诊断，不能被解释为
排序器的输入。若需要未四舍五入数值、每个 seed 的 WGA 和训练 epoch，使用
CSV 原文件。

## 9. 输出产物清单与 hash

结果目录：`outputs/local_waterbirds_repairvalue_comparison/`

| 文件 | 用途 | 数据行 | SHA-256 |
|---|---|---:|---|
| `per_seed_comparison.csv` | 每个方法、seed、预算、RV-Q 条件的原子记录 | 200 | `ce9f9411f959977abb39d9a125f2838ce577950b88b0682f118aa89d3d3f1f89` |
| `comparison_summary.csv` | 8 个条件的 bootstrap 和配对比较汇总 | 8 | `d88dd0aafa0de5262917c15c22c1c398cd596a7cef8ecdd07c6dd289e2ea22b0` |
| `protocol.json` | 运行参数、条件网格、边界和 62 个输入文件 fingerprint | — | `ff3a4025ef00f51250d81c5f084bc92f5d482415734229513c54b2ee7f101284` |
| `STATUS.md` | 简短证据边界声明 | — | `8d39b1250537ccd55f100b8494aefbb572b9abef370fc4b32b0a23fbaa73a12b` |

`protocol.json` 的 `input_fingerprint.files` 是输入数据的 canonical 清单，
包括 train/val 特征、20 组（2 noise × 10 seed）public/private train manifest
和对应 probe dynamics，共 62 个文件。该字段比只记录输出 hash 更重要：它
能够检测“结果文件没变但输入缓存被替换”的情况。

审计注意：当前 runner 的 62-file fingerprint 没有包含评估时读取的
`features/test.npz`。为使本次结果的评估输入仍可核验，补充记录其 SHA-256
为 `16898a9d15e862f2bfcf2c0ac2391a6d8f89dcadeaca7a42c23e2cb79a92c69f`
（29,736,831 bytes）。后续若将该包升级为正式确认性 artifact，应把
`test.npz` 纳入 runner 的 `input_fingerprint`，并重新生成 `protocol.json`；
在此之前不能声称 62-file 清单覆盖了全部输入。

PowerShell 校验输出包：

```powershell
Get-FileHash outputs/local_waterbirds_repairvalue_comparison/per_seed_comparison.csv -Algorithm SHA256
Get-FileHash outputs/local_waterbirds_repairvalue_comparison/comparison_summary.csv -Algorithm SHA256
Get-FileHash outputs/local_waterbirds_repairvalue_comparison/protocol.json -Algorithm SHA256
Get-FileHash outputs/local_waterbirds_repairvalue_comparison/STATUS.md -Algorithm SHA256
```

## 10. CSV 字段字典

### 10.1 `per_seed_comparison.csv`

| 字段 | 含义 |
|---|---|
| `dataset` | 固定为 `waterbirds` |
| `noise_name` | `uniform20` 或 `minority_high_40` |
| `seed` | 外层 corruption / retraining seed，0–9 |
| `method` | `loss`、`noise_score`、`entropy` 或 `expected_repair_value` |
| `condition_index` | RV-Q 条件编号；reference 行为 `-1` |
| `tail_fraction` | RV-Q 的 `tau`；reference 为空 |
| `validation_fraction` | RV-Q 使用的验证子集比例；reference 为空 |
| `validation_noise_rate` | RV-Q 分数计算时的验证标签扰动率；主结果为 0 |
| `validation_noise_mode` | `symmetric` 或 reference 标记 `reference` |
| `budget_fraction` | 查询预算比例，0.02 或 0.05 |
| `budget_count` | 预算对应的训练样本数，96 或 240 |
| `query_noise_precision` | 查询前缀中 `is_noisy` 的比例，事后统计 |
| `query_noisy_count` | 查询前缀中 noisy 样本数 |
| `query_jaccard_to_reference` | 查询集合 Jaccard；当前比较包未填充，保留 schema |
| `average_accuracy` | 修正后 test average accuracy |
| `balanced_accuracy` | 修正后 test balanced accuracy |
| `wga` | 修正后 test worst-group accuracy |
| `baseline_wga` | 同 seed/noise/budget 的无修正基线 WGA |
| `baseline_average_accuracy` | 无修正基线 average accuracy |
| `delta_wga` | `wga - baseline_wga`，小数形式（不是 pp） |
| `delta_average_accuracy` | `average_accuracy - baseline_average_accuracy` |
| `retraining_best_epoch` | validation balanced accuracy 最佳 epoch |
| `retraining_best_validation_metric` | 最佳 validation balanced accuracy |
| `retraining_scope` | 固定为 `frozen_feature_linear_head` |
| `validation_size` | RV-Q 实际评分的验证样本数；reference 为空 |
| `validation_label_accuracy` | 评分标签与干净公开验证标签的一致率；reference 为空 |
| `score_rank_correlation_to_noise` | RV-Q score rank 与事后 noise mask proxy 的 rank correlation |
| `active_tail_count` | 验证 tail 中非零权重样本数 |
| `active_tail_count_min_by_class` | 各类别 tail 非零数的最小值 |
| `posthoc_minority_query_rate` | 查询前缀的 minority 比例，事后统计 |
| `posthoc_noisy_minority_recall` | 查询到的 noisy-minority 占全部 noisy-minority 的比例 |
| `posthoc_query_group_entropy` | 查询 group 分布熵 |
| `posthoc_query_group_count` | 查询中出现的 group 数 |

空字段是有意的：reference 方法没有 RV-Q validation condition；不要把空值
填成 0 后再汇总。

### 10.2 `comparison_summary.csv`

条件键为 `noise_name`、`budget_fraction`、`tail_fraction`、
`validation_fraction`、`validation_noise_rate`、`validation_noise_mode`。
其余字段含义如下：

| 字段组 | 含义 |
|---|---|
| `n_seeds` | 条件内 seed 数，应为 10 |
| `mean_delta_wga_pp`, `delta_wga_ci_low/high_pp` | RV-Q 相对无修正的绝对效果及 CI（pp） |
| `positive_absolute_seeds` | 绝对效果为正的 seed 数 |
| `mean_query_noise_precision_pct` | 查询噪声精度均值（%） |
| `mean_posthoc_minority_query_rate_pct` | minority 查询率均值（%） |
| `mean_vs_loss_pp`, `vs_loss_ci_low/high_pp`, `positive_vs_loss_seeds`, `sign_test_vs_loss_p` | RV-Q 与 Loss 的 seed-paired 对比 |
| `mean_vs_noise_score_pp`, `vs_noise_score_ci_low/high_pp`, `positive_vs_noise_score_seeds`, `sign_test_vs_noise_score_p` | RV-Q 与 NoiseScore 的 seed-paired 对比 |
| `mean_vs_entropy_pp`, `vs_entropy_ci_low/high_pp`, `positive_vs_entropy_seeds`, `sign_test_vs_entropy_p` | RV-Q 与 Entropy 的 seed-paired 对比 |

### 10.3 `protocol.json`

重点键：`analysis_status`（探索性状态）、`methods`、`reference_methods`、
`seeds`、`noise_names`、`budgets`、`conditions`、`training`、
`retraining_scope`、`initial_probe_reused_from_cache`、
`full_network_training_executed`、`private_fields_used_for`、
`bootstrap_replicates` 和 `input_fingerprint`。论文或审稿回复中的协议数字应
优先引用此 JSON，而不是手工复制命令行默认值。

## 11. 复现命令

在仓库根目录运行：

```powershell
python scripts/run_waterbirds_repairvalue_comparison.py `
  --input-dir outputs/ablation_waterbirds_10seeds `
  --probe-dir outputs/ablation_waterbirds_10seeds/stage1/probes `
  --output-dir outputs/local_waterbirds_repairvalue_comparison `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --noise-names uniform20 minority_high_40 `
  --budgets 0.02 0.05 `
  --taus 0.20 0.50 `
  --validation-fractions 1.0 `
  --validation-noise-rates 0.0 `
  --methods loss noise_score entropy expected_repair_value `
  --epochs 20 --batch-size 4096 --device auto
```

没有 GPU 时将最后一项改为 `--device cpu`。命令会复用已有 probe dynamics；
缺少 probe dynamics 或对应 manifest 时应失败，而不是偷偷重新训练或生成
替代输入。重跑前应保留原结果目录，另指定新的 `--output-dir` 以便比较。

## 12. 论文使用边界与已知限制

### 可以声称

- 在固定 frozen-feature linear-head protocol 和 10 个 Waterbirds seeds 下，
  RV-Q 在 `tau=0.50` 的若干条件中相对 Loss 有稳定的 seed-paired WGA 优势。
- RV-Q 的收益依赖噪声结构、预算和 tail fraction；`minority_high_40` 下它
  仍可能落后于 NoiseScore。
- 该结果是对“价值估计 + 噪声后验代理”的条件性可行性证据，并给出了完整
  的输入 fingerprints 和逐 seed 数据。

### 不可以声称

- RV-Q 在所有 Waterbirds 条件、所有预算或所有基线中都最好。
- 这是 full ResNet-50 end-to-end 的迁移成功，或已经修复注册的 transfer failure。
- `tau=0.50` 是预注册、盲选或在未见数据上确认的超参数。
- 查询噪声精度、minority 查询率或 group entropy 是模型输入；它们是 private
  事后诊断。
- 10 个 seed 的 bootstrap CI 等价于跨数据集泛化置信区间。

主要限制包括：冻结特征降低了端到端优化难度；RV-Q 使用公开验证标签；
验证噪声敏感性只在此包的显式条件中评估；预算仅覆盖 2% 和 5%；结果依赖
缓存 probe dynamics、PyTorch/设备实现和输入文件的精确版本。因此推荐在
主文中把该包放在“conditional feasibility / sensitivity audit”位置，并将
完整表格和字段字典作为补充材料或 artifact documentation。
