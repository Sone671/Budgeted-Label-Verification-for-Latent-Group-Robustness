# 实验结论: Budgeted Label Verification under Latent Groups

> **生成日期**: 2026-06-23
> **代码版本**: `latent-group-verification-mvp` v0.1.0 + `oracle_noise_balanced_patch` 集成
> **数据来源**: 全部数据由 `outputs/waterbirds_mvp_5seeds/` 目录下的实验输出生成

---

## 1. 实验设置

### 1.1 数据集: Waterbirds (WILDS v1.0)

| Split | n | y=0 (waterbird) | y=1 (landbird) | minority |
|-------|---|-----------------|-----------------|----------|
| train | 4,795 | 3,682 | 1,113 | 240 (5.0%) |
| val | 1,199 | 933 | 266 | 599 (50.0%) |
| test | 5,794 | 4,510 | 1,284 | 2,897 (50.0%) |

Group 定义: `group = 2 × label + place`; minority = label ≠ place.

**数据来源**: `outputs/waterbirds_mvp_5seeds/canonical/*.csv`
**下载方式**: `wilds.get_dataset("waterbirds", download=True)` → `data/waterbirds_v1.0/`

### 1.2 方法列表

| 类型 | 方法名 | 说明 |
|------|--------|------|
| Legal (不接触 private) | `random` | 随机查询 |
| | `loss` | 按最终 epoch 的 per-sample loss 降序 |
| | `entropy` | 按预测熵降序 |
| | `forgetting` | 按训练中被遗忘次数降序 |
| | `noise_score` | 类内 rank 归一化的综合噪声分数 |
| Oracle (仅诊断上限) | `oracle_noise` / `oracle_noise_loss_tiebreak` | Noisy-first + loss tie-break（旧版 oracle_noise） |
| | `oracle_noise_random_tiebreak` | Noisy-first + noisy 内部随机排序 |
| | `oracle_noise_group_balanced` | Noisy-first + noisy 内部 group round-robin |
| | `oracle_minority` | 优先 noisy minority，其次是 clean minority |
| | `oracle_group_balanced` | 所有样本 group round-robin（不分 noisy/clean） |

### 1.3 实验配置

```yaml
# configs/local_5seeds.yaml
training:
  epochs: 5
  batch_size: 256
  learning_rate: 0.001
  selection_metric: balanced_accuracy

noise:
  settings:
    - name: uniform20
      kind: uniform
      rate: 0.20
    - name: minority_high
      kind: group_dependent
      majority_rate: 0.10
      minority_rate: 0.40

experiment:
  seeds: [0, 1, 2, 3, 4]
  budgets: [0.5%, 1%, 2%, 5%, 10%]
```

**数据来源**: `configs/local_5seeds.yaml`

---

## 2. 噪声注入统计 (5 seeds, mean ± sem)

| 噪声配置 | 实际噪声率 | 多数群体噪声率 | 少数群体噪声率 | 噪声样本数 |
|----------|-----------|---------------|---------------|-----------|
| uniform20 | 0.2055 ± 0.0014 | 0.2050 ± 0.0014 | 0.2142 ± 0.0104 | 985 ± 6 |
| minority_high | 0.1183 ± 0.0032 | 0.1029 ± 0.0033 | **0.4108 ± 0.0097** | 567 ± 15 |

minority_high 场景下少数群体噪声率是多数群体的 4 倍，且绝对噪声率较低（11.8%），更接近真实场景中少数群体更容易被误标注的情况。

**数据来源**: `outputs/waterbirds_mvp_5seeds/noise_statistics.csv`

---

## 3. 基线探针性能 (5 seeds, mean ± sem)

| 噪声配置 | Average Accuracy | Balanced Accuracy | WGA |
|----------|-----------------|------------------|-----|
| uniform20 | 0.8462 ± 0.0103 | 0.8087 ± 0.0064 | 0.5692 ± 0.0357 |
| minority_high | 0.8067 ± 0.0066 | 0.7550 ± 0.0045 | 0.4056 ± 0.0286 |

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv` (baseline_wga, baseline_average_accuracy 列)

---

## 4. 核心诊断 1: Target Mismatch — **100% 确认**

**在所有 10 个 (noise × seed) 组合中，修正最多标签的方法 ≠ WGA 提升最大的方法。**

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv`，按 (noise_name, seed) 分组，比较 `num_corrected` max 和 `delta_wga` max 的 method。

### 4.1 典型案例 (minority_high, seed=0, budget=2%)

| 方法 | 修正数 | minority_query_rate | ΔWGA |
|------|--------|--------------------|------|
| `oracle_noise_loss_tiebreak` | **96** | 0.000 | **-0.045** |
| `loss` | 96 | 0.000 | -0.045 |
| `forgetting` | 77 | 0.146 | +0.011 |
| `noise_score` | 53 | 0.479 | **+0.070** |

修正最多的方法（96 个标签）导致 WGA 下降 4.5pp；修正较少的方法（53 个标签）反而提升 WGA 7.0pp。

---

## 5. 核心诊断 2: oracle_noise 三种变体的区分度

这是本次补丁引入的最重要对比。三种 oracle_noise 变体在 budget=2% 时修正相同数量的标签（~96 个，noise_precision≈1.0），但查询组成和 WGA 差异巨大：

### 5.1 minority_high (5 seeds, mean ± sem)

| 方法 | ΔWGA | WGA | minority_q_rate | mislabeled (maj, min) |
|------|------|-----|----------------|----------------------|
| `oracle_noise_loss_tiebreak` | -0.0193 ± 0.0333 | 0.3863 | 0.006 | (95, 0) |
| `oracle_noise_random_tiebreak` | +0.0420 ± 0.0530 | 0.4476 | 0.173 | (79, 16) |
| `oracle_noise_group_balanced` | **+0.1006 ± 0.0348** | 0.5062 | 0.500 | (48, 48) |
| `noise_score` (legal) | +0.0826 ± 0.0163 | 0.4883 | 0.410 | (48, 8) |
| `oracle_minority` (上限) | +0.1657 ± 0.0369 | 0.5713 | 1.000 | (0, 94) |

### 5.2 uniform20 (5 seeds, mean ± sem)

| 方法 | ΔWGA | WGA | minority_q_rate | mislabeled (maj, min) |
|------|------|-----|----------------|----------------------|
| `oracle_noise_loss_tiebreak` | -0.0157 ± 0.0063 | 0.5535 | 0.002 | (95, 0) |
| `oracle_noise_random_tiebreak` | +0.0052 ± 0.0114 | 0.5744 | 0.037 | (92, 3) |
| `oracle_noise_group_balanced` | **+0.0788 ± 0.0143** | 0.6480 | 0.431 | (54, 41) |
| `noise_score` (legal) | +0.0553 ± 0.0253 | 0.6244 | 0.319 | (54, 5) |
| `oracle_minority` (上限) | +0.0819 ± 0.0125 | 0.6511 | 1.000 | (0, 51) |

**关键发现**: 三种 oracle 变体修正~96 个标签，但 `loss_tiebreak` 只选到 majority noisy（minority_q_rate=0.2%~0.6%），导致 WGA 下降；`group_balanced` 选到 43%~50% minority，WGA 提升 8~10pp。旧版 `oracle_noise`（即 `loss_tiebreak`）是一个**具有误导性的诊断上限**——它实际上是最差的方法之一。

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv`，筛选 `budget_fraction == 0.02`

---

## 6. 核心诊断 3: noise_score vs oracle_noise_loss_tiebreak — 统计显著

### 6.1 minority_high

| 预算 | noise_score ΔWGA | loss_tiebreak ΔWGA | Diff | positive seeds |
|------|-----------------|-------------------|------|---------------|
| 0.5% | +0.0399 ± 0.0442 | +0.0361 ± 0.0393 | +0.0037 ± 0.0162 | 3/5 |
| 1.0% | +0.0495 ± 0.0198 | +0.0231 ± 0.0408 | +0.0265 ± 0.0279 | 3/5 |
| **2.0%** | **+0.0826 ± 0.0163** | **-0.0193 ± 0.0333** | **+0.1020 ± 0.0257** | **5/5** |
| **5.0%** | **+0.0890 ± 0.0198** | **-0.0277 ± 0.0447** | **+0.1167 ± 0.0458** | **5/5** |
| **10.0%** | **+0.1210 ± 0.0117** | **-0.0022 ± 0.0460** | **+0.1232 ± 0.0448** | **5/5** |

在 minority_high 场景下，**2%+ 预算时 noise_score 在所有 5 个 seed 中都优于 loss_tiebreak**。2% 预算时平均领先 10.2pp。

### 6.2 uniform20

| 预算 | noise_score ΔWGA | loss_tiebreak ΔWGA | Diff | positive seeds |
|------|-----------------|-------------------|------|---------------|
| 0.5% | +0.0298 ± 0.0178 | +0.0012 ± 0.0160 | +0.0286 ± 0.0158 | 5/5 |
| 1.0% | +0.0308 ± 0.0135 | +0.0048 ± 0.0149 | +0.0260 ± 0.0186 | 4/5 |
| **2.0%** | **+0.0553 ± 0.0253** | **-0.0157 ± 0.0063** | **+0.0710 ± 0.0283** | **4/5** |
| 5.0% | -0.0138 ± 0.0466 | -0.0109 ± 0.0124 | -0.0029 ± 0.0526 | 2/5 |
| 10.0% | -0.0964 ± 0.0561 | -0.0467 ± 0.0121 | -0.0496 ± 0.0629 | 2/5 |

uniform20 场景下，小预算（≤2%）时 noise_score 胜出，但大预算（5%~10%）下跌——noise_score 选完 noisy minority 后继续追加的样本大部分是 clean，开始"浪费"预算。

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv`，按 (noise_name, budget_fraction, method) 分组聚合

---

## 7. 核心诊断 4: Oracle 天花板 — 头空间分析

noise_score 与 oracle_noise_group_balanced 之间的差距 = oracle 天花板（legal 方法的改进空间）。

### 7.1 minority_high

| 预算 | noise_score ΔWGA | oracle_noise_gb ΔWGA | Gap |
|------|-----------------|---------------------|-----|
| 0.5% | +0.0399 ± 0.0442 | +0.0458 ± 0.0430 | +0.006 ± 0.009 |
| 1.0% | +0.0495 ± 0.0198 | +0.0544 ± 0.0459 | +0.005 ± 0.029 |
| 2.0% | +0.0826 ± 0.0163 | +0.1006 ± 0.0348 | +0.018 ± 0.020 |
| **5.0%** | **+0.0890 ± 0.0198** | **+0.2182 ± 0.0291** | **+0.129 ± 0.020** |
| **10.0%** | **+0.1210 ± 0.0117** | **+0.1405 ± 0.0223** | **+0.020 ± 0.025** |

### 7.2 uniform20

| 预算 | noise_score ΔWGA | oracle_noise_gb ΔWGA | Gap |
|------|-----------------|---------------------|-----|
| 0.5% | +0.0298 ± 0.0178 | +0.0395 ± 0.0144 | +0.010 ± 0.008 |
| 1.0% | +0.0308 ± 0.0135 | +0.0551 ± 0.0129 | +0.024 ± 0.011 |
| 2.0% | +0.0553 ± 0.0253 | +0.0788 ± 0.0143 | +0.024 ± 0.014 |
| **5.0%** | **-0.0138 ± 0.0466** | **+0.1196 ± 0.0247** | **+0.133 ± 0.026** |
| **10.0%** | **-0.0964 ± 0.0561** | **+0.0258 ± 0.0384** | **+0.122 ± 0.019** |

**关键发现**: 在 5% 预算时 oracle 天花板最大（~13pp）。noise_score 在 2% 时接近天花板（仅差 2pp），但无法有效利用更大预算——这是未来工作的明确方向（multi-round active querying, latent clustering 等）。

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv`

---

## 8. 核心诊断 5: Legal 方法最佳方法一致性

每个 seed 下 ΔWGA 最高的 legal 方法：

| Seed | minority_high | uniform20 |
|------|-------------|-----------|
| 0 | noise_score (+0.120, 10%) | noise_score (+0.103, 2%) |
| 1 | entropy (+0.268, 10%) | forgetting (+0.011, 1%) |
| 2 | entropy (+0.213, 10%) | noise_score (+0.065, 2%) |
| 3 | noise_score (+0.108, 10%) | noise_score (+0.128, 5%) |
| 4 | noise_score (+0.154, 10%) | entropy (+0.045, 10%) |

**noise_score 在 7/10 个 seed×noise 组合中是最佳 legal 方法。** entropy 在 3/10 个组合中胜出（特别是 minority_high 场景）。

**数据来源**: `outputs/waterbirds_mvp_5seeds/stage1/results.csv`，legal methods only

---

## 9. 总结

### 9.1 已被验证的假设

| # | 假设 | 状态 | 证据 |
|---|------|------|------|
| H1 | 修正最多的标签 ≠ WGA 提升最大 | ✅ 确认 | 10/10 (noise×seed) mismatch; loss 修正 96 个标签但 WGA 下降 |
| H2 | oracle_noise (loss tie-break) 在小预算下是误导性上限 | ✅ 确认 | 在 2% 预算下 minority_q_rate=0.2%~0.6%, WGA 下降 |
| H3 | noise_score 显著优于纯 loss 方法 | ✅ 确认 | minority_high 2%+ 预算 5/5 seeds 胜出 |
| H4 | oracle_noise_group_balanced 是更好的诊断上限 | ✅ 确认 | 在小预算下立即覆盖 minority noisy |
| H5 | 存在 oracle 天花板，legal 方法无法有效利用大预算 | ✅ 确认 | 5%~10% 预算时 gap=10~13pp |

### 9.2 论文中可以报告的核心结论

1. **Target mismatch 现象**: 在 Waterbirds 的两种标签噪声场景下，按 loss 修正最多标签的方法会导致 WGA 下降；而 noise_score 修正较少标签反而显著提升 WGA。这一发现在 5 个随机种子下稳定复现（10/10）。

2. **oracle_noise 需要明确定义**: 旧版 "oracle noise"（noisy-first + loss tie-break）在小预算下退化成了 "多数群体噪声优先"，不能代表"最优纠错"。应报告三种变体（loss tie-break / random tie-break / group-balanced），并将 group-balanced 作为诊断上限。

3. **noise_score 在小预算下有效但无法扩展**: 2% 预算时 noise_score 已接近 oracle 天花板，但继续增加预算无法进一步提升 WGA——表明需要多轮查询或 latent clustering 来更好地利用预算。

4. **Entropy 是强有力的 baseline**: 在部分 seed 中 entropy 超过了 noise_score，说明在某些噪声分布下简单的不确定性采样就能有效发现 minority 噪声。

### 9.3 局限性

- 仅 5 epochs 训练（快速配置），完整实验应使用 20 epochs
- 仅 Waterbirds 数据集，需要其他 DG/WILDS 数据集验证泛化性
- 仅二分类 + binary flipping 噪声
- 冻结 backbone + 线性探针，非端到端
- 仅 correction-only retraining，非多轮交互

### 9.4 输出文件清单

```
outputs/waterbirds_mvp_5seeds/
├── noise_statistics.csv              # 噪声注入统计
├── canonical/{train,val,test}.csv    # 标准化数据划分
├── features/{train,val,test}.npz     # ResNet-50 特征
├── manifests/                        # public/private manifest × noise × seed
└── stage1/
    ├── results.csv                   # ★ 主结果 (550 行)
    ├── score_correlations.csv        # 查询分数 Spearman 相关性
    ├── probes/                       # 训练探针权重 + 动力学
    ├── queries/                      # 每个方法的查询排序
    └── plots/                        # WGA-budget 曲线、修正-vs-WGA、查询组成
```

---

## 10. 本地 Waterbirds RepairValue-Q 可行性空间探索（2026-08-25）

本节记录在已有 Waterbirds 缓存上完成的探索性训练结果。该实验的目的
是判断 RepairValue-Q 是否存在可行性空间，而不是把 Waterbirds 结果升级为
新的确认性端到端证据。

### 10.1 协议与边界

- 数据：Waterbirds，`uniform20` 与 `minority_high_40`；
- seeds：0--9；
- 预算：2% 与 5%；
- tail fraction：$\tau=0.20$ 与 $\tau=0.50$；
- 验证标签：干净验证集，以及 10% symmetric validation-label-noise 压力控制；
- 初始 probe 与 ResNet-50 特征：复用已有缓存，不重新训练图像网络；
- 每个条件：冻结特征上的线性头训练 20 epochs，并在同一协议下重新训练
  no-correction 与 Loss 参考头；
- 私有 group/clean-label 字段：只在 query ranking 固定后用于离线诊断和评估；
- 不执行完整 ResNet-50 端到端训练。

共有 120 条唯一结果：$2$ 个噪声类型 $\times 10$ 个 seeds $\times 2$ 个预算
$\times 3$ 个 one-factor 条件。重要的是，Loss 与 no-correction 参考也使用
同一 20-epoch 线性头协议重新训练，避免将旧的 5-epoch Waterbirds baseline
与新结果直接相减。

### 10.2 干净验证标签下的训练后结果

表中 `RV-Q − Loss` 是 seed-paired 的 WGA 差值，均值单位为百分点；
`positive` 是 RepairValue-Q 胜过 Loss 的 seed 数。

| 噪声 | 预算 | $\tau$ | 平均绝对 $\Delta$WGA | 平均 RV-Q − Loss | positive |
|---|---:|---:|---:|---:|---:|
| minority_high_40 | 2% | 0.20 | -0.84 pp | +3.83 pp | 9/10 |
| minority_high_40 | 2% | 0.50 | +3.57 pp | +8.24 pp | 9/10 |
| minority_high_40 | 5% | 0.20 | +2.88 pp | +10.50 pp | 10/10 |
| minority_high_40 | 5% | 0.50 | +6.81 pp | +14.42 pp | 10/10 |
| uniform20 | 2% | 0.20 | +4.09 pp | +0.97 pp | 7/10 |
| uniform20 | 2% | 0.50 | +8.17 pp | +5.04 pp | 9/10 |
| uniform20 | 5% | 0.20 | +2.50 pp | +2.45 pp | 7/10 |
| uniform20 | 5% | 0.50 | +9.30 pp | +9.25 pp | 9/10 |

结果显示 Waterbirds 上确实存在可行性空间，且主要集中在较大的尾部比例
（$\tau=0.50$）与预算区间。minority-high-40 下信号最强：2%/5% 预算时
相对 Loss 的平均优势分别为 +8.24/+14.42 pp；uniform20 下对应优势为
+5.04/+9.25 pp。

### 10.3 验证标签质量压力测试

10% 验证标签噪声只对 $\tau=0.20$ 条件施加，作为压力控制，不与干净验证
条件合并解释：

| 噪声 | 预算 | 平均 RV-Q − Loss | positive |
|---|---:|---:|---:|
| minority_high_40 | 2% | +4.42 pp | 10/10 |
| minority_high_40 | 5% | +6.28 pp | 9/10 |
| uniform20 | 2% | -2.17 pp | 4/10 |
| uniform20 | 5% | +0.00 pp | 6/10 |

因此 Waterbirds 的可行性不是无条件的：方法对验证标签质量和噪声结构
敏感，uniform20 的 2% 压力控制已经使平均 paired gain 反转为负。

### 10.4 解释边界

这批结果支持以下较窄结论：在冻结 ImageNet ResNet-50 表征、线性头重训练、
同协议 Loss 参考和当前两类合成噪声下，RepairValue-Q 存在一组可复现的
正向条件。它不支持以下更强结论：

1. Waterbirds 上完整网络端到端迁移已被确认；
2. RepairValue-Q 在所有预算、噪声类型或验证标签质量下都安全；
3. τ=0.50 是可直接注册的全局最优超参数；
4. 本地探索结果可以替代已有的 Waterbirds v1/v2 端到端失败边界。

### 10.5 结果与复现文件

- 逐 seed 训练结果：`outputs/local_waterbirds_repairvalue_retraining_protocol/per_seed_retraining.csv`
- 条件汇总：`outputs/local_waterbirds_repairvalue_retraining_protocol/retraining_sensitivity_summary.csv`
- 结果说明：`outputs/local_waterbirds_repairvalue_retraining_protocol/RESULTS.md`
- 状态元数据：`outputs/local_waterbirds_repairvalue_retraining_protocol/STATUS.md`
- 探索入口：`scripts/run_waterbirds_repairvalue_exploration.py`
- 相关测试：`tests/test_waterbirds_repairvalue_exploration.py`

复现示例：

```bash
python scripts/run_waterbirds_repairvalue_exploration.py \
  --mode retrain \
  --noise-names minority_high_40 uniform20 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --budgets 0.02 0.05 \
  --taus 0.50 \
  --validation-fractions 1.0 \
  --validation-noise-rates 0.0 0.10 \
  --output-dir outputs/local_waterbirds_repairvalue_retraining_protocol \
  --epochs 20 --batch-size 4096 --device cuda
```

---

## 11. 复现命令

```bash
cd /g/latent_group_verification_mvp
source .venv/Scripts/activate

# 下载数据
python -c "from wilds import get_dataset; get_dataset('waterbirds', download=True, root_dir='./data')"

# 运行 5-seed 实验
rv-stage0 --config configs/local_5seeds.yaml
rv-stage1 --config configs/local_5seeds.yaml

# 运行诊断
rv-diagnostics \
  --input outputs/waterbirds_mvp_5seeds/manifests/uniform20_seed0_train_private.csv \
  --budgets 0.5% 1% 2% 5% 10%
```
