# RepairValue-Q v3: Calibrated Adaptive-tau Waterbirds Validation

日期: 2026-08-27  
状态: 探索性证据块；不修改已注册的 Waterbirds 确认性协议

## 一句话结论

v3 在 v2 的五折、样本外 capture-rate 选择器上加入了 held-out direction-norm
校准，并在 Waterbirds 的 20-seed、2 噪声、2 budget 网格上完成 80/80 个
cell。它避免了 v1 的 tau=0.05 崩塌，但没有超过固定 tau=0.50，也没有超过
v2 capture 规则；因此目前更适合作为附录中的方法诊断，不建议作为论文主方法。

## 1. v3 规则

候选尾部比例为 `{0.05, 0.10, 0.20, 0.30, 0.50}`。对每个候选和每个
validation fold：

1. 在折外验证集拟合类平衡尾部梯度方向；
2. 在最多 4096 个训练样本上按一阶修复效应选择与预算相同规模的 top-k；
3. 在留出折上计算预测 top-k 效应与 oracle top-k 效应的 capture ratio；
4. 同时用 `predicted_effect / ||held_direction||` 得到尺度校准的绝对效应。

最终分数为两项在候选 tau 上的归一化秩的等权平均：

`v3(tau) = 0.5 * rank(capture_ratio) + 0.5 * rank(calibrated_effect)`。

选择仅使用训练特征/噪声标签、验证特征/概率/干净验证标签、噪声代理和预算；
测试私有标签只用于最终 WGA 与事后诊断。tau 按 `(noise, seed, budget)` 选择，
因此是预算感知的。

实现位置：`src/robust_verify/modern_baselines.py` 中的
`select_tail_fraction_by_calibrated_capture` 和
`expected_repair_value_calibrated_tau_scores`；方法名、标签和 dispatch 已加入
`src/robust_verify/scoring.py`。这是 standalone 实验接线，尚未加入
`ADAPTIVE_METHODS` 的完整端到端预算循环。

## 2. 实验协议

- 数据：`outputs/ablation_waterbirds_20seeds`
- seeds：0--19；噪声：`uniform20`、`minority_high_40`
- budget：2%、5%；tau 候选如上
- 特征：缓存的 ResNet-50 frozen features
- 重训：linear head，20 epochs，batch 4096，lr 1e-3，weight decay 1e-4，
  patience 5，按 validation balanced accuracy 选 checkpoint
- 对照：Loss、NoiseScore、固定 RepairValue-Q tau=0.20 和 tau=0.50，均取自
  已完成的 20-seed comparison package
- 统计：10,000 次 seed-level bootstrap percentile CI；精确双侧 sign test
- 等价性检查：每个 cell 重算的无修正 baseline WGA 与参考值最大偏差
  `5.6e-17`；固定 tau=0.50 的 seed 0 replay 通过

## 3. Waterbirds 结果

数值单位为 WGA 变化百分点；括号内为 95% seed-bootstrap CI。

| 噪声 | budget | tau 均值 (范围) | 查询精度 | 绝对 Delta WGA | v3 - Loss | v3 - NoiseScore | v3 - tau=.20 | v3 - tau=.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform20 | 2% | 0.325 (0.10--0.50) | 17.1% | +4.05 [+1.98,+6.70] | +2.94 [-0.55,+6.35] | +2.45 [-1.23,+5.88] | +1.29 [-1.95,+4.15] | -2.67 [-4.30,-1.28] |
| uniform20 | 5% | 0.295 (0.10--0.50) | 11.8% | +5.33 [+2.32,+8.75] | +7.83 [+4.24,+11.12] | +8.12 [+4.93,+11.12] | +3.52 [+0.37,+7.39] | -1.73 [-3.92,+0.61] |
| minority-high-40 | 2% | 0.400 (0.20--0.50) | 19.5% | +2.54 [-1.43,+6.32] | +7.87 [+4.05,+11.40] | -3.26 [-7.55,+0.32] | +3.08 [-0.83,+6.92] | -1.34 [-3.44,+0.07] |
| minority-high-40 | 5% | 0.375 (0.20--0.50) | 11.3% | +7.17 [+4.44,+10.06] | +15.37 [+12.15,+18.77] | -2.55 [-5.28,+0.19] | +5.98 [+2.80,+10.03] | -1.34 [-2.43,-0.39] |

所有 cell 的选择分布为：`tau=0.10` 3 次、`0.20` 19 次、`0.30` 26 次、
`0.50` 32 次；没有选择 `0.05`。v3 的 tau 分布比 v2 capture 规则
(`0.50` 39/80) 更偏向 0.20--0.30，但这没有转化为更高 WGA。

## 4. 与 v2 和固定 tau 的比较

v3 相对 v2 capture 规则的绝对 Delta WGA 分别为：

| 条件 | v2 capture | v3 calibrated | v3 - v2 |
|---|---:|---:|---:|
| uniform20 @2% | +4.50 | +4.05 | -0.46 pp |
| uniform20 @5% | +6.00 | +5.33 | -0.67 pp |
| minority-high @2% | +2.79 | +2.54 | -0.25 pp |
| minority-high @5% | +7.39 | +7.17 | -0.22 pp |

相对固定 tau=0.50，v3 在四个条件均为负，差值为 -1.34 到 -2.67 pp；
其中 uniform20@2% 和 minority-high@5% 的 bootstrap 区间明确为负。相对
固定 tau=0.20，v3 在四个条件点估计均为正，但只有 uniform20@5% 和
minority-high@5% 的区间不跨 0。相对 NoiseScore，uniform20 下 v3 点估计
为正，minority-high 下仍落后，符合已有 Waterbirds 噪声机制结果。

## 5. 解释与论文建议

v3 成功解决的是 v1 的选择退化：它不再机械地选最小 tau，并能在无需知道
测试标签的情况下选择中等或宽尾配置。但 `||held_direction||` 校准没有恢复
固定 tau=0.50 的性能，反而比 v2 略低，说明候选 tau 的尺度估计仍受验证集
有限样本和 frozen-feature 线性近似影响。

建议：

- 可放入附录作为“校准型 adaptive-tau 的负结果/敏感性分析”；
- 不建议替换当前固定 tau=0.50 的 Waterbirds 主结果，也不建议宣称 v3
  击败 NoiseScore 或 v2；
- 若继续做新版本，应先预注册独立外层 validation、固定效应标度和候选集，
  再进行跨数据集测试；不建议根据本次 Waterbirds 结果继续定向调权重。

## 6. 复现与输出

运行命令：

```powershell
$env:PYTHONPATH="src"
python scripts/run_waterbirds_repairvalue_adaptive_tau.py `
  --input-dir outputs/ablation_waterbirds_20seeds `
  --probe-dir outputs/ablation_waterbirds_20seeds/stage1/probes `
  --rule calibrated_capture `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --noise-names uniform20 minority_high_40 `
  --budgets 0.02 0.05 `
  --output-dir outputs/local_waterbirds_repairvalue_calibrated_tau `
  --epochs 20 --batch-size 4096 --device auto
```

结果文件：

- `outputs/local_waterbirds_repairvalue_calibrated_tau/per_seed_adaptive.csv`
- `outputs/local_waterbirds_repairvalue_calibrated_tau/comparison_summary.csv`
- `outputs/local_waterbirds_repairvalue_calibrated_tau/protocol.json`

验证：`pytest -q tests/test_modern_baselines.py` -> 12 passed；seed 0 exact-
protocol smoke 和完整 80-cell 网格均通过。
