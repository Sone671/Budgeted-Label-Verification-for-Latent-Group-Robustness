# RepairValue-Q Adaptive-tau CelebA Validation

日期: 2026-08-27  
状态: 探索性跨数据集验证；不修改已注册 confirmatory 协议

## 结论

在 CelebA 的 10-seed、两类噪声、2%/5% budget 网格上，capture-rate
规则在 40 个 `(noise, seed, budget)` 单元格中有 39 个选择
`tau=0.50`，仅 `minority_high_40, seed=4, budget=2%` 选择
`tau=0.30`。因此该规则在 CelebA 上能够恢复论文当前使用的宽尾部
配置，但没有超过固定 `tau=0.50`；它更像是合法的自动化复现规则，而
不是新的性能上界。

## 协议

- 输入: `outputs/ablation_celeba_10seeds` 的冻结 ResNet-50 特征与 probe
  动力学；缺少验证概率历史时从缓存的最佳 probe checkpoint 重算。
- seeds: 0--9。
- 噪声: `uniform20`、`minority_high_40`。
- budget: 2%、5%。
- 重训: 冻结特征线性头，20 epoch，batch 4096，learning rate 1e-3，
  weight decay 1e-4，patience 5，balanced accuracy 选点。
- tau 候选: `{0.05, 0.10, 0.20, 0.30, 0.50}`。
- 选择: 五折分层 validation，预算感知的 held-out first-order capture
  ratio；最多抽样 4096 个训练样本。测试集私有标签只用于最终 WGA。
- 方法: Loss、NoiseScore、固定 `tau=0.20`、固定 `tau=0.50`、
  capture-adaptive tau。

## 结果

数值单位为 WGA 变化的百分点；区间是 10,000 次 seed bootstrap。

| 噪声 | budget | 选中 tau 均值 | adaptive 绝对 ΔWGA | adaptive - Loss | adaptive - NoiseScore | adaptive - tau=.20 | adaptive - tau=.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform20 | 2% | 0.50 | +5.31 [+4.45,+6.01] | +0.51 [-0.61,+1.56] | -3.56 [-4.17,-3.01] | +0.42 [+0.14,+0.71] | 0.00 |
| uniform20 | 5% | 0.50 | +7.11 [+6.45,+7.69] | +1.44 [+0.09,+2.82] | -2.57 [-3.06,-2.03] | +0.38 [+0.22,+0.57] | 0.00 |
| minority-high-40 | 2% | 0.48 | +2.08 [+1.48,+2.74] | +0.95 [+0.22,+1.61] | -7.22 [-7.82,-6.66] | +0.25 [0.00,+0.60] | +0.03 |
| minority-high-40 | 5% | 0.50 | +3.12 [+2.59,+3.72] | +3.60 [+2.49,+4.76] | -6.97 [-7.78,-6.11] | +0.66 [+0.25,+1.29] | 0.00 |

当 adaptive 选择 `tau=0.50` 时，查询排序与固定 `tau=0.50` 完全一致；
因此三行的 paired 差异为零。唯一的 `tau=0.30` 单元格相对固定
`tau=0.50` 只带来约 0.03 pp 的差异。

## 解释边界

1. CelebA 的 validation 一阶信号比 Waterbirds 更支持 `tau=0.50`，说明
   规则不是固定偏向小 tau；Waterbirds 中它会选择 0.2/0.3，而 CelebA
   中几乎总是选择 0.5。
2. 相对 Loss，adaptive 在四个设置均为正，但 uniform20@2% 的区间跨过
   0，不能宣称普遍显著优于 Loss。
3. 相对 NoiseScore，四个设置均明显落后，特别是 minority-high 噪声。
   这与论文已有的 CelebA 噪声机制结果一致。
4. adaptive 没有超过固定 `tau=0.50`。因此本实验验证的是“可以在不显式
   指定 tau 的情况下识别宽尾配置”，不是“自适应 tau 带来额外 WGA 提升”。
5. 这是 frozen-feature linear-head 结果，不等价于 CelebA full-network
   retraining；不能直接替换已注册的端到端证据块。

## 复现与输出

运行脚本:

```powershell
$env:PYTHONPATH="src"
python scripts/run_celeba_repairvalue_adaptive_tau.py `
  --input-dir outputs/ablation_celeba_10seeds `
  --probe-dir outputs/ablation_celeba_10seeds/stage1/probes `
  --output-dir outputs/local_celeba_repairvalue_capture_tau `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --noise-names uniform20 minority_high_40 `
  --budgets 0.02 0.05 `
  --epochs 20 --batch-size 4096 --device auto
```

结果文件:

- `outputs/local_celeba_repairvalue_capture_tau/per_seed_comparison.csv`
- `outputs/local_celeba_repairvalue_capture_tau/comparison_summary.csv`
- `outputs/local_celeba_repairvalue_capture_tau/protocol.json`

完整性检查: 240 行结果（40 个单元格 × 6 个方法），每个单元格均包含
baseline、Loss、NoiseScore、固定两种 tau 和 adaptive 结果。

## 论文建议

建议将本结果作为 adaptive-tau 的跨数据集验证或附录敏感性实验，措辞为
“在 CelebA 上自动恢复宽尾 `tau=0.50` 配置”。不建议把它作为击败
NoiseScore 或固定 `tau=0.50` 的新主方法，也不建议在看到 Waterbirds 和
CelebA 结果后继续做针对性 v3 调参。若未来要做 v3，应预先固定一个带
尺度校准的绝对修复效应目标，并在独立外层 validation 上评估。
