# CelebA 本地现代基线工作记录

## 证据级别

本运行是**后验、探索性冻结特征筛查**。它复用已经观察过的 CelebA
uniform20 seed 0--9 manifests，不能与服务器上注册的 seed 10--19
端到端确认实验合并，也不能被描述为独立确认。

AUTO-D3M-Q 是查询动作空间适配：原 AUTO-D3M 用 100 个 TRAK trials
识别并删除负 alignment 样本；本实现用一个冻结线性头 trial、64 维确定性
投影和类内归因 PCA，再把负 alignment 秩与 NoiseScore 相乘，以选择“查询并
纠正标签”的样本。因此它不是官方 AUTO-D3M 的精确复现。

## 固定条件

- 数据：CelebA，Eyeglasses 目标，Male 伪相关属性。
- 特征：既有 ImageNet 预训练 ResNet-50 冻结特征。
- 噪声：uniform20。
- 查询预算：10%（每 seed 16,277 个样本）。
- 外层 seed：0--9。
- 线性头：20 epochs，batch size 4096，AdamW，学习率 0.001，weight decay
  0.0001，以 clean-validation balanced accuracy 选择 checkpoint。
- 方法：Loss、TracIn-CP (single)、TracIn-CP (multi)、RepairValue (tail)、
  AUTO-D3M-Q (adapt.)。

## 完整结果

区间为按外层 seed 重采样的 50,000 次 percentile bootstrap 95% CI；所有
WGA 数值均为百分点。

| 方法 | Delta WGA vs 不修正 | 配对 vs Loss | 查询噪声精度 | 负收益 seed |
|---|---:|---:|---:|---:|
| Loss | +4.84 [+3.39, +5.94] | -- | 99.31% | 1/10 |
| TracIn-CP (single) | -6.62 [-8.01, -5.40] | -11.45 [-12.33, -10.60] | 99.33% | 10/10 |
| TracIn-CP (multi) | -8.07 [-9.69, -6.62] | -12.90 [-13.75, -12.05] | 99.16% | 10/10 |
| RepairValue (tail) | +8.31 [+7.59, +8.94] | +3.47 [+2.56, +4.45] | 6.59% | 0/10 |
| AUTO-D3M-Q (adapt.) | +12.29 [+11.65, +12.92] | +7.45 [+6.34, +8.64] | 12.76% | 0/10 |

这组结果支持的只是机制方向：在当前固定特征条件中，更多 checkpoint 没有
修复 TracIn 的动作错位；把查询分数对齐到干净验证尾损失或伪弱组 alignment
更值得做独立端到端确认。低查询噪声精度但高 WGA 增益也说明“命中尽可能多的
错标”不是充分、也未必是必要的组修复排序目标。

## 复现与审计

运行：

```powershell
.\.venv\Scripts\python.exe -m robust_verify.cli.stage1 `
  --config configs\celeba_local_modern_baselines_uniform20_10pct.yaml

.\.venv\Scripts\python.exe scripts\analyze_local_modern_baselines.py
```

关键文件：

- 配置：`configs/celeba_local_modern_baselines_uniform20_10pct.yaml`
- 50 行原始结果：
  `outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/results.csv`
- 运行输入 manifest：
  `outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/run_manifest.json`
- 方法汇总与逐 seed 配对表：
  `outputs/local_celeba_modern_baselines_uniform20_10pct/audit/`
- 查询 ID 与模型 checkpoint：
  `outputs/local_celeba_modern_baselines_uniform20_10pct/stage1/queries/` 与
  `stage1/checkpoints/`。

SHA256：

- 配置文件：`390b1124dce5b145cb9dd6f2e75a31bcc0ad10a03770eec22c6893199c5b426a`
- 原始结果：`3d936ec6c2ddd0157e0936b50bff12aa542166265d8844d05b31d84f3805b007`
- 方法汇总：`0c50ad36ebee242a0687b76a17029f7623f7abce0c2a372524c235f32b049fa4`
