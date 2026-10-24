# PARD：物理感知分布式鲁棒抗干扰决策

PARD（Physics-Aware Robust Decision）是与项目既有 VAE、相似检索、Q 网络、Q-Ensemble、ICDT 相互独立的新算法。它不读取旧模型权重、潜向量或检索索引；输入为仿真 CSV 的原始字段，输出为混合连续/离散雷达动作。

## 方法

原始状态字段与动作字段各自构成 token。数值 token 由标准化值、`log1p(abs(x))` 与 Fourier 特征构造；类别字段由新训练的 embedding 构造。Token 经 Transformer 交互，并结合频谱间隔、频带交叠、带宽比、增益差和时序比等七项物理交互特征，输出 7 个 SINR 条件分位数：0.05、0.10、0.25、0.50、0.75、0.90、0.95。

训练目标为：加权分位数损失 + 同场景 pairwise 排序损失 + 对状态数值字段的对抗一致性损失。数据只能按 `scenario_id` 整体划分，严禁按 CSV 行随机切分。

推理时，不使用历史场景检索。对各离散波形组合，CEM 在连续动作空间搜索；每个候选动作在干扰参数扰动集合中采样，按最差风险分数决策：

`risk = 0.65 * lower-tail-CVaR + 0.35 * (median - beta * quantile-spread)`。

## 服务器执行

```bash
cd radar
export PYTHONPATH=src
python -m radar.pard.train \
  --csv data/simulated/simulated_radar_jamming_high_contrast_10k.csv \
  --output outputs/pard
```

训练完成后，checkpoint `outputs/pard/pard_best.pt` 包含模型、字段词表、训练集统计量和超参数；`manifest.json` 保存严格按场景的切分规模与验证指标。

## 推荐评估

使用项目现有信号级 RadarSimPy 留出协议，但将新方法与以下基线并列：Retrieve+Q-Union、Q-Ensemble、ICDT。主指标使用稠密 oracle 真 regret；同时报告 2%、5%、10%、20% 干扰参数扰动下的 mean/median/p95 regret、最坏方向退化和动作翻转率。

PARD 的训练模块和推理优化器均不会隐式运行。`RobustCEMOptimizer` 只输出模型推荐的动作，不调用仿真器；信号级精评估应由单独的评估脚本显式触发。
