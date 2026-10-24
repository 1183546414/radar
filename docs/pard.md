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

## 单场景推理

准备只包含干扰状态的 JSON（类型不适用的参数可省略，程序会填充为 0）：

```json
{
  "Jammer_Type": "jammer_wide_band",
  "Jammer_CarrierFreq_GHz": 4.2,
  "Jammer_Gain_dB": 32.0,
  "Jammer_Bandwidth_GHz": 0.30
}
```

```bash
PYTHONPATH=src python -m radar.pard.recommend \
  --checkpoint outputs/pard/pard_best.pt \
  --state jammer_state.json \
  --output outputs/pard/recommendation.json
```

命令会直接输出雷达类型、调频类型、载频、增益、带宽、脉宽，以及经过干扰参数扰动最坏情况筛选后的 `robust_score`。它不调用仿真器。

## 与既有算法的批量对比

```bash
PYTHONPATH=src python -m radar.pard.evaluate \
  --checkpoint outputs/pard/pard_best.pt \
  --csv data/simulated/simulated_radar_jamming_high_contrast_10k.csv \
  --output outputs/pard_eval
```

该命令以与训练完全一致的 `seed=20261024`、按完整 `scenario_id` 留出的测试集评估，输出：

- `pard_per_scenario.csv`：逐场景推荐动作、推荐 SINR、K 候选 oracle、regret、Hit@1；
- `pard_summary.json`：mean / median / p95 regret、平均推荐 SINR、Hit@1。

如果已有 RadarSimPy 稠密 oracle 表（必须含 `scenario_id` 和 `dense_oracle_sinr_db`，或 `oracle_sinr_db`），加入 `--dense-oracle path/to/oracle.csv`。报告将额外输出 `mean_true_regret_db`、中位数和 p95，因而可以与项目的真 regret 主口径直接横向比较。

上面的默认 `recorded` 模式将 PARD 与 K 个已记录动作上的其他排序方法公平对比。端到端连续动作对比请使用 `cem` 模式，并传入信号级评估适配器：

```bash
PYTHONPATH=src python -m radar.pard.evaluate \
  --checkpoint outputs/pard/pard_best.pt --csv data/simulated/simulated_radar_jamming_high_contrast_10k.csv \
  --action-mode cem --scorer your_project.radarsimpy_backend:score \
  --dense-oracle outputs/dense_oracle_actions.csv --output outputs/pard_eval_cem
```

适配器是一个函数 `score(state: dict, action: dict) -> float`，内部应调用与既有方法相同的 RadarSimPy 信号级 SINR 后端。这样 PARD 的 CEM 连续推荐动作、既有算法动作和稠密 oracle 全部由同一个后端打分，指标才能有效比较。

## 推荐评估

使用项目现有信号级 RadarSimPy 留出协议，但将新方法与以下基线并列：Retrieve+Q-Union、Q-Ensemble、ICDT。主指标使用稠密 oracle 真 regret；同时报告 2%、5%、10%、20% 干扰参数扰动下的 mean/median/p95 regret、最坏方向退化和动作翻转率。

PARD 的训练模块和推理优化器均不会隐式运行。`RobustCEMOptimizer` 只输出模型推荐的动作，不调用仿真器；信号级精评估应由单独的评估脚本显式触发。
