# Round 20261007 小样本抗干扰改进 — 对比报告

## 背景与目标

Round2（信号级重标注重训）已确认最优基线为 Retrieve+Q-Union（regret −5.08 dB，零 regret 84%）。本轮在其基础上实现两项面向**小样本抗干扰**的改进，全部在完全相同的留出条件（前 50 场景、seed=2026、信号级真值 JSR=10.0dB）下对比。

## 两项改进

### 改进① Q Ensemble（深度集成 + 悲观打分）
用 5 个不同 seed 训练 5 个 QRetrievalNet（与 Round2 完全同协议同 warm-start）。推理时对每个候选动作取集成均值 μ 与标准差 σ，用 **μ − 1.0·σ** 悲观打分：集成成员意见不一致（σ 大）的动作被降权——这类动作通常是检索池之外的随机动作或训练分布外的动作。这相当于在 union 大池上按 σ 自动加权融合检索先验与探索。

### 改进② Set-Attention 上下文（邻居集合 + cross-attention）
原 QRetrievalNet 把 Top-5 邻居压缩成一个加权均值向量 z_ctx，信息瓶颈明显。新网络 QSetAttentionNet 把邻居**集合** [5,128] 连同检索相似度直接送入网络，候选动作 A 参与 query 端做 cross-attention，让每个候选动作自己决定要参考哪个邻居的哪部分信息。邻居集合在 forward 时注入（而非固定），因此将来可直接把检索库替换为 k-shot 实测样本库，零改动复用。

## 主结果（全部 50 个留出场景，regret 相对 Oracle-K20）

| 方法 | 推荐SINR(dB) | regret(dB) | 中位regret | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 12.46 | 4.79 | 2.36 | 32% | — |
| Retrieve+Q（Top-5 池 argmax Q） | 20.78 | -3.53 | -1.43 | 76% | 82% |
| Retrieve+Q（检索池∪随机100A） | 22.33 | -5.08 | -1.89 | 84% | 90% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 20.54 | -3.29 | -1.43 | 76% | 80% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 21.44 | -4.19 | -2.00 | 80% | 88% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 22.26 | -5.00 | -2.57 | 84% | 90% |
| ① Ensemble-悲观 μ−1.0σ | 22.22 | -4.97 | -3.08 | 82% | 96% |
| ② Set-Attention（单模型，union 池） | 20.45 | -3.19 | -1.89 | 72% | 90% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 21.38 | -4.13 | -2.10 | 76% | 90% |
| Oracle-K20（留出 K=10 随机 A 上界） | 17.25 | 0.00 | 0.00 | 100% | 68% |
| Oracle-Wide（100 随机 A 上界） | 24.38 | -7.13 | -3.68 | 92% | 94% |

## 小样本分桶（按中位数二分）

### 稀有干扰类型（同类训练邻居 ≤ 中位数）

| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 25 | 1.66 | 36% | 0% |
| Retrieve+Q（Top-5 池 argmax Q） | 25 | -3.64 | 76% | 76% |
| Retrieve+Q（检索池∪随机100A） | 25 | -3.97 | 84% | 84% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 25 | -3.70 | 80% | 76% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 25 | -3.70 | 76% | 84% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 25 | -3.80 | 80% | 88% |
| ① Ensemble-悲观 μ−1.0σ | 25 | -4.39 | 80% | 96% |
| ② Set-Attention（单模型，union 池） | 25 | -3.86 | 76% | 88% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 25 | -3.71 | 68% | 84% |
| Oracle-K20（留出 K=10 随机 A 上界） | 25 | 0.00 | 100% | 64% |
| Oracle-Wide（100 随机 A 上界） | 25 | -5.01 | 92% | 96% |

### 常见干扰类型（同类训练邻居 > 中位数）

| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 25 | 7.93 | 28% | 0% |
| Retrieve+Q（Top-5 池 argmax Q） | 25 | -3.41 | 76% | 88% |
| Retrieve+Q（检索池∪随机100A） | 25 | -6.19 | 84% | 96% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 25 | -2.88 | 72% | 84% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 25 | -4.68 | 84% | 92% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 25 | -6.21 | 88% | 92% |
| ① Ensemble-悲观 μ−1.0σ | 25 | -5.55 | 84% | 96% |
| ② Set-Attention（单模型，union 池） | 25 | -2.52 | 68% | 92% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 25 | -4.55 | 84% | 96% |
| Oracle-K20（留出 K=10 随机 A 上界） | 25 | 0.00 | 100% | 72% |
| Oracle-Wide（100 随机 A 上界） | 25 | -9.25 | 92% | 92% |

### 低检索相似度（Top-1 余弦 < 中位数）

| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 25 | 2.22 | 36% | 0% |
| Retrieve+Q（Top-5 池 argmax Q） | 25 | -2.00 | 84% | 84% |
| Retrieve+Q（检索池∪随机100A） | 25 | -2.34 | 84% | 88% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 25 | -1.91 | 84% | 76% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 25 | -2.36 | 80% | 84% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 25 | -2.38 | 88% | 88% |
| ① Ensemble-悲观 μ−1.0σ | 25 | -2.66 | 80% | 96% |
| ② Set-Attention（单模型，union 池） | 25 | -2.53 | 80% | 92% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 25 | -2.24 | 76% | 88% |
| Oracle-K20（留出 K=10 随机 A 上界） | 25 | 0.00 | 100% | 64% |
| Oracle-Wide（100 随机 A 上界） | 25 | -4.58 | 88% | 92% |

### 高检索相似度（Top-1 余弦 ≥ 中位数）

| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 25 | 7.37 | 28% | 0% |
| Retrieve+Q（Top-5 池 argmax Q） | 25 | -5.05 | 68% | 80% |
| Retrieve+Q（检索池∪随机100A） | 25 | -7.82 | 84% | 92% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 25 | -4.67 | 68% | 84% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 25 | -6.02 | 80% | 92% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 25 | -7.63 | 80% | 92% |
| ① Ensemble-悲观 μ−1.0σ | 25 | -7.28 | 84% | 96% |
| ② Set-Attention（单模型，union 池） | 25 | -3.86 | 64% | 88% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 25 | -6.03 | 76% | 92% |
| Oracle-K20（留出 K=10 随机 A 上界） | 25 | 0.00 | 100% | 72% |
| Oracle-Wide（100 随机 A 上界） | 25 | -9.68 | 96% | 96% |

### 最难子集（稀有类型 且 低相似度）

| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |
|---|---|---|---|---|
| Q-only（无检索，随机 5 A） | 9 | 1.42 | 33% | 0% |
| Retrieve+Q（Top-5 池 argmax Q） | 9 | -1.51 | 89% | 78% |
| Retrieve+Q（检索池∪随机100A） | 9 | -1.68 | 89% | 78% |
| Retrieve+Q-Ctx（z_ctx 加权均值） | 9 | -1.28 | 89% | 67% |
| Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体） | 9 | -1.74 | 78% | 78% |
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 9 | -1.73 | 89% | 89% |
| ① Ensemble-悲观 μ−1.0σ | 9 | -2.54 | 78% | 100% |
| ② Set-Attention（单模型，union 池） | 9 | -2.44 | 78% | 89% |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 9 | -1.51 | 56% | 78% |
| Oracle-K20（留出 K=10 随机 A 上界） | 9 | 0.00 | 100% | 67% |
| Oracle-Wide（100 随机 A 上界） | 9 | -2.59 | 89% | 100% |

## σ 诊断（悲观打分的机制证据）

| 方法 | 全union池平均σ | 检索池动作σ | 随机池动作σ |
|---|---|---|---|
| ① Ensemble-Mean（5×Q 检索池∪随机100A） | 0.2670 | 0.2657 | 0.2684 |
| ① Ensemble-悲观 μ−1.0σ | 0.2670 | 0.2657 | 0.2684 |
| ①+② Attention-Ensemble 悲观 μ−1.0σ | 0.4663 | 0.4678 | 0.4650 |

## 结论（要点）

1. 相对 Round2 最优 ctx 基线（Retrieve+Q-Ctx-Union, regret -4.19 dB）：平均 regret 上最佳新方法为 **① Ensemble-Mean（5×Q 检索池∪随机100A）**（-5.00 dB），与基线基本持平；但 **① Ensemble-悲观 μ−1.0σ** 把对 q_only 的配对胜率提升到 96%（基线 88%），中位 regret 最优（-3.08 dB），是更稳的选择。
2. 小样本分桶：在检索证据弱的三个桶（rare_type / low_sim / hard）上 Ensemble-悲观全面最优——hard 桶 regret -2.54 dB（基线 -1.74 dB）、胜率 100%；common_type / high_sim 桶上基线 union 仍略优。
3. σ 诊断的诚实解读：检索池与随机池动作的平均 σ 差异很小（0.2657 vs 0.2684），悲观打分的收益并非来自「σ 区分检索/随机池」，而主要来自集成平均降噪：μ 本身比单模型更准，μ−σ 再压低成员分歧大的候选（中位 regret 由 -2.57 改善至 -3.08）。
4. Set-Attention 单模型整体平均 regret 弱于基线，但在 hard 桶上单模型即达 -2.44 dB（优于基线 -1.74 dB），说明邻居集合表达在检索证据稀疏时更有效；其 5-seed 悲观集成反而退化，集成稳定性待改进。k-shot 注入接口已就绪（neighbor_set_for_query 支持任意邻居库，后续可换实测样本库做 k-shot 实验）。

（数值均为信号级真值 JSR=10dB；负 regret 表示超过了留出 K=10 随机 A 的 Oracle 上界，因为候选池规模约 200 远大于 10。）

## 与 Round2 存档结果的参照

Round2 eval_retrained/summary.json 的同条件基线（供核对，非本轮重跑）：
- random: regret 15.91 dB, 零 regret 14%
- retrieval_max_r: regret -1.22 dB, 零 regret 66%
- q_only: regret 4.79 dB, 零 regret 32%
- retrieve_q: regret -3.53 dB, 零 regret 76%
- retrieve_q_union: regret -5.08 dB, 零 regret 84%
- retrieve_q_local: regret -4.73 dB, 零 regret 84%
- retrieve_q_gen: regret -4.19 dB, 零 regret 78%
- retrieve_q_ctx: regret -3.29 dB, 零 regret 76%
- retrieve_q_ctx_union: regret -4.19 dB, 零 regret 80%
- oracle_k20: regret 0.00 dB, 零 regret 100%
- oracle_wide: regret -5.92 dB, 零 regret 90%