# 信号级真值最终评估（JSR=10dB，50 holdout 场景）

> 本报告仅呈现重训后（Round2，信号级标签）的最终成绩。重训前的历史对比数据见 `archive/`，逐场景配对明细见 `comparison_paired_scenarios.csv`。

## 1. 设置

| 项目 | 说明 |
| --- | --- |
| Q 模型 | 本轮重训（信号级重标注 + 信号级 ranking oracle） |
| 训练标签 | 信号级 SINR（74,860 仿真 + 25,140 相位编码解析回退） |
| ranking/早停 oracle | RadarSimPy 信号级（与标签同源） |
| 评估真值 | 信号级（RadarSimPy，同后端同 JSR） |
| holdout 场景 | 前 50 个（seed=2026） |

## 2. 各方法对比（真值 = 信号级 SINR）

| 方法 | 推荐动作信号级SINR (dB) | Regret vs K20 (dB) | Regret中位数 (dB) | Oracle-Wide (dB) | 零Regret占比 |
| --- | --- | --- | --- | --- | --- |
| random | 1.34 | 15.91 | 17.29 | 24.69 | 0.14 |
| retrieval_max_r | 18.47 | -1.22 | -1.17 | 23.73 | 0.66 |
| q_only | 12.46 | 4.79 | 2.36 | 23.73 | 0.32 |
| retrieve_q | 20.78 | -3.53 | -1.43 | 23.03 | 0.76 |
| retrieve_q_union | 22.33 | -5.08 | -1.89 | 23.15 | 0.84 |
| retrieve_q_local | 21.98 | -4.73 | -2.34 | 23.50 | 0.84 |
| retrieve_q_gen | 21.44 | -4.19 | -1.57 | 23.72 | 0.78 |
| retrieve_q_ctx | 20.54 | -3.29 | -1.43 | 24.17 | 0.76 |
| retrieve_q_ctx_union | 21.44 | -4.19 | -2.00 | 24.03 | 0.80 |
| oracle_k20 | 17.25 | 0.00 | 0.00 | 23.72 | 1.00 |
| oracle_wide | 23.18 | -5.92 | -3.53 | 23.18 | 0.90 |

## 3. 配对胜率 vs Q-only（逐场景）

| 方法 | 胜率 |
| --- | --- |
| random | 26% |
| retrieval_max_r | 76% |
| retrieve_q | 82% |
| retrieve_q_union | 90% |
| retrieve_q_local | 94% |
| retrieve_q_gen | 92% |
| retrieve_q_ctx | 80% |
| retrieve_q_ctx_union | 88% |
| oracle_k20 | 68% |
| oracle_wide | 96% |

## 4. 参考基线（真值上界）

- Oracle-K20 均值: 17.25 dB（同 K=20 随机动作池）
- Oracle-Wide 均值: 23.18 dB（100 随机动作上界）

## 5. 结论要点

- **retrieve_q_union**（主方案）: 推荐 SINR 22.33 dB，Regret vs K20 -5.08 dB，零 Regret 占比 84%，对 q_only 胜率 90%
- **retrieve_q_local**: 推荐 SINR 21.98 dB，零 Regret 占比 84%，对 q_only 胜率 94%（最高）
- **q_only**（对照，无检索）: 推荐 SINR 12.46 dB
- 主方案推荐 SINR 距 Oracle-Wide 上界（23.18 dB）约 0.85 dB

## 6. 本轮产物清单（outputs/20260909_signal_level/）

| 路径 | 内容 |
| --- | --- |
| `archive/` | 重训前产物（radarsimpy 两轮验证 + 信号级留出评估 Round1，仅存档） |
| `data/` | 重标注 CSV（100,000 行信号级 SINR）+ relabel_report.json |
| `processed/` | 本轮重建的 (S,A,R) 标准化数据（R 为信号级） |
| `models/latent_q/` | 重训 QLatentNet（q_latent_best.pt + metrics.json + 潜向量缓存） |
| `models/latent_q_retrieval/` | 重训 QRetrievalNet（q_retrieval_best.pt + metrics.json） |
| `eval_retrained/` | 重训后信号级留出评估（summary.json/csv + per_scenario_detail.json） |
| `code/` | signal_sinr.py / relabel_signal_level.py / train_signal_round.py / eval_signal_round.py / compare_rounds.py |
| `train_report.json` | 重训三阶段耗时与后端统计 |
| `comparison_table.csv` / `comparison_report.md` | 本报告 |
