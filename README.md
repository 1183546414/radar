# Radar Anti-Jamming

雷达抗干扰场景的离线机器学习 / 强化学习实验项目。

## 目录结构

```
radar/
├── main.py                 # 主流程入口
├── requirements.txt
├── src/radar/              # 核心 Python 包
│   ├── paths.py            # 统一路径配置
│   ├── frozen_vae.py       # 冻结 VAE 编码（Q 网络等复用）
│   ├── data/               # 数据集构建 (S, A, R)
│   ├── train/              # VAE、Q 值等训练
│   ├── infer/              # 推理与相似检索
│   ├── analyze/            # 聚类等分析
│   ├── viz/                # 可视化
│   └── rl/                 # DQN 等 RL 实验
├── scripts/
│   ├── etl/                # CSV → JSON 等数据预处理
│   └── simulation/         # 仿真数据生成与匹配
├── data/
│   ├── v2/                 # 主训练数据 (json + processed)
│   ├── v1/                 # 早期数据
│   ├── simulated/          # 大规模仿真 CSV
│   └── radar_raw/          # 原始雷达 CSV
├── outputs/                # 训练产出与图表
│   ├── state_vae/
│   ├── q_value/
│   ├── checkpoints/
│   └── figures/
├── docs/                   # 设计文档与伪代码
└── archive/                # 历史代码与旧目录备份
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主流程（需在项目根目录）
python main.py
```

## 数据说明

| 路径 | 含义 |
|------|------|
| `data/simulated/simulated_radar_jamming_high_contrast_10k.csv` | 仿真主表（2500 场景 × 20 动作 = 5 万行） |
| `data/v2/processed/` | `S_std.npy`, `A_std.npy`, `R_std.npy`, `scenario_id.npy` |
| `outputs/state_vae/` | VAE 权重与潜向量 |

生成仿真 CSV（需先执行）:

```bash
python scripts/simulation/simulate_data.py   # NUM_STATES=2500, K=20
```

- **状态 S**：干扰类型 + 载频 + 干扰参数（同 `scenario_id` 下 K 行相同）  
- **动作 A**：雷达载频 + 雷达参数（每场景 K 种候选）  
- **奖励 R**：SINR（由 `calculate_sinr_high_contrast` 计算）  

## 单独运行模块

```bash
export PYTHONPATH=src

python -m radar.data.build_dataset
python -m radar.train.state_vae_contrastive
python -m radar.analyze.clustering
python -m radar.infer.holdout_baselines
python scripts/run_holdout_baselines.py   # 6 baseline 对比表
python -m radar.rl.train_dqn
```

## 修改路径

所有路径集中在 `src/radar/paths.py`，请勿在脚本中硬编码旧路径。

`main.py` 第 5 步：**留出 500 个未见干扰场景** → 对比学习 AE 编码 → 训练集潜空间同类型 Top-5 检索 → Q 选 A → 与留出随机 K 动作参考上界对比（`outputs/q_value/holdout_eval_metrics.json`）。该参考不是全局 Oracle；方法搜索更大候选池时允许出现负差值。

状态编码器按 `scenario_id` 去重并按场景切分，解码器只能从潜向量重建状态，不再使用输入直连。升级代码后必须依次重训状态编码器、基础 Q 和检索条件 Q；旧编码器会被明确拒绝加载。

部署推荐器为 `radar.infer.q_ensemble.QEnsembleRecommender`，输出包含物理雷达动作、预测 SINR、模型不确定性、检索邻居和模型版本。`main.py` 中的单模型 `Retrieve+Q-Union` 保留为可复现训练基线。

设计文档：`docs/offline_rl_pseudocode.py`（检索 + Q 打分 + 行为先验等离线 RL 方案）。
