# 2026-06-09 会话修改导出

## 源码（覆盖到原项目对应路径）

| 文件 | 改动摘要 |
|------|----------|
| `src/radar/train/retrieval_ranking.py` | 同类型 Top-5 检索；`build_jammer_type_bank` |
| `src/radar/train/q_retrieval_net.py` | 相似度加权 `z_ctx`；QLatentNet 热启动 |
| `src/radar/train/train_q_retrieval.py` | 全量 z_ctx 预计算；Hit@1 早停；重训流程 |
| `src/radar/infer/holdout_baselines.py` | `retrieve_q_ctx_union`；加权 z_ctx |
| `src/radar/infer/demo_recommend.py` | 演示检索同类型 Top-5 |

## 产出物（可选）

- `outputs/q_value/latent_q_retrieval/` — 重训后的 QRetrievalNet 权重与 z_ctx 缓存
- `outputs/logs/` — 训练与留出评估日志

## 恢复方式

将 `src/` 下文件复制回 radar 项目根目录对应位置；模型权重复制到 `outputs/q_value/latent_q_retrieval/`。
