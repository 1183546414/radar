# -*- coding: utf-8 -*-
"""默认推荐打分器：Retrieve + Q-Ensemble 悲观打分（Round 20261007 采纳，20261024 重训）。

Round2（20260909）最优基线为 Retrieve+Q-Union；Round 20261007 在完全相同协议
（同数据/split/检索库/信号级真值）下验证了 5-seed QRetrievalNet 集成 + 悲观
打分 μ−λσ（λ=1.0）：
  * 对 q_only 配对胜率 96%（基线 90%），中位 regret 最优（−3.08 dB）；
  * 小样本分桶（稀有干扰类型 / 低检索相似度 / hard）全面最优，hard 桶胜率 100%。
20261022 算法结构修正后，集成成员已在 20261024 用新编码器重训
（outputs/20261024_ensemble_retrain，成员位于 outputs/q_value/latent_q_retrieval_ensemble），
新结构下的收益复验见该轮 eval_holdout_summary.json 与修正记录 md 12.8 节。

打分流程（与 eval_fewshot_round.py 的 ensemble_ctx_pessimistic 完全同源）：
  1. 干扰状态 S -> 冻结 VAE 潜向量 z_query
  2. 同类型 Top-5 相似训练场景 -> 检索候选池 + z_ctx（相似度 softmax 加权均值）
  3. 候选池 ∪ 随机 100 A -> union 池
  4. 5 个 QRetrievalNet 分别打分 -> μ、σ -> score = μ − λ·σ
  5. argmax score 输出推荐动作

改进② Set-Attention（QSetAttentionNet）本轮仅保留接口
（outputs/20261007_fewshot/code/attention_q_net.py），不进默认主流程。

用法:
    from radar.infer.q_ensemble import QEnsembleRecommender
    rec = QEnsembleRecommender()
    detail = rec.recommend_for_holdout(sid)          # 留出场景演示
    detail = rec.recommend(jammer_row, seed=sid)     # 任意干扰状态（pd.Series）

命令行演示（不做真值仿真）:
    python -m radar.infer.q_ensemble
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from radar.paths import OUT_Q_VALUE
from radar.infer import holdout_baselines as hb
from radar.train.q_retrieval_net import QRetrievalNet, predict_q_retrieval

# ---- Round 产出路径（20261024 起指向修正结构重训产物，见 12.8 节）----
DEFAULT_ENSEMBLE_DIR = OUT_Q_VALUE / "latent_q_retrieval_ensemble"

# ---- 默认推理配置（与 20261007 评估一致）----
DEFAULT_SEEDS = (42, 1042, 2042, 3042, 4042)
LAMBDA_PESSIMISTIC = 1.0
TOP_K_SIMILAR = 5
UNION_RANDOM_CANDIDATES = 100
HOLDOUT_SEED = 2026
MODEL_VERSION = "q-retrieval-ensemble-20261024-v1"


def _radar_action_dict(row: pd.Series) -> dict:
    """把训练库行转换成稳定、与存储列名解耦的部署输出。"""
    return {
        "radar_type": str(row["Radar_Type"]),
        "chirp_type": str(row["Radar_ChirpType"]),
        "carrier_freq_ghz": float(row["Radar_CarrierFreq_GHz"]),
        "gain_db": float(row["Radar_Gain_dB"]),
        "bandwidth_ghz": float(row["Radar_Bandwidth_GHz"]),
        "pulse_width_ns": float(row["Radar_PulseWidth_ns"]),
    }


def load_q_ensemble(
    models_dir: Path = DEFAULT_ENSEMBLE_DIR,
    seeds: tuple = DEFAULT_SEEDS,
    device: Optional[torch.device] = None,
) -> list[QRetrievalNet]:
    """加载 5 个不同 seed 的 QRetrievalNet 集成成员。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for seed in seeds:
        p = Path(models_dir) / f"seed{seed}" / "q_retrieval_best.pt"
        if not p.is_file():
            raise FileNotFoundError(f"缺少集成成员权重: {p}")
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = QRetrievalNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


class QEnsembleRecommender:
    """Retrieve + Q-Ensemble 悲观打分推荐器（当前默认打分方式）。

    Args:
        models_dir: 集成成员权重目录（默认 Round 20261007 产出）。
        seeds: 集成成员 seed 列表。
        lambda_pessimistic: 悲观系数 λ，score = μ − λ·σ。
        load_context: 是否立即加载检索上下文（数据 + VAE + 检索库）。
    """

    def __init__(
        self,
        models_dir: Path = DEFAULT_ENSEMBLE_DIR,
        seeds: tuple = DEFAULT_SEEDS,
        lambda_pessimistic: float = LAMBDA_PESSIMISTIC,
        device: Optional[torch.device] = None,
        load_context: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_pessimistic = lambda_pessimistic
        self.models = load_q_ensemble(models_dir, seeds, self.device)
        self.ctx = None
        if load_context:
            self.load_context()

    def load_context(self) -> None:
        """加载检索上下文（新管线数据 + 检索库 + 冻结 VAE，radar.paths 默认路径）。幂等。"""
        if self.ctx is None:
            self.ctx = hb.load_holdout_context(
                n_holdout=hb.N_HOLDOUT_SCENARIOS, holdout_seed=HOLDOUT_SEED
            )
        return self.ctx

    def recommend(
        self,
        jammer_row: pd.Series,
        seed: int,
        top_k: int = 5,
        pessimistic: bool = True,
    ) -> dict:
        """对一个干扰状态推荐雷达动作。

        Args:
            jammer_row: 干扰状态行（含 Jammer_* 列，与训练 CSV 同格式）。
            seed: 该查询的随机池种子——同一 seed 复现同一 union 池；
                评估协议用 holdout_seed + scenario_id，实际部署可传场景 id。
            top_k: 返回打分最高的前 k 个候选。
            pessimistic: True 用 μ−λ·σ，False 用 μ。

        Returns:
            dict(row_indices, scores, sigmas, neighbor_sids, sims, n_union)
            row_indices 为训练库行索引，row_indices[0] 即推荐动作。
        """
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("检索上下文未加载，请先调用 load_context()")

        _, z_q = hb._encode_query(ctx, jammer_row)
        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, TOP_K_SIMILAR,
            str(jammer_row["Jammer_Type"]), ctx.jammer_type_bank,
        )
        retrieval_cand = hb._collect_candidate_row_indices(
            ctx.train_scenario_ids, neighbor_sids
        )
        rng = np.random.default_rng(int(seed))
        union = hb.build_union_candidate_rows(
            retrieval_cand, ctx.train_row_pool, UNION_RANDOM_CANDIDATES, rng
        )
        z_ctx = hb._z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)

        a_mat = ctx.A_std[union]
        q_all = np.stack(
            [predict_q_retrieval(m, z_q.reshape(1, -1), a_mat, z_ctx.reshape(1, -1),
                                 self.device)
             for m in self.models]
        )
        mu = q_all.mean(axis=0)
        sigma = q_all.std(axis=0)
        score = mu - self.lambda_pessimistic * sigma if pessimistic else mu

        order = np.argsort(-score)[: min(top_k, len(union))]
        selected_rows = union[order]
        selected_mu = mu[order]
        sc_r = ctx.artifacts.get("sc_R")
        predicted_sinr = (
            sc_r.inverse_transform(selected_mu.reshape(-1, 1)).reshape(-1)
            if sc_r is not None
            else np.full(len(order), np.nan)
        )
        return {
            "model_version": MODEL_VERSION,
            "selection_rule": (
                f"mu-{self.lambda_pessimistic:g}*sigma" if pessimistic else "mu"
            ),
            "radar_action": _radar_action_dict(ctx.train_df.iloc[int(selected_rows[0])]),
            "predicted_sinr_db": float(predicted_sinr[0]),
            "uncertainty_std": float(sigma[order][0]),
            "row_indices": selected_rows.tolist(),
            "scores": score[order].tolist(),
            "sigmas": sigma[order].tolist(),
            "predicted_sinr_db_top_k": predicted_sinr.tolist(),
            "neighbor_sids": neighbor_sids.tolist(),
            "sims": np.asarray(sims).tolist(),
            "n_union": int(len(union)),
        }

    def recommend_for_holdout(self, sid: int, **kwargs) -> dict:
        """对留出 CSV 中场景 sid 的第一行干扰状态推荐（评估同款入口）。"""
        ctx = self.ctx
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        if len(block) == 0:
            raise KeyError(f"holdout CSV 中无场景 {sid}")
        return self.recommend(block.iloc[0], seed=HOLDOUT_SEED + int(sid), **kwargs)


def _demo() -> None:
    rec = QEnsembleRecommender()
    ctx = rec.ctx
    sids = list(ctx.holdout_sids)[:3]
    for sid in sids:
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        d = rec.recommend_for_holdout(sid)
        print(f"\n=== 场景 {sid} | 干扰类型 {jrow['Jammer_Type']} ===")
        print(f"Top-5 邻居场景: {d['neighbor_sids']} | 相似度: "
              f"{[round(s, 4) for s in d['sims']]}")
        print(f"union 池 {d['n_union']} 候选，悲观打分 Top-5（行索引, score, σ）:")
        for r, s, sg in zip(d["row_indices"], d["scores"], d["sigmas"]):
            print(f"  row={r}  score={s:.4f}  sigma={sg:.4f}")
        print("推荐动作 A_raw:", ctx.train_df.iloc[d["row_indices"][0]].to_dict())


if __name__ == "__main__":
    _demo()
