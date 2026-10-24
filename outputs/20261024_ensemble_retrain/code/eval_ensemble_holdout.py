# -*- coding: utf-8 -*-
"""重训 Q-Ensemble 收益复验 — 新留出协议（500 场景 × K=10）。

回答 20261022 全面实验遗留的复验问题：当时 Ensemble-Mean/悲观打分基于旧结构模型，
其"均值不显著优于 Union、但尾部（p95/CVaR）更优"的结论是否在新结构下成立。

方法（与 eval_fewshot_round.py 的 ensemble_ctx_pessimistic 同源）：
  候选池 = 检索池(Top-5) ∪ 随机 100A（与 retrieve_q_union 相同）
  打分 = 5-seed QRetrievalNet 集成 μ、σ → score = μ − λ·σ，argmax
对比基线直接读 holdout_baselines 的 per_scenario_detail.json（q_only / retrieve_q_union）。

用法:
    python outputs/20261024_ensemble_retrain/code/eval_ensemble_holdout.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import torch

from radar.infer import holdout_baselines as hb
from radar.paths import OUT_Q_VALUE
from radar.infer.q_ensemble import DEFAULT_ENSEMBLE_DIR, DEFAULT_SEEDS
from radar.train.q_retrieval_net import predict_q_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = ROUND / "eval_holdout_summary.json"
BASELINE_DIR = OUT_Q_VALUE / "holdout_baselines"
HOLDOUT_SEED = 2026
LAMBDA = 1.0


def load_members():
    models = []
    for s in DEFAULT_SEEDS:
        ckpt = torch.load(DEFAULT_ENSEMBLE_DIR / f"seed{s}" / "q_retrieval_best.pt",
                          map_location=DEVICE, weights_only=False)
        m = hb.QRetrievalNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def main():
    t0 = time.perf_counter()
    models = load_members()
    ctx = hb.load_holdout_context(n_holdout=hb.N_HOLDOUT_SCENARIOS, holdout_seed=HOLDOUT_SEED)
    print(f"Ensemble ×{len(models)} | 留出 {len(ctx.holdout_sids)} 场景 | device={DEVICE}")

    recs, regrets, q_mu_list, q_std_list = [], [], [], []
    rng = np.random.default_rng(HOLDOUT_SEED)
    for si, sid in enumerate(ctx.holdout_sids):
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        oracle_k = hb._oracle_k20(block)
        _, z_q = hb._encode_query(ctx, jrow)
        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, 5,
            str(jrow["Jammer_Type"]), ctx.jammer_type_bank)
        retrieval_cand = hb._collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
        sid_rng = np.random.default_rng(HOLDOUT_SEED + int(sid))
        union = hb.build_union_candidate_rows(
            retrieval_cand, ctx.train_row_pool, hb.UNION_RANDOM_CANDIDATES, sid_rng)
        z_ctx = hb._z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)

        a_mat = ctx.A_std[union]
        q_all = np.stack([
            predict_q_retrieval(m, z_q.reshape(1, -1), a_mat, z_ctx.reshape(1, -1), DEVICE)
            for m in models
        ])
        mu, sigma = q_all.mean(0), q_all.std(0)
        best_i = int(np.argmax(mu - LAMBDA * sigma))
        r_rec = hb._sinr_on_query(jrow, ctx.train_df, int(union[best_i]))
        recs.append(r_rec)
        regrets.append(oracle_k - r_rec)
        q_mu_list.append(float(mu[best_i]))
        q_std_list.append(float(sigma[best_i]))
        if verbose_progress(si):
            print(f"  {si+1}/{len(ctx.holdout_sids)} sid={sid} rec={r_rec:.2f}")

    rec_arr, reg_arr = np.array(recs), np.array(regrets)
    result = {
        "round": "20261024_ensemble_retrain",
        "method": "q_ensemble_ctx_pessimistic",
        "protocol": "500 场景 × K=10, 检索池∪随机100A, μ−1.0σ（与 holdout_baselines 同真值同种子）",
        "seeds": list(DEFAULT_SEEDS),
        "lambda": LAMBDA,
        "n_evaluated": len(recs),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "median_recommended_sinr_db": float(np.median(rec_arr)),
        "p95_recommended_sinr_db": float(np.percentile(rec_arr, 5)),
        "mean_regret_vs_k_db": float(reg_arr.mean()),
        "median_regret_vs_k_db": float(np.median(reg_arr)),
        "p95_regret_db": float(np.percentile(reg_arr, 95)),
        "cvar10_regret_db": float(reg_arr[reg_arr >= np.percentile(reg_arr, 90)].mean()),
        "frac_zero_regret_vs_k": float((reg_arr <= 1e-4).mean()),
        "per_scenario_recommended_sinr_db": recs,
        "per_scenario_regret_vs_k_db": regrets,
    }

    # 配对对比：vs q_only / retrieve_q_union（单模型）
    with open(BASELINE_DIR / "per_scenario_detail.json", encoding="utf-8") as f:
        base = json.load(f)
    paired = {}
    for ref in ("q_only", "retrieve_q_union"):
        rb = np.asarray(base[ref]["recommended_sinr_db"])
        wins = int((rec_arr > rb + 1e-6).sum())
        ties = int((np.abs(rec_arr - rb) <= 1e-6).sum())
        delta = rec_arr - rb
        paired[ref] = {
            "wins": wins, "ties": ties, "losses": len(recs) - wins - ties,
            "win_rate": wins / len(recs),
            "mean_delta_sinr_db": float(delta.mean()),
            "delta_p05_db": float(np.percentile(delta, 5)),
        }
    result["paired"] = paired

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n推荐 SINR mean={result['mean_recommended_sinr_db']:.2f} dB, "
          f"regret={result['mean_regret_vs_k_db']:+.3f} dB, "
          f"p95 regret={result['p95_regret_db']:.2f} dB, "
          f"CVaR10={result['cvar10_regret_db']:.2f} dB")
    for ref, d in paired.items():
        print(f"  vs {ref:<18} 胜率 {100*d['win_rate']:.1f}%  "
              f"平均ΔSINR {d['mean_delta_sinr_db']:+.3f} dB")
    print(f"耗时 {(time.perf_counter()-t0)/60:.1f} min | 输出: {OUT_PATH}")


def verbose_progress(si: int) -> bool:
    return si < 2 or (si + 1) % 100 == 0


if __name__ == "__main__":
    main()
