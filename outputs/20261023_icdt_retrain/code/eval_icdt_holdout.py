# -*- coding: utf-8 -*-
"""ICDT（新编码器重训版）— 新留出协议（500 场景 × K=10）评估。

与 outputs/20261013_icdt/code/eval_icdt.py 的差异：
  1. 评估协议改用 radar.infer.holdout_baselines 的 500 场景留出集
     （与 outputs/q_value/holdout_baselines/summary.csv 同一真值、同一候选池、同一随机种子）
  2. ICDT 模型读本目录 models/icdt_ensemble（新编码器 z 重训）
  3. R 标准化参数取 data/v2/processed（新管线）
  4. 配对胜率直接对 holdout_baselines 的 per_scenario_detail.json 计算（vs q_only / union）

方法: icdt_m0（零观测）/ icdt_rand_m5（随机观测 5 次）/ icdt_act_m5（主动观测 5 次），
候选池均为 检索池 ∪ 随机 100A（与主方法 retrieve_q_union 相同），悲观打分 μ−1.0σ。

用法:
    python outputs/20261023_icdt_retrain/code/eval_icdt_holdout.py
"""
from __future__ import annotations

import csv
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
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

from radar.infer import holdout_baselines as hb
from radar.paths import DATA_V2_PROCESSED, OUT_Q_VALUE
from smoke_icdt import (A_DIM, FEAT_DIM, ICDTSmoke, K_NEIGHBORS, N_OBS_SLOTS,
                        SEQ_LEN, TYPE_CAND, TYPE_NB, TYPE_OBS, TYPE_QUERY,
                        Z_DIM)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ICDT_DIR = ROUND / "models" / "icdt_ensemble"
OUT_DIR = ROUND / "eval_icdt_holdout"
BASELINE_DIR = OUT_Q_VALUE / "holdout_baselines"

SEEDS = (42, 1042, 2042, 3042, 4042)
HOLDOUT_SEED = 2026          # 与 holdout_baselines 相同
TOP_K = 5
LAMBDA = 1.0                 # 悲观系数（与 20261007/20261013 轮默认一致）
N_HOLDOUT = 500

METHOD_LABELS = {
    "icdt_m0": "ICDT 零观测（0 次真值查询）",
    "icdt_rand_m5": "ICDT 随机观测 m=5（5 次真值查询）",
    "icdt_act_m5": "ICDT 主动观测 m=5（5 次真值查询）",
}
METHOD_ORDER = list(METHOD_LABELS)


def load_globals():
    """R 标准化参数与 scenario_id 全表（邻居经验 / 观测标准化用）。"""
    R_raw = np.load(DATA_V2_PROCESSED / "R_raw.npy").reshape(-1).astype(np.float64)
    R_std = np.load(DATA_V2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)
    mu, sd = float(R_raw.mean()), float(R_raw.std())
    assert np.abs((R_raw - mu) / sd - R_std).max() < 1e-4, "R 标准化参数校验失败"
    scen = np.load(DATA_V2_PROCESSED / "scenario_id.npy")
    return R_std, scen, mu, sd


def load_icdt_models():
    models = []
    for s in SEEDS:
        net = ICDTSmoke().to(DEVICE)
        net.load_state_dict(torch.load(ICDT_DIR / f"seed{s}" / "icdt_best.pt",
                                       map_location=DEVICE, weights_only=True))
        net.eval()
        models.append(net)
    return models


@torch.no_grad()
def icdt_union_scores(models, z_q, union_cand, ctx, R_std_all, scen_full,
                      neighbor_sids, sims, obs_rows, obs_rstd):
    """对候选池批量 ICDT 打分，返回 (mu, sigma) [n_cand]（序列构造与训练同分布）。"""
    B = len(union_cand)
    toks = np.zeros((B, SEQ_LEN, FEAT_DIM), dtype=np.float32)
    pad = np.ones((B, SEQ_LEN), dtype=bool)
    t_base = A_DIM + 2 + Z_DIM

    m = len(obs_rows)
    if m > 0:
        toks[:, :m, :A_DIM] = ctx.A_std[np.asarray(obs_rows)][None, :, :]
        toks[:, :m, A_DIM] = np.asarray(obs_rstd, dtype=np.float32)[None, :]
        toks[:, :m, t_base + TYPE_OBS] = 1.0
        pad[:, :m] = False
    for k, nb_sid in enumerate(neighbor_sids[:K_NEIGHBORS]):
        rows = np.where(scen_full == int(nb_sid))[0]
        best = int(rows[int(np.argmax(R_std_all[rows]))])
        toks[:, N_OBS_SLOTS + k, :A_DIM] = ctx.A_std[best]
        toks[:, N_OBS_SLOTS + k, A_DIM] = R_std_all[best]
        toks[:, N_OBS_SLOTS + k, A_DIM + 1] = float(sims[k])
        toks[:, N_OBS_SLOTS + k, t_base + TYPE_NB] = 1.0
    toks[:, N_OBS_SLOTS + K_NEIGHBORS, A_DIM + 2:A_DIM + 2 + Z_DIM] = z_q
    toks[:, N_OBS_SLOTS + K_NEIGHBORS, t_base + TYPE_QUERY] = 1.0
    toks[:, -1, :A_DIM] = ctx.A_std[union_cand]
    toks[:, -1, t_base + TYPE_CAND] = 1.0
    pad[:, N_OBS_SLOTS:] = False

    tt = torch.from_numpy(toks).to(DEVICE)
    pp = torch.from_numpy(pad).to(DEVICE)
    q_all = np.stack([net(tt, pp).cpu().numpy() for net in models])  # [S, B]
    return q_all.mean(0), q_all.std(0)


def eval_icdt_method(ctx, method, icdt_models, R_std_all, scen_full, r_mu, r_sd,
                     top_k=TOP_K):
    m_budget = int(method.rsplit("m", 1)[1])
    active = "act" in method
    rng = np.random.default_rng(HOLDOUT_SEED)
    recs, regs, oracles = [], [], []
    n_sims_total = 0
    for sid in ctx.holdout_sids:
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        oracle_k = hb._oracle_k20(block)
        _, z_q = hb._encode_query(ctx, jrow)
        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, top_k,
            str(jrow["Jammer_Type"]), ctx.jammer_type_bank)
        retrieval_cand = hb._collect_candidate_row_indices(ctx.train_scenario_ids,
                                                           neighbor_sids)
        sid_rng = np.random.default_rng(HOLDOUT_SEED + int(sid))
        union = hb.build_union_candidate_rows(retrieval_cand, ctx.train_row_pool,
                                              hb.UNION_RANDOM_CANDIDATES, sid_rng)

        obs_rows: list[int] = []
        obs_rstd: list[float] = []
        n_sims = 0

        def simulate(row_idx: int) -> float:
            nonlocal n_sims
            n_sims += 1
            return hb._sinr_on_query(jrow, ctx.train_df, int(row_idx))

        if not active:
            if m_budget > 0:
                picks = sid_rng.choice(union, size=min(m_budget, len(union)),
                                       replace=False)
                for p in picks:
                    obs_rows.append(int(p))
                    obs_rstd.append((simulate(p) - r_mu) / r_sd)
            mu, sig = icdt_union_scores(icdt_models, z_q, union, ctx, R_std_all,
                                        scen_full, neighbor_sids, sims,
                                        obs_rows, obs_rstd)
            score = mu - LAMBDA * sig
        else:
            for _step in range(m_budget):
                mu, sig = icdt_union_scores(icdt_models, z_q, union, ctx, R_std_all,
                                            scen_full, neighbor_sids, sims,
                                            obs_rows, obs_rstd)
                score = mu - LAMBDA * sig
                cand_idx = [i for i, r in enumerate(union)
                            if int(r) not in set(obs_rows)]
                if not cand_idx:
                    break
                pick_i = max(cand_idx, key=lambda i: score[i])
                pick = int(union[pick_i])
                obs_rows.append(pick)
                obs_rstd.append((simulate(pick) - r_mu) / r_sd)
            mu, sig = icdt_union_scores(icdt_models, z_q, union, ctx, R_std_all,
                                        scen_full, neighbor_sids, sims,
                                        obs_rows, obs_rstd)
            score = mu - LAMBDA * sig
        n_sims_total += n_sims
        best_row = int(union[int(np.argmax(score))])
        r_rec = hb._sinr_on_query(jrow, ctx.train_df, best_row)
        recs.append(r_rec)
        regs.append(oracle_k - r_rec)
        oracles.append(oracle_k)

    rec_arr, reg_arr = np.array(recs), np.array(regs)
    return {
        "method": method, "label": METHOD_LABELS[method], "n_evaluated": len(recs),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "mean_oracle_k_db": float(np.mean(oracles)),
        "mean_regret_vs_k_db": float(reg_arr.mean()),
        "median_regret_vs_k_db": float(np.median(reg_arr)),
        "frac_zero_regret_vs_k": float((reg_arr <= 1e-4).mean()),
        "total_truth_queries": int(n_sims_total),
        "lambda": LAMBDA,
        "per_scenario_recommended_sinr_db": rec_arr.tolist(),
        "per_scenario_regret_vs_k_db": reg_arr.tolist(),
    }


def paired_win(rec_a: list[float], rec_b: list[float]) -> dict:
    a = np.asarray(rec_a)
    b = np.asarray(rec_b)
    n = len(a)
    wins = int((a > b + 1e-6).sum())
    ties = int((np.abs(a - b) <= 1e-6).sum())
    return {"wins": wins, "ties": ties, "losses": n - wins - ties,
            "win_rate": wins / n if n else float("nan")}


def main():
    t0 = time.perf_counter()
    R_std_all, scen_full, r_mu, r_sd = load_globals()
    icdt_models = load_icdt_models()
    print(f"ICDT ×{len(icdt_models)} | device={DEVICE}")

    ctx = hb.load_holdout_context(n_holdout=N_HOLDOUT)
    print(f"留出集: {len(ctx.holdout_sids)} 场景 × K={ctx.holdout_df.groupby('scenario_id').size().iloc[0]}")

    results = []
    per_scenario: dict[str, dict] = {}
    for m in METHOD_ORDER:
        t_m = time.perf_counter()
        row = eval_icdt_method(ctx, m, icdt_models, R_std_all, scen_full, r_mu, r_sd)
        per_scenario[m] = {
            "recommended_sinr_db": row.pop("per_scenario_recommended_sinr_db"),
            "regret_vs_k_db": row.pop("per_scenario_regret_vs_k_db"),
        }
        results.append(row)
        print(f"--- {m}: rec={row['mean_recommended_sinr_db']:.2f} dB, "
              f"regret={row['mean_regret_vs_k_db']:+.3f} dB, "
              f"零regret={row['frac_zero_regret_vs_k']:.1%}, "
              f"查询 {row['total_truth_queries']} 次 ({time.perf_counter()-t_m:.0f}s)")

    # 配对胜率：vs holdout_baselines 的 q_only / retrieve_q_union 逐场景 SINR
    with open(BASELINE_DIR / "per_scenario_detail.json", encoding="utf-8") as f:
        base_detail = json.load(f)
    paired: dict[str, dict] = {}
    for ref in ("q_only", "retrieve_q_union"):
        ref_rec = base_detail[ref]["recommended_sinr_db"]
        paired[ref] = {
            m: paired_win(per_scenario[m]["recommended_sinr_db"], ref_rec)
            for m in METHOD_ORDER
        }

    summary = {
        "round": "20261023_icdt_retrain",
        "encoder": "outputs/state_vae/state_embeddings_mu.npy（20261023 修正结构重训）",
        "protocol": "与 outputs/q_value/holdout_baselines 完全一致（500 场景 × K=10, 解析真值, seed 2026）",
        "candidate_pool": "检索池(Top-5) ∪ 随机 100A（与 retrieve_q_union 相同）",
        "scoring": f"5-seed 集成, 悲观 μ−{LAMBDA}σ",
        "n_holdout_scenarios": len(ctx.holdout_sids),
        "methods": results,
        "paired_win_rate": paired,
        "elapsed_min": (time.perf_counter() - t0) / 60,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "per_scenario_detail.json").write_text(
        json.dumps(per_scenario, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["method", "label", "n_evaluated", "mean_recommended_sinr_db",
                  "mean_regret_vs_k_db", "median_regret_vs_k_db",
                  "frac_zero_regret_vs_k", "total_truth_queries"]
    with open(OUT_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f"\n=== 配对胜率 ===")
    for ref, d in paired.items():
        for m, st in d.items():
            print(f"  {m:<14} vs {ref:<18} 胜率 {100*st['win_rate']:.1f}%")
    print(f"完成，耗时 {summary['elapsed_min']:.1f} min | 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
