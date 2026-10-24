# -*- coding: utf-8 -*-
"""ICDT 失效机制诊断：打分质量 vs 选择质量。

对前 150 个留出场景，在同一 union 池上对比：
  - ICDT（5-seed 集成均值打分）
  - 检索条件 Q（单模型，retrieve_q_union 同款）
指标：池内 Spearman 相关（排序质量）、池内 RMSE/R²（回归质量）、Hit@1（选择质量）。
若 ICDT 回归不差但排序/Hit@1 差 → 印证"训练目标（MSE）与任务（池内排序）错配"。
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
from radar.train.q_retrieval_net import QRetrievalNet, predict_q_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ICDT_ROUND = ROUND.parent / "20261023_icdt_retrain"
ICDT_DIR = ICDT_ROUND / "models" / "icdt_ensemble"
Q_RETRIEVAL_CKPT = hb.RETRIEVAL_MODEL_PATH
N_EVAL = 150
HOLDOUT_SEED = 2026

sys.path.insert(0, str(ICDT_ROUND / "code"))
from smoke_icdt import (A_DIM, FEAT_DIM, ICDTSmoke, K_NEIGHBORS, N_OBS_SLOTS,  # noqa: E402
                        SEQ_LEN, TYPE_CAND, TYPE_NB, TYPE_QUERY, Z_DIM)


def load_icdt_models():
    models = []
    for s in (42, 1042, 2042, 3042, 4042):
        net = ICDTSmoke().to(DEVICE)
        net.load_state_dict(torch.load(ICDT_DIR / f"seed{s}" / "icdt_best.pt",
                                       map_location=DEVICE, weights_only=True))
        net.eval()
        models.append(net)
    return models


def load_q_retrieval():
    ckpt = torch.load(Q_RETRIEVAL_CKPT, map_location=DEVICE, weights_only=False)
    m = QRetrievalNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
def icdt_scores(models, z_q, union, ctx, R_std_all, scen_full, neighbor_sids, sims):
    B = len(union)
    toks = np.zeros((B, SEQ_LEN, FEAT_DIM), dtype=np.float32)
    pad = np.ones((B, SEQ_LEN), dtype=bool)
    t_base = A_DIM + 2 + Z_DIM
    for k, nb_sid in enumerate(neighbor_sids[:K_NEIGHBORS]):
        rows = np.where(scen_full == int(nb_sid))[0]
        best = int(rows[int(np.argmax(R_std_all[rows]))])
        toks[:, N_OBS_SLOTS + k, :A_DIM] = ctx.A_std[best]
        toks[:, N_OBS_SLOTS + k, A_DIM] = R_std_all[best]
        toks[:, N_OBS_SLOTS + k, A_DIM + 1] = float(sims[k])
        toks[:, N_OBS_SLOTS + k, t_base + TYPE_NB] = 1.0
    toks[:, N_OBS_SLOTS + K_NEIGHBORS, A_DIM + 2:A_DIM + 2 + Z_DIM] = z_q
    toks[:, N_OBS_SLOTS + K_NEIGHBORS, t_base + TYPE_QUERY] = 1.0
    toks[:, -1, :A_DIM] = ctx.A_std[union]
    toks[:, -1, t_base + TYPE_CAND] = 1.0
    pad[:, N_OBS_SLOTS:] = False
    tt = torch.from_numpy(toks).to(DEVICE)
    pp = torch.from_numpy(pad).to(DEVICE)
    return np.mean([net(tt, pp).cpu().numpy() for net in models], axis=0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    icdt_models = load_icdt_models()
    q_model = load_q_retrieval()
    ctx = hb.load_holdout_context(n_holdout=hb.N_HOLDOUT_SCENARIOS, holdout_seed=HOLDOUT_SEED)

    r_mu = float(np.load(hb.DATA_V2_PROCESSED / "R_raw.npy").mean())
    r_sd = float(np.load(hb.DATA_V2_PROCESSED / "R_raw.npy").std())
    scen_full = np.load(hb.SCENARIO_ID_PATH)

    out = {"n_eval": N_EVAL, "icdt": {}, "q_retrieval": {}}
    acc = {k: {"sp": [], "hit": [], "se": [], "top1_gap": []} for k in ("icdt", "q_retrieval")}
    t0 = time.perf_counter()
    for si, sid in enumerate(ctx.holdout_sids[:N_EVAL]):
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        _, z_q = hb._encode_query(ctx, jrow)
        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, 5,
            str(jrow["Jammer_Type"]), ctx.jammer_type_bank)
        retrieval_cand = hb._collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
        sid_rng = np.random.default_rng(HOLDOUT_SEED + int(sid))
        union = hb.build_union_candidate_rows(
            retrieval_cand, ctx.train_row_pool, hb.UNION_RANDOM_CANDIDATES, sid_rng)
        r_true = np.array([hb._sinr_on_query(jrow, ctx.train_df, int(r)) for r in union])
        r_true_std = (r_true - r_mu) / r_sd
        z_ctx = hb._z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)

        s_icdt = icdt_scores(icdt_models, z_q, union, ctx, np.load(
            hb.DATA_V2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32),
            scen_full, neighbor_sids, sims)
        s_q = predict_q_retrieval(q_model, z_q.reshape(1, -1), ctx.A_std[union],
                                  z_ctx.reshape(1, -1), DEVICE)

        for key, s in (("icdt", s_icdt), ("q_retrieval", s_q)):
            a = acc[key]
            a["sp"].append(spearman(s, r_true_std))
            pred_std = s * 1.0  # 两模型输出均为 R_std 尺度
            a["se"].append(float(np.sqrt(np.mean((pred_std - r_true_std) ** 2))))
            a["hit"].append(int(np.argmax(s)) == int(np.argmax(r_true_std)))
            a["top1_gap"].append(float(
                r_true_std[int(np.argmax(r_true_std))] - r_true_std[int(np.argmax(s))]))
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{N_EVAL} ({time.perf_counter()-t0:.0f}s)")

    for key in acc:
        a = acc[key]
        out[key] = {
            "spearman_mean": float(np.nanmean(a["sp"])),
            "pool_rmse_std": float(np.mean(a["se"])),
            "union_hit_at_1": float(np.mean(a["hit"])),
            "top1_true_gap_db": float(np.mean(a["top1_gap"])) * r_sd,
        }
    (ROUND / "icdt_diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
