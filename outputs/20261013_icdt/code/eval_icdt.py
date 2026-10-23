# -*- coding: utf-8 -*-
"""ICDT（In-Context Decision Transformer）holdout 评估脚本。

协议与 20261007 轮完全一致（直接复用其 patch/基线选择函数，同 holdout、同真值）：
  前 50 留出场景、seed=2026、信号级真值 JSR=10dB、候选池 = 检索池 ∪ 随机 100A。

ICDT 方法族（核心新增）：
  * icdt_m0              无观测，纯上下文（z + 邻居经验）
  * icdt_rand_m{K}       随机 K 个候选动作做仿真观测 → 滚入上下文重打分
  * icdt_act_m{K}        悲观主动选点：迭代"打分→选未仿真候选中悲观分最高者→仿真
                         →滚入上下文"，K 次预算（主动上下文适配循环）
  观测动作均取自候选池内库行（真值用进程内 SignalSinrBackend，有缓存）。

基线对拍：q_only / retrieve_q_union（Round2 基线复现校验）/ ens_pessimistic
（20261007 已采纳默认）/ oracle_k20 / oracle_wide。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261013_icdt/code/eval_icdt.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
ROUND2 = HERE.parents[3] / "outputs" / "20260909_signal_level"
CODE07 = HERE.parents[3] / "outputs" / "20261007_fewshot" / "code"
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(CODE07))
sys.path.insert(0, str(ROUND2 / "code"))  # signal_sinr（SignalSinrBackend）

import numpy as np
import torch

import eval_fewshot_round as e07          # 20261007 轮评估模块（协议/patch 复用）
from eval_fewshot_round import hb, predict_q_retrieval  # noqa
from smoke_icdt import (A_DIM, D_MODEL, FEAT_DIM, ICDTSmoke, K_NEIGHBORS,
                         N_OBS_SLOTS, SEQ_LEN, TYPE_CAND, TYPE_NB, TYPE_OBS,
                         TYPE_QUERY, Z_DIM)

JSR_DB, N_HOLDOUT, HOLDOUT_SEED = 10.0, 50, 2026
TOP_K = 5
LAMBDA = 1.0
OUT_DIR = ROUND / "eval_icdt"
ICDT_DIR = ROUND / "models" / "icdt_ensemble"
SEEDS = (42, 1042, 2042, 3042, 4042)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_LABELS = {
    "q_only": "Q-only（无检索，随机 5 A）",
    "retrieve_q_union": "Retrieve+Q-Union（Round2 基线）",
    "ens_pessimistic": "Q-Ens-悲观 μ−1.0σ（20261007 默认）",
    "icdt_m0": "ICDT 零观测",
    "icdt_rand_m1": "ICDT 随机观测 m=1",
    "icdt_rand_m3": "ICDT 随机观测 m=3",
    "icdt_rand_m5": "ICDT 随机观测 m=5",
    "icdt_rand_m8": "ICDT 随机观测 m=8",
    "icdt_act_m1": "ICDT 主动观测 m=1",
    "icdt_act_m3": "ICDT 主动观测 m=3",
    "icdt_act_m5": "ICDT 主动观测 m=5",
    "icdt_act_m8": "ICDT 主动观测 m=8",
    "oracle_k20": "Oracle-K20（留出 K 随机 A 上界）",
    "oracle_wide": "Oracle-Wide（100 随机 A 上界）",
}
METHOD_ORDER = list(METHOD_LABELS)
ICDT_METHODS = [m for m in METHOD_ORDER if m.startswith("icdt_")]


def load_globals():
    """全表 R 标准化参数与 R_std/scenario_id（邻居经验/观测标准化用）。"""
    R_raw = np.load(ROUND2 / "processed" / "R_raw.npy").reshape(-1).astype(np.float64)
    R_std = np.load(ROUND2 / "processed" / "R_std.npy").reshape(-1).astype(np.float32)
    mu, sd = float(R_raw.mean()), float(R_raw.std())
    assert np.allclose((R_raw - mu) / sd, R_std, atol=1e-4), "R 标准化参数校验失败"
    scen = np.load(ROUND2 / "processed" / "scenario_id.npy")
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
                      neighbor_sids, sims, obs_rows, obs_rstd, device):
    """对候选池批量 ICDT 打分，返回 (mu, sigma) [n_cand]。

    序列构造与训练完全同分布：obs 槽 = 已仿真观测（评估时 m 固定）、
    nb 槽 = 邻居场景最优动作经验、query = z_q、cand = 候选动作。
    """
    B = len(union_cand)
    toks = np.zeros((B, SEQ_LEN, FEAT_DIM), dtype=np.float32)
    pad = np.ones((B, SEQ_LEN), dtype=bool)
    t_base = A_DIM + 2 + Z_DIM

    # obs 槽（固定 m，不足补零）
    m = len(obs_rows)
    if m > 0:
        toks[:, :m, :A_DIM] = ctx.A_std[np.asarray(obs_rows)][None, :, :]
        toks[:, :m, A_DIM] = np.asarray(obs_rstd, dtype=np.float32)[None, :]
        toks[:, :m, t_base + TYPE_OBS] = 1.0
        pad[:, :m] = False
    # nb 槽：邻居场景 argmax R_std 行（与训练一致的经验 token）
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

    tt = torch.from_numpy(toks).to(device)
    pp = torch.from_numpy(pad).to(device)
    q_all = np.stack([net(tt, pp).cpu().numpy() for net in models])  # [S, B]
    return q_all.mean(0), q_all.std(0)


def eval_icdt_method(ctx, method, icdt_models, R_std_all, scen_full, r_mu, r_sd,
                     top_k=TOP_K):
    """ICDT 方法族评估（含 rand/act 两种预算分配）。"""
    m_budget = int(method.rsplit("m", 1)[1])
    active = "act" in method
    rng = np.random.default_rng(HOLDOUT_SEED)
    recs, regs = [], []
    n_sims_total = 0
    for sid in ctx.holdout_sids:
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        oracle_k20 = hb._oracle_k20(block)
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
                                        obs_rows, obs_rstd, DEVICE)
            score = mu - LAMBDA * sig
        else:
            score = None
            for step in range(m_budget):
                mu, sig = icdt_union_scores(icdt_models, z_q, union, ctx, R_std_all,
                                            scen_full, neighbor_sids, sims,
                                            obs_rows, obs_rstd, DEVICE)
                score = mu - LAMBDA * sig
                cand_idx = [i for i, r in enumerate(union) if int(r) not in set(obs_rows)]
                if not cand_idx:
                    break
                pick_i = max(cand_idx, key=lambda i: score[i])
                pick = int(union[pick_i])
                obs_rows.append(pick)
                obs_rstd.append((simulate(pick) - r_mu) / r_sd)
            mu, sig = icdt_union_scores(icdt_models, z_q, union, ctx, R_std_all,
                                        scen_full, neighbor_sids, sims,
                                        obs_rows, obs_rstd, DEVICE)
            score = mu - LAMBDA * sig
        n_sims_total += n_sims
        best_row = int(union[int(np.argmax(score))])
        r_rec = hb._sinr_on_query(jrow, ctx.train_df, best_row)
        recs.append(r_rec)
        regs.append(oracle_k20 - r_rec)

    rec_arr, reg_arr = np.array(recs), np.array(regs)
    return {
        "method": method, "label": METHOD_LABELS[method], "n_evaluated": len(recs),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "mean_regret_vs_k20_db": float(reg_arr.mean()),
        "median_regret_vs_k20_db": float(np.median(reg_arr)),
        "frac_zero_regret_vs_k20": float((reg_arr <= 1e-4).mean()),
        "per_scenario_recommended_sinr_db": rec_arr.tolist(),
        "per_scenario_regret_vs_k20_db": reg_arr.tolist(),
        "total_truth_queries": int(n_sims_total),
    }


def eval_baseline_method(ctx, method, ens_models, device, top_k=TOP_K):
    """基线方法（q_only / retrieve_q_union / ens_pessimistic / oracles），逐场景。"""
    rng = np.random.default_rng(HOLDOUT_SEED)
    recs, regs = [], []
    for sid in ctx.holdout_sids:
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        oracle_k20 = hb._oracle_k20(block)
        _, z_q = hb._encode_query(ctx, jrow)
        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, top_k,
            str(jrow["Jammer_Type"]), ctx.jammer_type_bank)
        retrieval_cand = hb._collect_candidate_row_indices(ctx.train_scenario_ids,
                                                           neighbor_sids)
        sid_rng = np.random.default_rng(HOLDOUT_SEED + int(sid))
        if method == "q_only":
            row_idx = hb._select_q_only(ctx, z_q, int(sid), sid_rng)
            r_rec = hb._sinr_on_query(jrow, ctx.train_df, row_idx)
        elif method == "retrieve_q_union":
            row_idx, _ = hb._select_retrieve_q_union(ctx, z_q, retrieval_cand, sid_rng)
            r_rec = hb._sinr_on_query(jrow, ctx.train_df, row_idx)
        elif method == "ens_pessimistic":
            union = hb.build_union_candidate_rows(retrieval_cand, ctx.train_row_pool,
                                                  hb.UNION_RANDOM_CANDIDATES, sid_rng)
            z_ctx = hb._z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)
            a_mat = ctx.A_std[union]
            q_all = np.stack(
                [predict_q_retrieval(m, z_q.reshape(1, -1), a_mat,
                                     z_ctx.reshape(1, -1), device) for m in ens_models])
            score = q_all.mean(0) - LAMBDA * q_all.std(0)
            row_idx = int(union[int(np.argmax(score))])
            r_rec = hb._sinr_on_query(jrow, ctx.train_df, row_idx)
        elif method == "oracle_k20":
            r_rec = hb._holdout_truth(block, hb._select_oracle_k20(block))
        elif method == "oracle_wide":
            r_rec = hb._oracle_wide(jrow, hb.ORACLE_WIDE_CANDIDATES, rng)
        else:
            raise ValueError(method)
        recs.append(r_rec)
        regs.append(oracle_k20 - r_rec)
    rec_arr, reg_arr = np.array(recs), np.array(regs)
    return {
        "method": method, "label": METHOD_LABELS[method], "n_evaluated": len(recs),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "mean_regret_vs_k20_db": float(reg_arr.mean()),
        "median_regret_vs_k20_db": float(np.median(reg_arr)),
        "frac_zero_regret_vs_k20": float((reg_arr <= 1e-4).mean()),
        "per_scenario_recommended_sinr_db": rec_arr.tolist(),
        "per_scenario_regret_vs_k20_db": reg_arr.tolist(),
    }


def _bucket_stats(regs, mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"n": 0}
    r = regs[idx]
    return {"n": int(len(idx)), "mean_regret_vs_k20_db": float(r.mean()),
            "median_regret_vs_k20_db": float(np.median(r)),
            "frac_zero_regret_vs_k20": float((r <= 1e-4).mean())}


def _paired(ref, rec, mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"n": 0, "win_rate": float("nan")}
    d = rec[idx] - ref[idx]
    wins = int((d > 1e-6).sum())
    return {"n": int(len(idx)), "wins": wins,
            "ties": int((np.abs(d) <= 1e-6).sum()), "losses": int((d < -1e-6).sum()),
            "win_rate": wins / len(idx)}


def write_report(summary, path):
    L = ["# Round 20261013 ICDT（上下文决策 Transformer）评估报告", "",
         f"协议：前 {N_HOLDOUT} 留出场景、seed={HOLDOUT_SEED}、信号级真值 JSR={JSR_DB}dB、"
         "候选池=检索池∪随机100A，与 20261007 轮完全同条件。", "",
         "## 主结果（regret vs Oracle-K20，dB）", "",
         "| 方法 | regret | 中位 | 零regret | 胜率 vs q_only | 真值查询数 |",
         "|---|---|---|---|---|---|"]
    paired = summary["paired_vs_q_only"]
    for r in summary["methods"]:
        m = r["method"]
        wr = paired["results"].get(m, {}).get("win_rate")
        wr_s = f"{wr*100:.0f}%" if wr is not None else "—"
        nq = r.get("total_truth_queries", "—")
        L.append(f"| {r['label']} | {r['mean_regret_vs_k20_db']:+.2f} | "
                 f"{r['median_regret_vs_k20_db']:+.2f} | "
                 f"{r['frac_zero_regret_vs_k20']*100:.0f}% | {wr_s} | {nq} |")
    L += ["", "## 小样本分桶（mean regret dB，n）", "",
          "| 方法 | rare_type | low_sim | hard | common_type | high_sim |", "|---|---|---|---|---|---|"]
    for r in summary["methods"]:
        m = r["method"]
        b = summary["buckets"][m]
        L.append(f"| {r['label']} | "
                 + " | ".join(f"{b[k]['mean_regret_vs_k20_db']:+.2f} (n={b[k]['n']})"
                              if b[k].get("n") else "—" for k in
                              ("rare_type", "low_sim", "hard", "common_type", "high_sim"))
                 + " |")
    L += ["", "## k-shot 预算曲线（mean regret dB）", "",
          "| 预算 m | 随机选点 | 主动选点 |", "|---|---|---|"]
    for mk in (0, 1, 3, 5, 8):
        key_r = "icdt_m0" if mk == 0 else f"icdt_rand_m{mk}"
        key_a = "icdt_m0" if mk == 0 else f"icdt_act_m{mk}"
        rnd = next((r for r in summary["methods"] if r["method"] == key_r), None)
        act = next((r for r in summary["methods"] if r["method"] == key_a), None)
        if rnd is None or act is None:
            continue
        L.append(f"| {mk} | "
                 f"{rnd['mean_regret_vs_k20_db']:+.2f} | {act['mean_regret_vs_k20_db']:+.2f} |")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    t0 = time.time()
    random.seed(HOLDOUT_SEED)
    np.random.seed(HOLDOUT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from signal_sinr import SignalSinrBackend
    backend = SignalSinrBackend(jsr_db=JSR_DB)
    hb.set_sinr_backend(backend.sinr)
    e07._patch_hb_paths()

    print(f"设备: {DEVICE}", flush=True)
    ctx = hb.load_holdout_context(n_holdout=N_HOLDOUT, holdout_seed=HOLDOUT_SEED)
    print(f"留出场景 {len(ctx.holdout_sids)}，检索库 {len(ctx.sid_bank)} 场景", flush=True)

    R_std_all, scen_full, r_mu, r_sd = load_globals()
    ens_models = e07._load_ensemble_models(DEVICE)
    icdt_models = load_icdt_models()
    print(f"QRetrievalNet ×{len(ens_models)}, ICDT ×{len(icdt_models)}", flush=True)

    results, per_scen = [], {}
    for m in METHOD_ORDER:
        t1 = time.time()
        print(f"--- {m}: {METHOD_LABELS[m]} ---", flush=True)
        if m.startswith("icdt_"):
            row = eval_icdt_method(ctx, m, icdt_models, R_std_all, scen_full, r_mu, r_sd)
        else:
            row = eval_baseline_method(ctx, m, ens_models, DEVICE)
        per_scen[m] = {"recommended_sinr_db": row.pop("per_scenario_recommended_sinr_db"),
                       "regret_vs_k20_db": row.pop("per_scenario_regret_vs_k20_db")}
        results.append(row)
        print(f"    regret {row['mean_regret_vs_k20_db']:+.2f} dB "
              f"({time.time()-t1:.1f}s)", flush=True)

    # ---- 分桶（与 20261007 同定义）----
    top1_list, ntype_list = [], []
    for sid in ctx.holdout_sids:
        jrow = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid].iloc[0]
        _, z_q = hb._encode_query(ctx, jrow)
        _, sims = hb._topk_similar_scenarios(z_q, ctx.z_bank, ctx.sid_bank, TOP_K,
                                             str(jrow["Jammer_Type"]),
                                             ctx.jammer_type_bank)
        top1_list.append(float(sims[0]) if len(sims) else np.nan)
        ntype_list.append(int((ctx.jammer_type_bank == str(jrow["Jammer_Type"])).sum()))
    top1_arr, ntype_arr = np.array(top1_list), np.array(ntype_list)
    med_sim, med_nt = np.nanmedian(top1_arr), np.median(ntype_arr)
    masks = {
        "rare_type": ntype_arr <= med_nt, "common_type": ntype_arr > med_nt,
        "low_sim": top1_arr < med_sim, "high_sim": top1_arr >= med_sim,
        "hard": (ntype_arr <= med_nt) & (top1_arr < med_sim),
    }
    print(f"分桶中位: top1_sim={med_sim:.3f}, 同类邻居={med_nt}", flush=True)

    q_ref = np.array(per_scen["q_only"]["recommended_sinr_db"], dtype=np.float64)
    n_s = len(q_ref)
    buckets = {m: {k: _bucket_stats(np.array(per_scen[m]["regret_vs_k20_db"]), mk)
                   for k, mk in masks.items()} for m in METHOD_ORDER}
    paired = {"reference": "q_only", "n_scenarios": n_s, "results": {
        m: _paired(q_ref, np.array(per_scen[m]["recommended_sinr_db"]),
                   np.ones(n_s, dtype=bool))
        for m in METHOD_ORDER if m != "q_only"}}

    summary = {
        "round": "20261013_icdt", "n_holdout_scenarios": n_s,
        "holdout_seed": HOLDOUT_SEED, "jsr_db": JSR_DB,
        "reference_regret": "oracle_k20 = max SINR among holdout K random A per scenario",
        "lambda_pessimistic": LAMBDA, "icdt_seeds": list(SEEDS),
        "methods": results, "buckets": buckets, "paired_vs_q_only": paired,
        "elapsed_min": (time.time() - t0) / 60,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    rows = [{k: v for k, v in r.items() if not k.startswith("per_scenario")}
            for r in results]
    import csv as _csv
    keys = sorted({k for r in rows for k in r})  # 各方法字段并集（ICDT 多真值查询数）
    with open(OUT_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_report(summary, ROUND / "comparison_report.md")
    print(f"\n完成，耗时 {summary['elapsed_min']:.1f} min | 输出: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
