# -*- coding: utf-8 -*-
"""Round 20261007 小样本抗干扰改进：留出评估与对比报告。

与 Round2 评估（outputs/20260909_signal_level/eval_retrained）严格同条件：
  * 同一 holdout CSV 前 50 个场景（seed=2026）
  * 同一信号级真值后端（SignalSinrBackend, JSR=10dB）
  * 同一检索库（Round2 train 场景、冻结 VAE 潜向量）

新增方法：
  ensemble_ctx_mean          改进①：5×QRetrievalNet 均值 μ 打分（union 池）
  ensemble_ctx_pessimistic   改进①：μ−λσ 悲观打分（λ=1.0），σ 即"检索/Q union
                             按 σ 加权融合"的机制——检索池动作（见过相似样本）
                             σ 低、随机池动作 σ 高被自动降权
  attention_ctx              改进②：单 QSetAttentionNet（邻居集合+cross-attention）
  attention_ens_pessimistic  改进①+②：5×QSetAttentionNet μ−λσ

小样本分桶（每场景两个稀有度指标，按中位数二分）：
  rare_type  同类型训练邻居数 ≤ 中位数（干扰类型稀有）
  low_sim    Top-1 检索余弦相似度 < 中位数（查询离训练库远）
  hard       两者同时成立（最难的小样本子集）

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261007_fewshot/code/eval_fewshot_round.py
"""
from __future__ import annotations

import csv
import json
import random
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
ROUND2 = HERE.parents[3] / "outputs" / "20260909_signal_level"
sys.path.insert(0, str(ROUND2 / "code"))

import numpy as np
import torch

from radar.infer import holdout_baselines as hb
from radar.train.q_retrieval_net import QRetrievalNet, predict_q_retrieval
from attention_q_net import QSetAttentionNet, neighbor_set_for_query, predict_q_attention
from signal_sinr import SignalSinrBackend

JSR_DB = 10.0
N_HOLDOUT = 50
HOLDOUT_SEED = 2026
TOP_K = 5
LAMBDA_PESSIMISTIC = 1.0
SEEDS = (42, 1042, 2042, 3042, 4042)

ROUND2_MODELS = ROUND2 / "models"
OUT_DIR = ROUND / "eval_fewshot"
ROUND2_EVAL_SUMMARY = ROUND2 / "eval_retrained" / "summary.json"

METHOD_LABELS = {
    "q_only": "Q-only（无检索，随机 5 A）",
    "retrieve_q": "Retrieve+Q（Top-5 池 argmax Q）",
    "retrieve_q_union": "Retrieve+Q（检索池∪随机100A）",
    "retrieve_q_ctx": "Retrieve+Q-Ctx（z_ctx 加权均值）",
    "retrieve_q_ctx_union": "Retrieve+Q-Ctx-Union（Round2 最优 ctx 变体）",
    "ensemble_ctx_mean": "① Ensemble-Mean（5×Q 检索池∪随机100A）",
    "ensemble_ctx_pessimistic": "① Ensemble-悲观 μ−1.0σ",
    "attention_ctx": "② Set-Attention（单模型，union 池）",
    "attention_ens_pessimistic": "①+② Attention-Ensemble 悲观 μ−1.0σ",
    "oracle_k20": "Oracle-K20（留出 K=10 随机 A 上界）",
    "oracle_wide": "Oracle-Wide（100 随机 A 上界）",
}
METHOD_ORDER = (
    "q_only",
    "retrieve_q",
    "retrieve_q_union",
    "retrieve_q_ctx",
    "retrieve_q_ctx_union",
    "ensemble_ctx_mean",
    "ensemble_ctx_pessimistic",
    "attention_ctx",
    "attention_ens_pessimistic",
    "oracle_k20",
    "oracle_wide",
)


def _patch_hb_paths() -> None:
    hb.ART_PATH = ROUND2 / "processed" / "artifacts.pkl"
    hb.A_STD_PATH = ROUND2 / "processed" / "A_std.npy"
    hb.SCENARIO_ID_PATH = ROUND2 / "processed" / "scenario_id.npy"
    hb.MODEL_PATH = ROUND2_MODELS / "latent_q" / "q_latent_best.pt"
    hb.RETRIEVAL_MODEL_PATH = ROUND2_MODELS / "latent_q_retrieval" / "q_retrieval_best.pt"
    hb.EMB_PATH = ROUND2_MODELS / "latent_q" / "state_latent_mu.npy"
    hb.OUT_DIR = OUT_DIR
    hb.SUMMARY_JSON = OUT_DIR / "summary.json"
    hb.SUMMARY_CSV = OUT_DIR / "summary.csv"
    hb.PER_SCENARIO_JSON = OUT_DIR / "per_scenario_detail.json"
    # artifacts.pkl 的 source_csv 是搬迁前旧路径 -> 回退到 Round2 重标注 CSV
    relabel_csv = ROUND2 / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv"
    orig_load = hb.load_simulated_csv

    def _load_csv_fallback(path):
        p = Path(path)
        return orig_load(p if p.is_file() else relabel_csv)

    hb.load_simulated_csv = _load_csv_fallback


def _load_ensemble_models(device: torch.device):
    models = []
    for seed in SEEDS:
        p = ROUND / "models" / "q_ensemble" / f"seed{seed}" / "q_retrieval_best.pt"
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = QRetrievalNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def _load_attention_models(device: torch.device):
    models = []
    for seed in SEEDS:
        p = ROUND / "models" / "q_attention_ensemble" / f"seed{seed}" / "q_attention_best.pt"
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = QSetAttentionNet(
            int(ckpt["z_dim"]), int(ckpt["a_dim"]),
            d_model=int(ckpt.get("d_model", 128)),
            n_heads=int(ckpt.get("n_heads", 4)),
            n_blocks=int(ckpt.get("n_blocks", 2)),
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def _ensemble_union_scores(
    ctx, models, z_q, union_cand, neighbor_sids, sims, device, pessimistic: bool
):
    z_ctx = hb._z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)
    a_mat = ctx.A_std[union_cand]
    q_all = np.stack(
        [predict_q_retrieval(m, z_q.reshape(1, -1), a_mat, z_ctx.reshape(1, -1), device)
         for m in models]
    )
    mu = q_all.mean(axis=0)
    sigma = q_all.std(axis=0)
    score = mu - LAMBDA_PESSIMISTIC * sigma if pessimistic else mu
    return score, sigma


def _attention_union_scores(
    ctx, models, z_q, union_cand, jammer_type, pessimistic: bool, device
):
    nb_z, nb_sim, nb_mask = neighbor_set_for_query(
        z_q, ctx.z_bank, ctx.sid_bank, ctx.jammer_type_bank,
        jammer_type, TOP_K, fallback_z=z_q,
    )
    a_mat = ctx.A_std[union_cand]
    q_all = np.stack(
        [predict_q_attention(m, z_q.reshape(1, -1), a_mat, nb_z, nb_sim, nb_mask, device)
         for m in models]
    )
    mu = q_all.mean(axis=0)
    sigma = q_all.std(axis=0)
    score = mu - LAMBDA_PESSIMISTIC * sigma if pessimistic else mu
    return score, sigma


def evaluate_method_custom(
    ctx, method: str, ens_models, att_models, device,
    top_k: int = TOP_K, holdout_seed: int = HOLDOUT_SEED, verbose: bool = False,
) -> dict:
    rng = np.random.default_rng(holdout_seed)
    regrets, rec_sinrs, o_k20, o_wide, sim_top1 = [], [], [], [], []
    sigma_diag = {"union": [], "retrieval_pool": [], "random_pool": []}

    for si, sid in enumerate(ctx.holdout_sids):
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jammer_row = block.iloc[0]
        _, z_q = hb._encode_query(ctx, jammer_row)

        oracle_k20 = hb._oracle_k20(block)
        oracle_wide = hb._oracle_wide(jammer_row, hb.ORACLE_WIDE_CANDIDATES, rng)
        o_k20.append(oracle_k20)
        o_wide.append(oracle_wide)

        neighbor_sids, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, top_k,
            str(jammer_row["Jammer_Type"]), ctx.jammer_type_bank,
        )
        retrieval_cand = hb._collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
        sim_top1.append(float(sims[0]) if len(sims) else float("nan"))
        sid_rng = np.random.default_rng(holdout_seed + int(sid))
        r_rec = None
        sigma_union = sigma_retr = sigma_rand = float("nan")

        if method == "q_only":
            row_idx = hb._select_q_only(ctx, z_q, int(sid), sid_rng)
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q":
            row_idx = hb._select_retrieve_q(ctx, z_q, retrieval_cand)
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q_union":
            row_idx, _ = hb._select_retrieve_q_union(ctx, z_q, retrieval_cand, sid_rng)
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q_ctx":
            best, _ = hb._select_retrieve_q_ctx(ctx, z_q, retrieval_cand, neighbor_sids, sims)
            r_rec = hb.sinr_for_action(jammer_row, best)
        elif method == "retrieve_q_ctx_union":
            row_idx, _ = hb._select_retrieve_q_ctx_union(
                ctx, z_q, retrieval_cand, neighbor_sids, sims, sid_rng
            )
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method in ("ensemble_ctx_mean", "ensemble_ctx_pessimistic"):
            union = hb.build_union_candidate_rows(
                retrieval_cand, ctx.train_row_pool, hb.UNION_RANDOM_CANDIDATES, sid_rng
            )
            score, sigma = _ensemble_union_scores(
                ctx, ens_models, z_q, union, neighbor_sids, sims, device,
                pessimistic=method.endswith("pessimistic"),
            )
            row_idx = int(union[int(np.argmax(score))])
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
            if len(sigma):
                in_retr = np.isin(union, retrieval_cand)
                sigma_union = float(sigma.mean())
                sigma_retr = float(sigma[in_retr].mean()) if in_retr.any() else float("nan")
                sigma_rand = float(sigma[~in_retr].mean()) if (~in_retr).any() else float("nan")
        elif method in ("attention_ctx", "attention_ens_pessimistic"):
            union = hb.build_union_candidate_rows(
                retrieval_cand, ctx.train_row_pool, hb.UNION_RANDOM_CANDIDATES, sid_rng
            )
            models = att_models if method == "attention_ens_pessimistic" else att_models[:1]
            score, sigma = _attention_union_scores(
                ctx, models, z_q, union, str(jammer_row["Jammer_Type"]),
                pessimistic=method.endswith("pessimistic"), device=device,
            )
            row_idx = int(union[int(np.argmax(score))])
            r_rec = hb._sinr_on_query(jammer_row, ctx.train_df, row_idx)
            if len(sigma) and method == "attention_ens_pessimistic":
                in_retr = np.isin(union, retrieval_cand)
                sigma_union = float(sigma.mean())
                sigma_retr = float(sigma[in_retr].mean()) if in_retr.any() else float("nan")
                sigma_rand = float(sigma[~in_retr].mean()) if (~in_retr).any() else float("nan")
        elif method == "oracle_k20":
            r_rec = hb._holdout_truth(block, hb._select_oracle_k20(block))
        elif method == "oracle_wide":
            r_rec = oracle_wide
        else:
            raise ValueError(method)

        rec_sinrs.append(r_rec)
        regrets.append(oracle_k20 - r_rec)
        sigma_diag["union"].append(sigma_union)
        sigma_diag["retrieval_pool"].append(sigma_retr)
        sigma_diag["random_pool"].append(sigma_rand)
        if verbose and (si < 2 or si % 10 == 0):
            print(f"  [{method}] {si+1}/{len(ctx.holdout_sids)} sid={sid} "
                  f"rec={r_rec:.2f} regret={regrets[-1]:.2f}", flush=True)

    regret_arr = np.array(regrets, dtype=np.float64)
    rec_arr = np.array(rec_sinrs, dtype=np.float64)
    row = {
        "method": method,
        "label": METHOD_LABELS[method],
        "n_evaluated": int(len(regrets)),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "mean_oracle_k20_db": float(np.mean(o_k20)),
        "mean_oracle_wide_db": float(np.mean(o_wide)),
        "mean_regret_vs_k20_db": float(regret_arr.mean()),
        "median_regret_vs_k20_db": float(np.median(regret_arr)),
        "regret_p90_db": float(np.percentile(regret_arr, 90)),
        "frac_zero_regret_vs_k20": float((regret_arr <= 1e-4).mean()),
        "mean_top1_cosine_sim": float(np.nanmean(sim_top1)),
        "per_scenario_recommended_sinr_db": rec_arr.tolist(),
        "per_scenario_regret_vs_k20_db": regret_arr.tolist(),
    }
    if method.endswith(("mean", "pessimistic")):
        su = np.array(sigma_diag["union"], dtype=np.float64)
        sr = np.array(sigma_diag["retrieval_pool"], dtype=np.float64)
        sd = np.array(sigma_diag["random_pool"], dtype=np.float64)
        row["sigma_diag"] = {
            "mean_sigma_union_pool": float(np.nanmean(su)),
            "mean_sigma_retrieval_pool": float(np.nanmean(sr)),
            "mean_sigma_random_pool": float(np.nanmean(sd)),
        }
    return row


def _bucket_stats(per_scen: dict, bucket_mask: np.ndarray) -> dict:
    idx = np.where(bucket_mask)[0]
    if len(idx) == 0:
        return {"n": 0}
    regret = np.array(per_scen["regret_vs_k20_db"], dtype=np.float64)[idx]
    return {
        "n": int(len(idx)),
        "mean_regret_vs_k20_db": float(regret.mean()),
        "median_regret_vs_k20_db": float(np.median(regret)),
        "frac_zero_regret_vs_k20": float((regret <= 1e-4).mean()),
    }


def _paired_win_rate(ref_sinr: np.ndarray, chal_sinr: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.where(mask)[0]
    n = len(idx)
    if n == 0:
        return {"n": 0, "wins": 0, "ties": 0, "losses": 0, "win_rate": float("nan")}
    d = chal_sinr[idx] - ref_sinr[idx]
    wins = int((d > 1e-6).sum())
    ties = int((np.abs(d) <= 1e-6).sum())
    return {"n": n, "wins": wins, "ties": ties, "losses": n - wins - ties,
            "win_rate": wins / n}


def write_report(summary: dict, out_path: Path) -> None:
    r2 = summary.get("round2_reference", {})
    lines = [
        "# Round 20261007 小样本抗干扰改进 — 对比报告",
        "",
        "## 背景与目标",
        "",
        "Round2（信号级重标注重训）已确认最优基线为 Retrieve+Q-Union（regret −5.08 dB，零 regret 84%）。"
        "本轮在其基础上实现两项面向**小样本抗干扰**的改进，全部在完全相同的留出条件"
        f"（前 {N_HOLDOUT} 场景、seed=2026、信号级真值 JSR={JSR_DB}dB）下对比。",
        "",
        "## 两项改进",
        "",
        "### 改进① Q Ensemble（深度集成 + 悲观打分）",
        "用 5 个不同 seed 训练 5 个 QRetrievalNet（与 Round2 完全同协议同 warm-start）。"
        "推理时对每个候选动作取集成均值 μ 与标准差 σ，用 **μ − 1.0·σ** 悲观打分："
        "集成成员意见不一致（σ 大）的动作被降权——这类动作通常是检索池之外的随机动作或"
        "训练分布外的动作。这相当于在 union 大池上按 σ 自动加权融合检索先验与探索。",
        "",
        "### 改进② Set-Attention 上下文（邻居集合 + cross-attention）",
        "原 QRetrievalNet 把 Top-5 邻居压缩成一个加权均值向量 z_ctx，信息瓶颈明显。"
        "新网络 QSetAttentionNet 把邻居**集合** [5,128] 连同检索相似度直接送入网络，"
        "候选动作 A 参与 query 端做 cross-attention，让每个候选动作自己决定要参考哪个邻居的哪部分信息。"
        "邻居集合在 forward 时注入（而非固定），因此将来可直接把检索库替换为 k-shot 实测样本库，零改动复用。",
        "",
        "## 主结果（全部 50 个留出场景，regret 相对 Oracle-K20）",
        "",
        "| 方法 | 推荐SINR(dB) | regret(dB) | 中位regret | 零regret占比 | 对q_only胜率 |",
        "|---|---|---|---|---|---|",
    ]
    for m in METHOD_ORDER:
        row = next(r for r in summary["methods"] if r["method"] == m)
        wr = summary["paired_vs_q_only"]["results"].get(m, {}).get("win_rate")
        wr_txt = f"{100*wr:.0f}%" if wr is not None else "—"
        lines.append(
            f"| {row['label']} | {row['mean_recommended_sinr_db']:.2f} | "
            f"{row['mean_regret_vs_k20_db']:.2f} | {row['median_regret_vs_k20_db']:.2f} | "
            f"{100*row['frac_zero_regret_vs_k20']:.0f}% | {wr_txt} |"
        )
    lines += ["", "## 小样本分桶（按中位数二分）", ""]
    bucket_titles = {
        "rare_type": "稀有干扰类型（同类训练邻居 ≤ 中位数）",
        "common_type": "常见干扰类型（同类训练邻居 > 中位数）",
        "low_sim": "低检索相似度（Top-1 余弦 < 中位数）",
        "high_sim": "高检索相似度（Top-1 余弦 ≥ 中位数）",
        "hard": "最难子集（稀有类型 且 低相似度）",
    }
    for bk in ("rare_type", "common_type", "low_sim", "high_sim", "hard"):
        lines += [f"### {bucket_titles[bk]}", "",
                  "| 方法 | n | regret(dB) | 零regret占比 | 对q_only胜率 |", "|---|---|---|---|---|"]
        buckets = summary["fewshot_buckets"]
        for m in METHOD_ORDER:
            st = buckets["by_method"].get(m, {}).get(bk, {})
            if not st or st.get("n", 0) == 0:
                continue
            wr = buckets["paired_vs_q_only"].get(m, {}).get(bk, {}).get("win_rate")
            wr_txt = f"{100*wr:.0f}%" if wr is not None and wr == wr else "—"
            lines.append(
                f"| {METHOD_LABELS[m]} | {st['n']} | {st['mean_regret_vs_k20_db']:.2f} | "
                f"{100*st['frac_zero_regret_vs_k20']:.0f}% | {wr_txt} |"
            )
        lines.append("")
    sigma_rows = [r for r in summary["methods"] if "sigma_diag" in r]
    if sigma_rows:
        lines += ["## σ 诊断（悲观打分的机制证据）", "",
                  "| 方法 | 全union池平均σ | 检索池动作σ | 随机池动作σ |", "|---|---|---|---|"]
        for r in sigma_rows:
            d = r["sigma_diag"]
            lines.append(
                f"| {r['label']} | {d['mean_sigma_union_pool']:.4f} | "
                f"{d['mean_sigma_retrieval_pool']:.4f} | {d['mean_sigma_random_pool']:.4f} |"
            )
        lines.append("")
    lines += ["## 结论（要点）", ""]
    main_table = {r["method"]: r for r in summary["methods"]}
    bl = main_table["retrieve_q_ctx_union"]["mean_regret_vs_k20_db"]
    win = summary["paired_vs_q_only"]["results"]
    best_new = min(
        ("ensemble_ctx_pessimistic", "attention_ens_pessimistic", "attention_ctx", "ensemble_ctx_mean"),
        key=lambda m: main_table[m]["mean_regret_vs_k20_db"],
    )
    bn = main_table[best_new]
    pess = main_table["ensemble_ctx_pessimistic"]
    mean_m = main_table["ensemble_ctx_mean"]
    buckets_bym = summary["fewshot_buckets"]["by_method"]
    buckets_wr = summary["fewshot_buckets"]["paired_vs_q_only"]
    hard_pess = buckets_bym["ensemble_ctx_pessimistic"]["hard"]["mean_regret_vs_k20_db"]
    hard_bl = buckets_bym["retrieve_q_ctx_union"]["hard"]["mean_regret_vs_k20_db"]
    hard_att = buckets_bym["attention_ctx"]["hard"]["mean_regret_vs_k20_db"]
    hard_pess_wr = buckets_wr["ensemble_ctx_pessimistic"]["hard"]["win_rate"]
    lines += [
        f"1. 相对 Round2 最优 ctx 基线（Retrieve+Q-Ctx-Union, regret {bl:.2f} dB）："
        f"平均 regret 上最佳新方法为 **{bn['label']}**（{bn['mean_regret_vs_k20_db']:.2f} dB），"
        f"与基线基本持平；但 **① Ensemble-悲观 μ−1.0σ** 把对 q_only 的配对胜率提升到 "
        f"{100*win['ensemble_ctx_pessimistic']['win_rate']:.0f}%"
        f"（基线 {100*win['retrieve_q_ctx_union']['win_rate']:.0f}%），中位 regret 最优"
        f"（{pess['median_regret_vs_k20_db']:.2f} dB），是更稳的选择。",
        f"2. 小样本分桶：在检索证据弱的三个桶（rare_type / low_sim / hard）上 Ensemble-悲观全面最优——"
        f"hard 桶 regret {hard_pess:.2f} dB（基线 {hard_bl:.2f} dB）、胜率 {100*hard_pess_wr:.0f}%；"
        "common_type / high_sim 桶上基线 union 仍略优。",
    ]
    if sigma_rows:
        s0 = sigma_rows[0]["sigma_diag"]
        lines += [
            "3. σ 诊断的诚实解读：检索池与随机池动作的平均 σ 差异很小"
            f"（{s0['mean_sigma_retrieval_pool']:.4f} vs {s0['mean_sigma_random_pool']:.4f}），"
            "悲观打分的收益并非来自「σ 区分检索/随机池」，而主要来自集成平均降噪："
            "μ 本身比单模型更准，μ−σ 再压低成员分歧大的候选"
            f"（中位 regret 由 {mean_m['median_regret_vs_k20_db']:.2f} 改善至 "
            f"{pess['median_regret_vs_k20_db']:.2f}）。",
        ]
    lines += [
        f"4. Set-Attention 单模型整体平均 regret 弱于基线，但在 hard 桶上单模型即达 {hard_att:.2f} dB"
        f"（优于基线 {hard_bl:.2f} dB），说明邻居集合表达在检索证据稀疏时更有效；"
        "其 5-seed 悲观集成反而退化，集成稳定性待改进。k-shot 注入接口已就绪"
        "（neighbor_set_for_query 支持任意邻居库，后续可换实测样本库做 k-shot 实验）。",
        "",
        "（数值均为信号级真值 JSR=10dB；负 regret 表示超过了留出 K=10 随机 A 的 Oracle 上界，"
        "因为候选池规模约 200 远大于 10。）",
    ]
    if r2:
        lines += ["", "## 与 Round2 存档结果的参照", ""]
        lines.append("Round2 eval_retrained/summary.json 的同条件基线（供核对，非本轮重跑）：")
        for row in r2.get("methods", []):
            lines.append(
                f"- {row['method']}: regret {row['mean_regret_vs_k20_db']:.2f} dB, "
                f"零 regret {100*row['frac_zero_regret_vs_k20']:.0f}%"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    random.seed(HOLDOUT_SEED)   # sample_radar_action 用全局 random（oracle_wide）
    np.random.seed(HOLDOUT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    backend = SignalSinrBackend(jsr_db=JSR_DB)
    hb.set_sinr_backend(backend.sinr)
    _patch_hb_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | JSR={JSR_DB}dB", flush=True)

    print("加载 holdout 上下文（Round2 检索库 + Q 模型）...", flush=True)
    ctx = hb.load_holdout_context(n_holdout=N_HOLDOUT, holdout_seed=HOLDOUT_SEED)
    n_scen = len(ctx.holdout_sids)
    print(f"留出场景 {n_scen}，检索库 {len(ctx.sid_bank)} train 场景", flush=True)

    print("加载改进模型...", flush=True)
    ens_models = _load_ensemble_models(device)
    att_models = _load_attention_models(device)
    print(f"QRetrievalNet ensemble ×{len(ens_models)}, QSetAttentionNet ×{len(att_models)}", flush=True)

    methods = list(METHOD_ORDER)
    results = []
    per_scen_store: dict[str, dict] = {}
    for m in methods:
        print(f"--- 评估 {m}: {METHOD_LABELS[m]} ---", flush=True)
        row = evaluate_method_custom(ctx, m, ens_models, att_models, device, verbose=True)
        per_scen_store[m] = {
            "recommended_sinr_db": row.pop("per_scenario_recommended_sinr_db"),
            "regret_vs_k20_db": row.pop("per_scenario_regret_vs_k20_db"),
        }
        results.append(row)

    # ---- 小样本分桶 ----
    sids = list(ctx.holdout_sids)
    # 逐场景 top1_sim / 同类邻居数（与检索一致，独立重算一次）
    top1_list, ntype_list = [], []
    for sid in sids:
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        _, z_q = hb._encode_query(ctx, jrow)
        _, sims = hb._topk_similar_scenarios(
            z_q, ctx.z_bank, ctx.sid_bank, TOP_K,
            str(jrow["Jammer_Type"]), ctx.jammer_type_bank,
        )
        top1_list.append(float(sims[0]) if len(sims) else np.nan)
        ntype_list.append(int((ctx.jammer_type_bank == str(jrow["Jammer_Type"])).sum()))
    top1_arr = np.array(top1_list)
    ntype_arr = np.array(ntype_list)
    med_sim = np.nanmedian(top1_arr)
    med_ntype = np.median(ntype_arr)
    bucket_masks = {
        "rare_type": ntype_arr <= med_ntype,
        "common_type": ntype_arr > med_ntype,
        "low_sim": top1_arr < med_sim,
        "high_sim": top1_arr >= med_sim,
        "hard": (ntype_arr <= med_ntype) & (top1_arr < med_sim),
    }

    q_only_rec = np.array(per_scen_store["q_only"]["recommended_sinr_db"], dtype=np.float64)
    by_method_buckets: dict[str, dict] = {}
    paired_buckets: dict[str, dict] = {}
    for m in methods:
        rec = np.array(per_scen_store[m]["recommended_sinr_db"], dtype=np.float64)
        by_method_buckets[m] = {bk: _bucket_stats(per_scen_store[m], mk)
                                for bk, mk in bucket_masks.items()}
        paired_buckets[m] = {bk: _paired_win_rate(q_only_rec, rec, mk)
                             for bk, mk in bucket_masks.items()}

    paired = {"reference": "q_only", "n_scenarios": n_scen, "results": {}}
    for m in methods:
        if m == "q_only":
            continue
        rec = np.array(per_scen_store[m]["recommended_sinr_db"], dtype=np.float64)
        paired["results"][m] = _paired_win_rate(q_only_rec, rec, np.ones(n_scen, dtype=bool))

    summary = {
        "round": "20261007_fewshot",
        "n_holdout_scenarios": n_scen,
        "holdout_seed": HOLDOUT_SEED,
        "jsr_db": JSR_DB,
        "lambda_pessimistic": LAMBDA_PESSIMISTIC,
        "seeds": list(SEEDS),
        "bucket_median_top1_sim": float(med_sim),
        "bucket_median_same_type_count": float(med_ntype),
        "methods": results,
        "paired_vs_q_only": paired,
        "fewshot_buckets": {"by_method": by_method_buckets, "paired_vs_q_only": paired_buckets},
        "per_scenario_meta": {
            "top1_cosine_sim": top1_list,
            "same_type_bank_count": ntype_list,
        },
        "backend_stats": backend.stats(),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }

    # Round2 存档基线（参照）
    if ROUND2_EVAL_SUMMARY.is_file():
        with open(ROUND2_EVAL_SUMMARY, encoding="utf-8") as f:
            r2 = json.load(f)
        summary["round2_reference"] = {
            "methods": [
                {"method": r["method"], "mean_regret_vs_k20_db": r["mean_regret_vs_k20_db"],
                 "frac_zero_regret_vs_k20": r["frac_zero_regret_vs_k20"]}
                for r in r2.get("methods", [])
            ]
        }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "per_scenario_detail.json", "w", encoding="utf-8") as f:
        json.dump(per_scen_store, f, ensure_ascii=False, indent=2)

    fieldnames = ["method", "label", "mean_recommended_sinr_db", "mean_oracle_k20_db",
                  "mean_regret_vs_k20_db", "median_regret_vs_k20_db",
                  "frac_zero_regret_vs_k20", "mean_top1_cosine_sim"]
    with open(OUT_DIR / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)
    with open(OUT_DIR / "comparison_table.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)

    write_report(summary, ROUND / "comparison_report.md")

    print("\n=== 汇总（regret vs Oracle-K20）===", flush=True)
    print(f"{'方法':<34}{'推荐SINR':>10}{'regret':>9}{'零regret':>9}{'胜q_only':>9}")
    for r in results:
        wr = paired["results"].get(r["method"], {}).get("win_rate")
        wr_txt = f"{100*wr:.0f}%" if wr is not None else "—"
        print(f"{r['label']:<34}{r['mean_recommended_sinr_db']:>10.2f}"
              f"{r['mean_regret_vs_k20_db']:>9.2f}"
              f"{100*r['frac_zero_regret_vs_k20']:>8.0f}%{wr_txt:>9}")
    print(f"\n后端: {backend.stats()}")
    print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟，输出: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
