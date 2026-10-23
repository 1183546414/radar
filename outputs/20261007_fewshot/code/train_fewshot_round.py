# -*- coding: utf-8 -*-
"""Round 20261007 小样本抗干扰改进：训练脚本。

改进① Q Ensemble：用 5 个不同 seed 训练 5 个 QRetrievalNet（与 Round2 完全
同协议：同数据/split/warm-start/超参），推理时用 μ−λσ 悲观打分。
改进② set-attention：新网络 QSetAttentionNet，把同类型 Top-K 邻居集合
[K,128]+相似度直接送入 cross-attention（候选动作参与 query），同样训 5 seeds。

数据与 split 全部复用 Round2（outputs/20260909_signal_level/）：
  * processed/（信号级 R 重标注）
  * models/latent_q/state_latent_mu.npy（冻结 VAE 潜向量）
  * models/latent_q_retrieval/z_ctx_per_row_same_type_weighted.npy（ensemble 用）
  * models/latent_q/q_latent_best.pt（warm-start）
原模块常量不动，仅运行期 patch（与 Round2 train_signal_round.py 同套路）。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261007_fewshot/code/train_fewshot_round.py
"""
from __future__ import annotations

import gc
import json
import math
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
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ---- Round2 路径与运行期 patch（在 import 具体训练模块前先定位路径）----
ROUND2 = HERE.parents[3] / "outputs" / "20260909_signal_level"
ROUND2_PROCESSED = ROUND2 / "processed"
ROUND2_MODELS = ROUND2 / "models"
ROUND2_CODE = ROUND2 / "code"
sys.path.insert(0, str(ROUND2_CODE))  # 便于 import signal_sinr

from radar.train import retrieval_ranking as rr  # noqa: E402

rr.DATA_V2_PROCESSED = ROUND2_PROCESSED  # 原始 artifacts source_csv 已失效（搬迁）
from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
_BACKEND = SignalSinrBackend(jsr_db=JSR_DB)
rr.sinr_from_csv_row = _BACKEND.sinr  # ranking oracle 与重标注标签同源

from radar.train.q_retrieval_net import (  # noqa: E402
    QRetrievalNet,
    warm_start_from_q_latent,
)
from radar.train.retrieval_ranking import (  # noqa: E402
    RetrievalRankingContext,
    listwise_ranking_loss,
)
from radar.train.train_q_latent import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATIENCE,
    RANK_LAMBDA,
    RANK_SCENARIOS_PER_STEP,
    RANK_STEPS_PER_EPOCH,
    SEED,
    TOP_K_RETRIEVE,
    VAL_RANK_MAX_SCENARIOS,
    VAL_RATIO,
    WEIGHT_DECAY,
    _split_by_scenario,
)
from radar.train.train_q_retrieval import (  # noqa: E402
    _ranking_loss_batch_retrieval,
    eval_retrieval_pool_selection_ctx,
)
from attention_q_net import (  # noqa: E402
    QSetAttentionNet,
    neighbor_set_for_query,
    predict_q_attention,
)

OUT_DIR = ROUND / "models"
Q_LATENT_CKPT = ROUND2_MODELS / "latent_q" / "q_latent_best.pt"
Z_CTX_CACHE = ROUND2_MODELS / "latent_q_retrieval" / "z_ctx_per_row_same_type_weighted.npy"
REPORT_PATH = ROUND / "train_report.json"

SEEDS = (42, 1042, 2042, 3042, 4042)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_round2_data():
    A_std = np.load(ROUND2_PROCESSED / "A_std.npy").astype(np.float32)
    R_std = np.load(ROUND2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)
    scenario_ids = np.load(ROUND2_PROCESSED / "scenario_id.npy")
    z = np.load(ROUND2_MODELS / "latent_q" / "state_latent_mu.npy").astype(np.float32)
    z_ctx = np.load(Z_CTX_CACHE)
    assert z_ctx.shape == z.shape, f"z_ctx 缓存形状不符: {z_ctx.shape} vs {z.shape}"
    tr_idx, val_idx, tr_scenarios, val_scenarios = _split_by_scenario(
        scenario_ids, VAL_RATIO, SEED
    )
    return A_std, R_std, scenario_ids, z, z_ctx, tr_idx, val_idx, tr_scenarios, val_scenarios


def _build_per_sid_neighbor_sets(
    z: np.ndarray,
    scenario_ids: np.ndarray,
    rank_ctx: RetrievalRankingContext,
    tr_sid_set: set[int],
    top_k: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """每场景的定长邻居集合 (nb_z[K,dz], nb_sim[K], mask[K])；train 场景排除自身。"""
    per_sid: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid in np.unique(scenario_ids):
        sid = int(sid)
        rows = rank_ctx.sid_to_rows[sid]
        z_q = z[int(rows[0])]
        qtype = str(rank_ctx.jammer_by_sid[sid]["Jammer_Type"])
        exclude = sid if sid in tr_sid_set else None
        per_sid[sid] = neighbor_set_for_query(
            z_q,
            rank_ctx.z_bank,
            rank_ctx.sid_bank,
            rank_ctx.jammer_type_bank,
            qtype,
            top_k,
            exclude_sid=exclude,
            fallback_z=z_q,
        )
    return per_sid


def _gather_neighbors(
    row_indices: np.ndarray,
    scenario_ids: np.ndarray,
    per_sid: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按行子集从 per_sid 收集邻居集合（避免整表 [N,K,dz] 大数组）。"""
    nb_z = nb_sim = nb_mask = None
    chunks = []
    sims, masks = [], []
    for i in row_indices:
        nz, ns, mk = per_sid[int(scenario_ids[i])]
        chunks.append(nz)
        sims.append(ns)
        masks.append(mk)
    nb_z = np.stack(chunks, axis=0).astype(np.float32)
    nb_sim = np.stack(sims, axis=0).astype(np.float32)
    nb_mask = np.stack(masks, axis=0)
    return nb_z, nb_sim, nb_mask


class _AttentionDataset(torch.utils.data.Dataset):
    """逐行取 (z, a, r, 邻居集合)，全部预转 torch 张量（零拷贝视图，省内存且免逐项转换）。"""

    def __init__(self, row_idx, z, a_std, r_std, scenario_ids, per_sid):
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.z_t = torch.from_numpy(z)
        self.a_t = torch.from_numpy(a_std)
        self.r_t = torch.from_numpy(r_std)
        self.sids = scenario_ids
        self.per_sid_t = {
            sid: (torch.from_numpy(nz), torch.from_numpy(ns), torch.from_numpy(mk))
            for sid, (nz, ns, mk) in per_sid.items()
        }

    def __len__(self) -> int:
        return len(self.row_idx)

    def __getitem__(self, i: int):
        j = int(self.row_idx[i])
        nz, ns, mk = self.per_sid_t[int(self.sids[j])]
        return self.z_t[j], self.a_t[j], self.r_t[j], nz, ns, mk


@torch.no_grad()
def _val_mse_attention(model, val_idx, data, per_sid, device, chunk: int = 4096) -> float:
    z, A_std, R_std, scenario_ids = (
        data["z"], data["A_std"], data["R_std"], data["scenario_ids"]
    )
    model.eval()
    se, n = 0.0, 0
    for s in range(0, len(val_idx), chunk):
        rows = np.asarray(val_idx[s : s + chunk])
        nb_z, nb_sim, nb_mask = _gather_neighbors(rows, scenario_ids, per_sid)
        zq = torch.from_numpy(z[rows]).to(device)
        ab = torch.from_numpy(A_std[rows]).to(device)
        rb = torch.from_numpy(R_std[rows]).to(device)
        zb = torch.from_numpy(nb_z).to(device)
        sb = torch.from_numpy(nb_sim).to(device)
        mb = torch.from_numpy(nb_mask).to(device)
        pred = model(zq, ab, zb, sb, mb).squeeze(-1)
        se += float(((pred - rb) ** 2).sum().item())
        n += len(rows)
    return se / max(n, 1)


def _eval_pool_attention(
    model: QSetAttentionNet,
    rank_ctx: RetrievalRankingContext,
    per_sid: dict,
    eval_scenarios: np.ndarray,
    device: torch.device,
    top_k: int,
) -> dict:
    """检索候选池 argmax QSetAttentionNet vs oracle（对齐 Round2 检索池评估）。"""
    hits, regrets = [], []
    for sid in eval_scenarios:
        cand_rows = rank_ctx.candidate_rows_for_query(int(sid), top_k, exclude_self=False)
        if len(cand_rows) == 0:
            continue
        z_q = rank_ctx.z[int(rank_ctx.sid_to_rows[int(sid)][0])]
        r_true = rank_ctx.true_sinr_on_query(int(sid), cand_rows)
        oracle_idx = int(np.argmax(r_true))
        nb_z, nb_sim, nb_mask = per_sid[int(sid)]
        q_scores = predict_q_attention(
            model, z_q.reshape(1, -1), rank_ctx.a_std[cand_rows], nb_z, nb_sim, nb_mask, device
        )
        pred_idx = int(np.argmax(q_scores))
        hits.append(pred_idx == oracle_idx)
        regrets.append(float(r_true[oracle_idx] - r_true[pred_idx]))
    if not hits:
        return {"n_eval_scenarios": 0, "oracle_hit_at_1": float("nan"),
                "mean_regret_sinr_db": float("nan")}
    regret_arr = np.array(regrets, dtype=np.float64)
    return {
        "n_eval_scenarios": int(len(hits)),
        "oracle_hit_at_1": float(np.mean(hits)),
        "mean_regret_sinr_db": float(regret_arr.mean()),
        "frac_zero_regret": float((regret_arr <= 1e-6).mean()),
    }


def _ranking_loss_batch_attention(
    model: QSetAttentionNet,
    rank_ctx: RetrievalRankingContext,
    per_sid: dict,
    query_sids: np.ndarray,
    device: torch.device,
    top_k: int,
) -> torch.Tensor:
    losses = []
    for sid in query_sids:
        cand_rows = rank_ctx.candidate_rows_for_query(int(sid), top_k, exclude_self=True)
        if len(cand_rows) == 0:
            continue
        z_q = rank_ctx.z[int(rank_ctx.sid_to_rows[int(sid)][0])]
        r_true = rank_ctx.true_sinr_on_query(int(sid), cand_rows)
        oracle_idx = int(np.argmax(r_true))
        n = len(cand_rows)
        nb_z, nb_sim, nb_mask = per_sid[int(sid)]
        zq_t = torch.from_numpy(z_q).to(device).unsqueeze(0).expand(n, -1)
        a_t = torch.from_numpy(rank_ctx.a_std[cand_rows]).to(device)
        zb = torch.from_numpy(nb_z).to(device).unsqueeze(0).expand(n, -1, -1)
        sb = torch.from_numpy(nb_sim).to(device).unsqueeze(0).expand(n, -1)
        mb = torch.from_numpy(nb_mask).to(device).unsqueeze(0).expand(n, -1)
        q_scores = model(zq_t, a_t, zb, sb, mb).squeeze(-1)
        losses.append(listwise_ranking_loss(q_scores, oracle_idx))
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def train_one_retrieval(seed: int, data: dict, rank_ctx: RetrievalRankingContext,
                        val_rank_scenarios: np.ndarray) -> dict:
    """改进①成员：与 Round2 train_q_retrieval_main 同协议，仅换 seed。"""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rank_rng = np.random.default_rng(seed)

    z, z_ctx, A_std, R_std = data["z"], data["z_ctx"], data["A_std"], data["R_std"]
    tr_idx, val_idx = data["tr_idx"], data["val_idx"]
    tr_scenarios = data["tr_scenarios"]
    z_dim, a_dim = z.shape[1], A_std.shape[1]

    out_dir = OUT_DIR / "q_ensemble" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "q_retrieval_best.pt"

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z[tr_idx]),
            torch.from_numpy(z_ctx[tr_idx]),
            torch.from_numpy(A_std[tr_idx]),
            torch.from_numpy(R_std[tr_idx]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = QRetrievalNet(z_dim, a_dim).to(DEVICE)
    warm_start_from_q_latent(model, Q_LATENT_CKPT, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    z_val_t = torch.from_numpy(z[val_idx]).to(DEVICE)
    z_ctx_val_t = torch.from_numpy(z_ctx[val_idx]).to(DEVICE)
    a_val_t = torch.from_numpy(A_std[val_idx]).to(DEVICE)
    r_val_t = torch.from_numpy(R_std[val_idx]).to(DEVICE)

    best_hit1, wait, epochs_run = -1.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for zq, zc, ab, rb in train_loader:
            zq, zc, ab, rb = zq.to(DEVICE), zc.to(DEVICE), ab.to(DEVICE), rb.to(DEVICE)
            pred = model(zq, ab, zc).squeeze(-1)
            loss = loss_fn(pred, rb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for _ in range(RANK_STEPS_PER_EPOCH):
            batch_sids = rank_rng.choice(
                tr_scenarios, size=min(RANK_SCENARIOS_PER_STEP, len(tr_scenarios)), replace=False
            )
            loss_rank = _ranking_loss_batch_retrieval(model, rank_ctx, batch_sids, DEVICE, TOP_K_RETRIEVE)
            optimizer.zero_grad()
            (RANK_LAMBDA * loss_rank).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(z_val_t, a_val_t, z_ctx_val_t).squeeze(-1), r_val_t).item()
        val_ret = eval_retrieval_pool_selection_ctx(
            model, rank_ctx, val_rank_scenarios, DEVICE, TOP_K_RETRIEVE, exclude_self=False
        )
        val_hit1 = float(val_ret["oracle_hit_at_1"])
        epochs_run = epoch
        if epoch == 1 or epoch % 20 == 0:
            print(f"  [retrieval seed={seed}] epoch {epoch:03d} val_mse={val_mse:.5f} hit@1={val_hit1:.3f}", flush=True)
        if val_hit1 > best_hit1 + 1e-6:
            best_hit1, wait = val_hit1, 0
            torch.save({"model": model.state_dict(), "z_dim": z_dim, "a_dim": a_dim,
                        "top_k_retrieve": TOP_K_RETRIEVE, "seed": seed, "warm_start": True}, model_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [retrieval seed={seed}] 早停 @ {epoch} hit@1={val_hit1:.3f}", flush=True)
                break

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        y_pred = model(z_val_t, a_val_t, z_ctx_val_t).squeeze(-1).cpu().numpy()
    r_val = R_std[val_idx]
    mse = float(np.mean((r_val - y_pred) ** 2))
    metrics = {
        "seed": seed, "epochs_run": epochs_run, "best_val_hit_at_1": best_hit1,
        "val_rmse_std": float(math.sqrt(mse)), "val_r2_std": float(r2_score(r_val, y_pred)),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  [retrieval seed={seed}] 完成 {metrics}", flush=True)
    return metrics


def train_one_attention(seed: int, data: dict, rank_ctx: RetrievalRankingContext,
                        per_sid: dict, val_rank_scenarios: np.ndarray) -> dict:
    """改进②成员：QSetAttentionNet（从头训练，其余协议对齐）。

    内存友好：邻居集合经 _AttentionDataset 按需从 per_sid 收集，
    不再构造整表 [N,K,dz] 大数组。
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rank_rng = np.random.default_rng(seed)

    z, A_std, R_std = data["z"], data["A_std"], data["R_std"]
    scenario_ids, tr_idx, val_idx = data["scenario_ids"], data["tr_idx"], data["val_idx"]
    tr_scenarios = data["tr_scenarios"]
    z_dim, a_dim = z.shape[1], A_std.shape[1]

    out_dir = OUT_DIR / "q_attention_ensemble" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "q_attention_best.pt"

    train_loader = DataLoader(
        _AttentionDataset(tr_idx, z, A_std, R_std, scenario_ids, per_sid),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = QSetAttentionNet(z_dim, a_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_hit1, wait, epochs_run = -1.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for zq, ab, rb, zb, sb, mb in train_loader:
            zq, ab, rb = zq.to(DEVICE), ab.to(DEVICE), rb.to(DEVICE)
            zb, sb, mb = zb.to(DEVICE), sb.to(DEVICE), mb.to(DEVICE)
            pred = model(zq, ab, zb, sb, mb).squeeze(-1)
            loss = loss_fn(pred, rb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for _ in range(RANK_STEPS_PER_EPOCH):
            batch_sids = rank_rng.choice(
                tr_scenarios, size=min(RANK_SCENARIOS_PER_STEP, len(tr_scenarios)), replace=False
            )
            loss_rank = _ranking_loss_batch_attention(model, rank_ctx, per_sid, batch_sids, DEVICE, TOP_K_RETRIEVE)
            optimizer.zero_grad()
            (RANK_LAMBDA * loss_rank).backward()
            optimizer.step()
        val_mse = _val_mse_attention(model, val_idx, data, per_sid, DEVICE)
        val_ret = _eval_pool_attention(model, rank_ctx, per_sid, val_rank_scenarios, DEVICE, TOP_K_RETRIEVE)
        val_hit1 = float(val_ret["oracle_hit_at_1"])
        epochs_run = epoch
        if epoch == 1 or epoch % 20 == 0:
            print(f"  [attention seed={seed}] epoch {epoch:03d} val_mse={val_mse:.5f} hit@1={val_hit1:.3f}", flush=True)
        if val_hit1 > best_hit1 + 1e-6:
            best_hit1, wait = val_hit1, 0
            torch.save({"model": model.state_dict(), "z_dim": z_dim, "a_dim": a_dim,
                        "top_k_retrieve": TOP_K_RETRIEVE, "seed": seed,
                        "d_model": model.d_model, "n_heads": model.n_heads,
                        "n_blocks": model.n_blocks, "warm_start": False}, model_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [attention seed={seed}] 早停 @ {epoch} hit@1={val_hit1:.3f}", flush=True)
                break

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(val_idx), 4096):
            rows = np.asarray(val_idx[s : s + 4096])
            nb_z, nb_sim, nb_mask = _gather_neighbors(rows, scenario_ids, per_sid)
            zq = torch.from_numpy(z[rows]).to(DEVICE)
            ab = torch.from_numpy(A_std[rows]).to(DEVICE)
            zb = torch.from_numpy(nb_z).to(DEVICE)
            sb = torch.from_numpy(nb_sim).to(DEVICE)
            mb = torch.from_numpy(nb_mask).to(DEVICE)
            preds.append(model(zq, ab, zb, sb, mb).squeeze(-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    r_val = R_std[val_idx]
    mse = float(np.mean((r_val - y_pred) ** 2))
    metrics = {
        "seed": seed, "epochs_run": epochs_run, "best_val_hit_at_1": best_hit1,
        "val_rmse_std": float(math.sqrt(mse)), "val_r2_std": float(r2_score(r_val, y_pred)),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  [attention seed={seed}] 完成 {metrics}", flush=True)
    del model, optimizer, ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"设备: {DEVICE} | JSR={JSR_DB}dB 信号级真值后端", flush=True)

    print("加载 Round2 数据...", flush=True)
    A_std, R_std, scenario_ids, z, z_ctx, tr_idx, val_idx, tr_scenarios, val_scenarios = _load_round2_data()
    data = {"z": z, "z_ctx": z_ctx, "A_std": A_std, "R_std": R_std,
            "scenario_ids": scenario_ids, "tr_idx": tr_idx, "val_idx": val_idx,
            "tr_scenarios": tr_scenarios}
    print(f"数据: z{z.shape}, A{A_std.shape}, 场景 {len(np.unique(scenario_ids))}，"
          f"train {len(tr_scenarios)} / val {len(val_scenarios)} 场景", flush=True)

    print("构建检索 ranking 上下文（train 库）...", flush=True)
    rank_ctx = RetrievalRankingContext.build(
        z=z, scenario_ids=scenario_ids, a_std=A_std, train_scenarios=tr_scenarios,
        csv_path=ROUND2 / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv",
    )
    val_rank_scenarios = val_scenarios[: min(VAL_RANK_MAX_SCENARIOS, len(val_scenarios))]

    report: dict = {"jsr_db": JSR_DB, "seeds": list(SEEDS),
                    "backend_stats": _BACKEND.stats(), "stages": {}}

    def _member_done(sub: str, seed: int, fname: str) -> bool:
        d = OUT_DIR / sub / f"seed{seed}"
        return (d / fname).is_file() and (d / "metrics.json").is_file()

    print(f"\n=== 改进①：Q Ensemble（QRetrievalNet × {len(SEEDS)} seeds）===", flush=True)
    ens_metrics = []
    for seed in SEEDS:
        if _member_done("q_ensemble", seed, "q_retrieval_best.pt"):
            m = json.load(open(OUT_DIR / "q_ensemble" / f"seed{seed}" / "metrics.json", encoding="utf-8"))
            print(f"  [retrieval seed={seed}] 已完成，跳过 {m}", flush=True)
            ens_metrics.append(m)
            continue
        ens_metrics.append(train_one_retrieval(seed, data, rank_ctx, val_rank_scenarios))
        gc.collect()
    report["stages"]["q_ensemble"] = {"members": ens_metrics}

    print(f"\n=== 改进②：QSetAttentionNet × {len(SEEDS)} seeds ===", flush=True)
    tr_sid_set = {int(s) for s in tr_scenarios}
    per_sid = _build_per_sid_neighbor_sets(z, scenario_ids, rank_ctx, tr_sid_set, TOP_K_RETRIEVE)
    att_metrics = []
    for seed in SEEDS:
        if _member_done("q_attention_ensemble", seed, "q_attention_best.pt"):
            m = json.load(open(OUT_DIR / "q_attention_ensemble" / f"seed{seed}" / "metrics.json", encoding="utf-8"))
            print(f"  [attention seed={seed}] 已完成，跳过 {m}", flush=True)
            att_metrics.append(m)
            continue
        att_metrics.append(train_one_attention(seed, data, rank_ctx, per_sid, val_rank_scenarios))
        gc.collect()
    report["stages"]["q_attention_ensemble"] = {"members": att_metrics}

    report["total_min"] = round((time.time() - t0) / 60, 2)
    report["backend_stats_final"] = _BACKEND.stats()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，总耗时 {report['total_min']} 分钟，报告: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
