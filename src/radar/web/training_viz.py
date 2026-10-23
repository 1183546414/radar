# -*- coding: utf-8 -*-
"""演示页：训练曲线与潜空间可视化资源。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from radar.paths import OUT_FIGURES, OUT_Q_VALUE, OUT_STATE_VAE

VAE_METRICS = OUT_STATE_VAE / "metrics.json"
Q_LATENT_METRICS = OUT_Q_VALUE / "latent_q" / "metrics.json"
Q_RETRIEVAL_METRICS = OUT_Q_VALUE / "latent_q_retrieval" / "metrics.json"

FIGURE_SPECS: List[Dict[str, str]] = [
    {
        "id": "vae_rmse",
        "title": "State AE 验证重构误差",
        "path": "eval/vae_val_rmse.png",
        "caption": "VAE/AE 训练过程中验证集重构 RMSE 曲线",
    },
    {
        "id": "tsne_scatter",
        "title": "潜向量 t-SNE（按干扰类型）",
        "path": "latent/tsne_scatter.png",
        "caption": "去重场景 μ 的二维 t-SNE，颜色为 Jammer_Type",
    },
    {
        "id": "tsne_density",
        "title": "潜向量 t-SNE 密度",
        "path": "latent/tsne_density.png",
        "caption": "潜空间分布密度（hexbin）",
    },
    {
        "id": "dashboard",
        "title": "训练与留出集总览",
        "path": "eval/training_dashboard.png",
        "caption": "VAE / Q 曲线与留出 baseline 摘要",
    },
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vae_summary(data: Optional[dict]) -> Dict[str, Any]:
    if not data:
        return {}
    s = data.get("summary", {})
    history = data.get("history", [])
    best_rmse = None
    if history:
        best_rmse = min(
            h.get("val_rec_rmse", (h.get("val_rec_mse", 0)) ** 0.5)
            for h in history
        )
    return {
        "model_type": s.get("model_type"),
        "latent_dim": s.get("latent_dim"),
        "epochs_run": s.get("epochs_run"),
        "best_val_rmse": round(float(best_rmse), 6) if best_rmse is not None else None,
    }


def _q_summary(data: Optional[dict]) -> Dict[str, Any]:
    if not data:
        return {}
    k20 = data.get("per_scenario_k20_selection", {})
    return {
        "epochs_run": data.get("epochs_run"),
        "val_pearson_std": data.get("val_pearson_std"),
        "val_rmse_std": data.get("val_rmse_std"),
        "oracle_hit_at_1_k20": k20.get("oracle_hit_at_1"),
        "mean_regret_db_k20": k20.get("mean_regret_sinr_db"),
    }


def get_training_viz_meta() -> Dict[str, Any]:
    """返回演示页可展示的训练图列表与关键指标。"""
    figures: List[Dict[str, Any]] = []
    for spec in FIGURE_SPECS:
        rel = spec["path"]
        full = OUT_FIGURES / rel
        if not full.is_file():
            continue
        figures.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "caption": spec["caption"],
                "url": f"/figures/{rel}",
                "mtime": int(full.stat().st_mtime),
            }
        )

    return {
        "figures": figures,
        "metrics": {
            "vae": _vae_summary(_load_json(VAE_METRICS)),
            "q_latent": _q_summary(_load_json(Q_LATENT_METRICS)),
            "q_retrieval": _q_summary(_load_json(Q_RETRIEVAL_METRICS)),
        },
        "figures_root": str(OUT_FIGURES),
    }
