# -*- coding: utf-8 -*-
"""500 场景核心评估。复用冻结模型，只扩展真正独立的评估场景。"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
OUT = HERE.parents[1] / "results" / "core_500"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs" / "20260909_signal_level" / "code"))

SRC = ROOT / "outputs" / "20261021_dense_oracle" / "code" / "eval_fewshot_current.py"
spec = importlib.util.spec_from_file_location("eval_fs_ext", SRC)
fs = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(fs)

from signal_sinr import SignalSinrBackend  # noqa: E402

N = 500
SEED = 2026
METHODS = (
    "q_only",
    "retrieve_q",
    "retrieve_q_union",
    "ensemble_ctx_mean",
    "ensemble_ctx_pessimistic",
)


def stats(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
        "std": float(x.std(ddof=1)), "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "cvar10": float(np.sort(x)[-max(1, int(np.ceil(.1 * len(x)))):].mean()),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    fs.OUT_DIR = OUT
    fs._patch_hb_paths()
    backend = SignalSinrBackend(jsr_db=10.0)
    fs.hb.set_sinr_backend(backend.sinr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; loading 500-scenario context", flush=True)
    ctx = fs.hb.load_holdout_context(n_holdout=N, holdout_seed=SEED)
    assert len(ctx.holdout_sids) == N, len(ctx.holdout_sids)
    ens = fs._load_ensemble_models(device)

    detail = {"scenario_id": [int(x) for x in ctx.holdout_sids]}
    meta = ctx.holdout_df.drop_duplicates("scenario_id").set_index("scenario_id")
    detail["jammer_type"] = [str(meta.loc[s, "Jammer_Type"]) for s in ctx.holdout_sids]
    rows = []
    for method in METHODS:
        print(f"[{method}] start", flush=True)
        row = fs.evaluate_method_custom(ctx, method, ens, [], device, verbose=True)
        rec = np.asarray(row.pop("per_scenario_recommended_sinr_db"), dtype=float)
        reg = np.asarray(row.pop("per_scenario_regret_vs_k20_db"), dtype=float)
        detail[f"{method}__sinr"] = rec.tolist()
        detail[f"{method}__regret_k20"] = reg.tolist()
        row["first50"] = stats(reg[:50])
        row["fresh450"] = stats(reg[50:])
        row["all500"] = stats(reg)
        rows.append(row)
        print(f"[{method}] fresh450 mean regret={reg[50:].mean():.3f}", flush=True)

    df = pd.DataFrame(detail)
    df.to_csv(OUT / "per_scenario.csv", index=False, encoding="utf-8-sig")
    by_type = []
    for method in METHODS:
        for jt, g in df.groupby("jammer_type"):
            x = g[f"{method}__regret_k20"].to_numpy(float)
            by_type.append({"method": method, "jammer_type": jt, **stats(x)})
    pd.DataFrame(by_type).to_csv(OUT / "by_jammer_type.csv", index=False, encoding="utf-8-sig")
    summary = {
        "protocol": {"n": N, "seed": SEED, "jsr_db": 10.0,
                     "first50_status": "historically reused", "fresh450_status": "not in prior reports"},
        "methods": rows, "backend": backend.stats(),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({r["method"]: r["fresh450"] for r in rows}, ensure_ascii=False, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
