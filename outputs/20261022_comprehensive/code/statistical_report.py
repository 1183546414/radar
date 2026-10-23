# -*- coding: utf-8 -*-
"""对 core_500 的逐场景结果做严格配对统计。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
RESULTS = HERE.parents[1] / "results" / "core_500"
METHODS = ["q_only", "retrieve_q", "retrieve_q_union", "ensemble_ctx_mean", "ensemble_ctx_pessimistic"]
REFERENCE = "retrieve_q_union"
N_BOOT = 10000
N_PERM = 100000
SEED = 20261022


def bootstrap_ci(x, rng):
    n = len(x)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        vals[i] = rng.choice(x, n, replace=True).mean()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def permutation_p(diff, rng):
    obs = abs(float(diff.mean()))
    count = 0
    block = 2000
    for start in range(0, N_PERM, block):
        k = min(block, N_PERM - start)
        signs = rng.choice((-1.0, 1.0), size=(k, len(diff)))
        count += int((np.abs((signs * diff).mean(axis=1)) >= obs).sum())
    return float((count + 1) / (N_PERM + 1))


def holm(pvals):
    order = np.argsort(pvals)
    out = np.empty(len(pvals))
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        out[idx] = min(1.0, running)
    return out.tolist()


def main():
    df = pd.read_csv(RESULTS / "per_scenario.csv")
    # fresh450 is the primary untouched evaluation subset.
    df = df.iloc[50:].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    ref = df[f"{REFERENCE}__sinr"].to_numpy(float)
    rows = []
    for method in METHODS:
        rec = df[f"{method}__sinr"].to_numpy(float)
        reg = df[f"{method}__regret_k20"].to_numpy(float)
        delta = rec - ref
        try:
            wp = float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
        except ValueError:
            wp = 1.0
        rows.append({
            "method": method, "n": len(df), "mean_regret_k20": float(reg.mean()),
            "median_regret_k20": float(np.median(reg)), "p90_regret_k20": float(np.percentile(reg, 90)),
            "p95_regret_k20": float(np.percentile(reg, 95)),
            "cvar10_regret_k20": float(np.sort(reg)[-int(np.ceil(.1*len(reg))):].mean()),
            "mean_delta_sinr_vs_union": float(delta.mean()),
            "delta_ci95": bootstrap_ci(delta, rng),
            "win_rate_vs_union": float((delta > 1e-6).mean()),
            "tie_rate_vs_union": float((np.abs(delta) <= 1e-6).mean()),
            "permutation_p": permutation_p(delta, rng), "wilcoxon_p": wp,
        })
    # Five stratified scenario folds expose split-to-split variability without
    # pretending that repeated partitions are independent test sets.
    fold_id = np.full(len(df), -1, dtype=int)
    for _, idx in df.groupby("jammer_type").groups.items():
        idx = np.asarray(list(idx)); idx = rng.permutation(idx)
        for k, part in enumerate(np.array_split(idx, 5)):
            fold_id[part] = k
    fold_rows = []
    for k in range(5):
        mask = fold_id == k
        for method in METHODS:
            x = df.loc[mask, f"{method}__regret_k20"].to_numpy(float)
            fold_rows.append({"fold": k, "method": method, "n": int(mask.sum()),
                              "mean_regret_k20": float(x.mean()), "median_regret_k20": float(np.median(x))})
    for key in ("permutation_p", "wilcoxon_p"):
        adjusted = holm([r[key] for r in rows])
        for r, p in zip(rows, adjusted):
            r[key + "_holm"] = p
    out = {"subset": "fresh450", "reference": REFERENCE, "n_boot": N_BOOT,
           "n_permutations": N_PERM, "seed": SEED, "results": rows,
           "five_stratified_folds": fold_rows}
    (RESULTS / "paired_statistics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = [{**r, "delta_ci95_lo": r["delta_ci95"][0], "delta_ci95_hi": r["delta_ci95"][1]}
            for r in rows]
    pd.DataFrame(flat).drop(columns="delta_ci95").to_csv(RESULTS / "paired_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(RESULTS / "five_fold_stability.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
