# -*- coding: utf-8 -*-
"""正式鲁棒性评估：更多随机方向 + 坐标极值搜索。"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "results" / "robust_formal"
SRC = HERE.parents[3] / "outputs" / "20261021_robust_pre" / "code" / "pre_perturb_sensitivity.py"
spec = importlib.util.spec_from_file_location("robust_pre", SRC)
rp = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(rp)

RADII = (0.02, 0.05, 0.10, 0.15, 0.20)
N_DIR = 20
SEED = 20261022


def metrics(sc, jam):
    pool = np.asarray([rp.sinr_of(a, jam) for a in sc["actions"]])
    old = float(pool[sc["best"]])
    return float(sc["base"][sc["best"]] - old), float(pool.max() - old), int(pool.argmax() != sc["best"])


def coordinate_candidates(jam, rho):
    dims = ["j_freq", "j_gain", "j_bw", "j_sweep_step_hz", "j_int_dur_ns", "j_int_delay_ns", "j_ft_num"]
    for dim in dims:
        if dim not in jam or (dim not in ("j_freq", "j_gain") and jam[dim] <= 0):
            continue
        for sign in (-1, 1):
            j = dict(jam)
            if dim == "j_gain": j[dim] = float(np.clip(jam[dim] + sign*10*rho, rp.J_GAIN_LO, rp.J_GAIN_HI))
            elif dim == "j_freq": j[dim] = float(np.clip(jam[dim]*(1+sign*rho), rp.J_FREQ_LO, rp.J_FREQ_HI))
            elif dim == "j_bw":
                lo, hi = rp.R_BANDWIDTH.get(jam["j_type"], (rp.J_BW_LO, rp.J_BW_HI))
                j[dim] = float(np.clip(jam[dim]*(1+sign*rho), max(lo,rp.J_BW_LO), min(hi,rp.J_BW_HI)))
            elif dim == "j_sweep_step_hz": j[dim] = float(np.clip(jam[dim]*(1+sign*rho),rp.SWEEP_LO,rp.SWEEP_HI))
            elif dim == "j_int_dur_ns": j[dim] = float(np.clip(jam[dim]*(1+sign*rho),rp.DUR_LO,rp.DUR_HI))
            elif dim == "j_int_delay_ns": j[dim] = float(np.clip(jam[dim]*(1+sign*rho),rp.DELAY_LO,rp.DELAY_HI))
            elif dim == "j_ft_num": j[dim] = float(np.clip(round(jam[dim]*(1+sign*rho)),rp.FT_LO,rp.FT_HI))
            if dim in ("j_int_dur_ns", "j_int_delay_ns"):
                j["j_int_intv_ns"] = j["j_int_dur_ns"] + j["j_int_delay_ns"] + 200.0
            yield dim, sign, j


def main():
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    scenarios,sanity=rp.load_scenarios(); rng=np.random.default_rng(SEED); rows=[]
    for rho in RADII:
        tr=time.time()
        for i,sc in enumerate(scenarios):
            candidates=[]
            for d in range(N_DIR):
                candidates.append(("random",d,rp.perturb_jam(sc["jam"],rho,rng)))
            candidates += [(f"coord:{dim}",sign,j) for dim,sign,j in coordinate_candidates(sc["jam"],rho)]
            local=[]
            for search,idx,j in candidates:
                deg,reg,flip=metrics(sc,j); local.append((deg,reg,flip,search,idx))
            worst=max(local,key=lambda z:z[0])
            rows.append({"scenario_id":sc["sid"],"jammer_type":sc["jtype"],"rho":rho,
                         "n_candidates":len(local),"worst_degradation":worst[0],
                         "static_regret_at_worst":worst[1],"flip_at_worst":worst[2],
                         "worst_search":worst[3],"worst_index":worst[4],
                         "max_static_regret":max(z[1] for z in local),
                         "any_flip":int(any(z[2] for z in local))})
            if (i+1)%500==0: print(f"rho={rho:.0%} {i+1}/{len(scenarios)}",flush=True)
        print(f"rho={rho:.0%} done {time.time()-tr:.1f}s",flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/"per_scenario_worst.csv",index=False,encoding="utf-8-sig")
    agg=df.groupby("rho").agg(n=("scenario_id","size"),mean_worst_degradation=("worst_degradation","mean"),
        median_worst_degradation=("worst_degradation","median"),p95_worst_degradation=("worst_degradation",lambda x:np.percentile(x,95)),
        cvar10_worst_degradation=("worst_degradation",lambda x:np.sort(x)[-int(np.ceil(.1*len(x))):].mean()),
        mean_max_static_regret=("max_static_regret","mean"),any_flip_rate=("any_flip","mean")).reset_index()
    agg.to_csv(OUT/"summary_by_radius.csv",index=False,encoding="utf-8-sig")
    bt=df[df.rho==.10].groupby("jammer_type").agg(n=("scenario_id","size"),mean_worst_degradation=("worst_degradation","mean"),
        p95_worst_degradation=("worst_degradation",lambda x:np.percentile(x,95)),any_flip_rate=("any_flip","mean")).reset_index()
    bt.to_csv(OUT/"summary_by_type_rho10.csv",index=False,encoding="utf-8-sig")
    summary={"seed":SEED,"n_scenarios":len(scenarios),"n_random_directions":N_DIR,"radii":list(RADII),
             "search":"random + one-coordinate boundary search","sanity":sanity,"by_radius":agg.to_dict('records'),
             "elapsed_min":round((time.time()-t0)/60,2)}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
