# -*- coding: utf-8 -*-
"""λ、Top-K、候选池规模和 JSR 扫描（500 场景，冻结模型）。"""
from __future__ import annotations
import importlib.util, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE=Path(__file__).resolve(); ROOT=HERE.parents[3]; OUT=HERE.parents[1]/"results"/"sweeps"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"outputs"/"20260909_signal_level"/"code")]
SRC=ROOT/"outputs"/"20261021_dense_oracle"/"code"/"eval_fewshot_current.py"
sp=importlib.util.spec_from_file_location("eval_fs_sweep",SRC); fs=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(fs)
from signal_sinr import SignalSinrBackend

SEED=2026; N=500

def summarize(vals):
 x=np.asarray(vals,float); return {"n":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"p90":float(np.percentile(x,90)),"p95":float(np.percentile(x,95)),"cvar10":float(np.sort(x)[-int(np.ceil(.1*len(x))):].mean())}

def evaluate_configs(ctx,models,device,configs,jsr=10.0):
 backend=SignalSinrBackend(jsr_db=jsr); fs.hb.set_sinr_backend(backend.sinr)
 regrets={c["name"]:[] for c in configs}; sinrs={c["name"]:[] for c in configs}
 for ii,sid in enumerate(ctx.holdout_sids):
  block=ctx.holdout_df[ctx.holdout_df.scenario_id==sid]; j=block.iloc[0]; _,z=fs.hb._encode_query(ctx,j)
  ok=fs.hb._oracle_k20(block)
  for c in configs:
   topk=c.get("top_k",5); pool=c.get("pool",100); lam=c.get("lambda",1.0)
   ns,sim=fs.hb._topk_similar_scenarios(z,ctx.z_bank,ctx.sid_bank,topk,str(j.Jammer_Type),ctx.jammer_type_bank)
   retr=fs.hb._collect_candidate_row_indices(ctx.train_scenario_ids,ns)
   rng=np.random.default_rng(SEED+int(sid))
   union=fs.hb.build_union_candidate_rows(retr,ctx.train_row_pool,pool,rng)
   zc=fs.hb._z_ctx_from_neighbor_sids(ctx,ns,sim); amat=ctx.A_std[union]
   qall=np.stack([fs.predict_q_retrieval(m,z.reshape(1,-1),amat,zc.reshape(1,-1),device) for m in models])
   score=qall.mean(0)-lam*qall.std(0); ridx=int(union[int(score.argmax())])
   val=fs.hb._sinr_on_query(j,ctx.train_df,ridx); sinrs[c["name"]].append(val); regrets[c["name"]].append(ok-val)
  if (ii+1)%50==0: print(f"JSR={jsr:g} {ii+1}/{len(ctx.holdout_sids)}",flush=True)
 return [{**c,"jsr_db":jsr,"all500":summarize(regrets[c["name"]]),"fresh450":summarize(regrets[c["name"]][50:]),"mean_sinr_fresh450":float(np.mean(sinrs[c["name"]][50:]))} for c in configs],backend.stats()

def main():
 t0=time.time(); OUT.mkdir(parents=True,exist_ok=True); random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 fs.OUT_DIR=OUT;fs._patch_hb_paths();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 ctx=fs.hb.load_holdout_context(n_holdout=N,holdout_seed=SEED);models=fs._load_ensemble_models(device)
 configs=[]
 for v in (0,.25,.5,1,1.5,2,3): configs.append({"name":f"lambda_{v:g}","lambda":v,"top_k":5,"pool":100,"sweep":"lambda"})
 for v in (1,3,5,10,20): configs.append({"name":f"topk_{v}","lambda":1,"top_k":v,"pool":100,"sweep":"top_k"})
 for v in (5,20,50,100,500): configs.append({"name":f"pool_{v}","lambda":1,"top_k":5,"pool":v,"sweep":"pool"})
 rows,stats=evaluate_configs(ctx,models,device,configs,10.0)
 # JSR transfer uses the chosen default configuration; model and candidate selection remain frozen.
 jsrrows=[]; jsrstats={}
 for j in (0,5,15,20):
  rr,ss=evaluate_configs(ctx,models,device,[{"name":"default","lambda":1,"top_k":5,"pool":100,"sweep":"jsr"}],float(j));jsrrows+=rr;jsrstats[str(j)]=ss
 allrows=rows+jsrrows
 (OUT/"summary.json").write_text(json.dumps({"protocol":{"n":N,"seed":SEED,"frozen_models":True},"rows":allrows,"backend_jsr10":stats,"backend_other_jsr":jsrstats,"elapsed_min":round((time.time()-t0)/60,2)},ensure_ascii=False,indent=2),encoding="utf-8")
 pd.json_normalize(allrows).to_csv(OUT/"summary.csv",index=False,encoding="utf-8-sig")
 print(json.dumps(allrows,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
