# -*- coding: utf-8 -*-
"""汇总本轮全部结果并把 250 场景重算为真 regret。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve(); BASE=HERE.parents[1]; RES=BASE/"results"
CORE=RES/"core_500"; DENSE=RES/"dense_oracle_fresh250"; METHODS=["q_only","retrieve_q","retrieve_q_union","ensemble_ctx_mean","ensemble_ctx_pessimistic"]

def main():
 core=pd.read_csv(CORE/"per_scenario.csv"); dense=pd.read_csv(DENSE/"dense_oracle_actions.csv")[["scenario_id","dense_oracle_db"]]
 df=core.merge(dense,on="scenario_id",how="inner"); assert len(df)==250,len(df)
 rows=[]
 for m in METHODS:
  x=df.dense_oracle_db-df[f"{m}__sinr"]
  rows.append({"method":m,"n":len(x),"mean_true_regret":float(x.mean()),"median_true_regret":float(x.median()),
   "p90":float(np.percentile(x,90)),"p95":float(np.percentile(x,95)),"cvar10":float(np.sort(x)[-25:].mean()),"oracle_rate":float((x<=1e-4).mean())})
 pd.DataFrame(rows).to_csv(RES/"true_regret_fresh250.csv",index=False,encoding="utf-8-sig")
 robust=json.loads((RES/"robust_formal"/"summary.json").read_text(encoding="utf-8")); sweeps=json.loads((RES/"sweeps"/"summary.json").read_text(encoding="utf-8")); paired=json.loads((CORE/"paired_statistics.json").read_text(encoding="utf-8"))
 sr=pd.json_normalize(sweeps["rows"]); best_top=sr[sr.sweep=="top_k"].sort_values("fresh450.mean").iloc[0]; best_pool=sr[sr.sweep=="pool"].sort_values("fresh450.mean").iloc[0]; best_lam=sr[sr.sweep=="lambda"].sort_values("fresh450.mean").iloc[0]
 r10=next(x for x in robust["by_radius"] if abs(x["rho"]-.1)<1e-9)
 lines=["# 2026-10-22 全面扩展实验报告","","## 结论","",
  f"本轮将主评估从固定 50 场景扩展到 500 场景，其中后 450 场景未出现在历次主报告；另对其中 250 场景计算 best-of-5000 + 坐标细化稠密 oracle。",
  "","## 250 个新场景：真 regret","","| 方法 | mean | median | p90 | p95 | CVaR10 |","|---|---:|---:|---:|---:|---:|"]
 for r in rows: lines.append(f"| {r['method']} | {r['mean_true_regret']:.2f} | {r['median_true_regret']:.2f} | {r['p90']:.2f} | {r['p95']:.2f} | {r['cvar10']:.2f} |")
 lines += ["","## 450 个新场景：配对结论","",
  "Ensemble 相对 Union 的平均提升很小且经配对检验不显著；可靠收益集中在 p95/CVaR 尾部风险。",
  "","## 超参数扫描","",
  f"- 最佳 λ：{best_lam['lambda']}，fresh450 弱参照 regret {best_lam['fresh450.mean']:.2f} dB。",
  f"- 最佳 Top-K：{int(best_top['top_k'])}，fresh450 弱参照 regret {best_top['fresh450.mean']:.2f} dB。",
  f"- 最佳随机池：{int(best_pool['pool'])}，fresh450 弱参照 regret {best_pool['fresh450.mean']:.2f} dB。",
  "","## 鲁棒性","",
  f"10% 扰动下，最坏方向平均退化 {r10['mean_worst_degradation']:.2f} dB，p95 {r10['p95_worst_degradation']:.2f} dB，至少一次决策翻转率 {100*r10['any_flip_rate']:.1f}%。",
  "","## 统计说明","",
  "所有方法在相同场景上做配对比较；提供 10,000 次 bootstrap、100,000 次符号置换、Wilcoxon 与 Holm 多重比较校正。JSR 扫描是冻结策略的跨条件测试。"]
 (BASE/"全面实验报告.md").write_text("\n".join(lines),encoding="utf-8")
 (RES/"final_summary.json").write_text(json.dumps({"true_regret_fresh250":rows,"paired":paired,"robust":robust,"sweeps":sweeps},ensure_ascii=False,indent=2),encoding="utf-8")
 print("\n".join(lines))
if __name__=="__main__":main()
