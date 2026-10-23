# -*- coding: utf-8 -*-
"""对历史报告未使用的后 450 场景中的前 250 个计算稠密近似 oracle。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve(); ROOT=HERE.parents[3]
SRC=ROOT/"outputs"/"20261021_dense_oracle"/"code"/"dense_oracle.py"
sp=importlib.util.spec_from_file_location("dense_fresh",SRC); m=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(m)
OUT=HERE.parents[1]/"results"/"dense_oracle_fresh250"; OUT.mkdir(parents=True,exist_ok=True)
src=ROOT/"data"/"v2"/"processed"/"holdout_eval_scenarios.csv"; df=pd.read_csv(src); sids=df.scenario_id.unique()[50:300]
subset=OUT/"holdout_fresh250.csv"; df[df.scenario_id.isin(sids)].to_csv(subset,index=False,encoding="utf-8-sig")
m.HOLDOUT_CSV=subset; m.OUT_DIR=OUT; m.N_HOLDOUT=250; m.GLOBAL_SEED=20261022
m.main()
