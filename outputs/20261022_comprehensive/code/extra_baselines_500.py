# -*- coding: utf-8 -*-
"""补充 Random 与 Retrieval-max-R 的 500 场景结果。"""
from __future__ import annotations
import importlib.util, json, random, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve(); ROOT=HERE.parents[3]; OUT=HERE.parents[1]/"results"/"extra_baselines_500"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"outputs"/"20260909_signal_level"/"code")]
SRC=ROOT/"outputs"/"20261021_dense_oracle"/"code"/"eval_fewshot_current.py"
sp=importlib.util.spec_from_file_location("eval_fs_extra",SRC);fs=importlib.util.module_from_spec(sp);assert sp.loader;sp.loader.exec_module(fs)
from signal_sinr import SignalSinrBackend
OUT.mkdir(parents=True,exist_ok=True);fs.OUT_DIR=OUT;fs._patch_hb_paths()
fs.hb.OUT_DIR=OUT;fs.hb.SUMMARY_JSON=OUT/"summary.json";fs.hb.SUMMARY_CSV=OUT/"summary.csv";fs.hb.PER_SCENARIO_JSON=OUT/"per_scenario_detail.json"
random.seed(2026);np.random.seed(2026);b=SignalSinrBackend(jsr_db=10);fs.hb.set_sinr_backend(b.sinr)
result=fs.hb.run_all_baselines(methods=["random","retrieval_max_r"],n_holdout=500,holdout_seed=2026,verbose=True)
result["backend_stats"]=b.stats();(OUT/"summary_with_backend.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
