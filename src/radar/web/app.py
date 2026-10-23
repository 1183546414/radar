# -*- coding: utf-8 -*-
"""抗干扰决策演示 Web 服务（FastAPI）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from radar.infer.demo_recommend import (
    compare_methods,
    get_demo_meta,
    load_demo_context,
    recommend_jammer,
)
from radar.paths import OUT_FIGURES, PROJECT_ROOT
from radar.viz.waveform_sim import generate_jammed_waveform_figures
from radar.web.training_viz import get_training_viz_meta

STATIC_DIR = PROJECT_ROOT / "demo" / "static"


class JammerInput(BaseModel):
    jammer_type: str = Field(..., description="干扰类型")
    carrier_freq_ghz: float = Field(..., ge=3.0, le=5.0)
    gain_db: float = Field(..., ge=20.0, le=40.0)
    bandwidth_ghz: float = Field(..., ge=0.001, le=0.6)
    sweep_step_ghz: float = Field(0.0, ge=0.0)
    int_sample_dur_ns: float = Field(0.0, ge=0.0)
    int_sample_delay_ns: float = Field(0.0, ge=0.0)
    int_sample_intv_ns: float = Field(0.0, ge=0.0)
    false_target_num: float = Field(0.0, ge=0.0)


class RecommendRequest(BaseModel):
    jammer: JammerInput
    method: Literal["retrieve_q_union", "q_only", "retrieval_max_r"] = "retrieve_q_union"
    seed: int = 42


class CompareRequest(BaseModel):
    jammer: JammerInput
    seed: int = 42


class RadarInput(BaseModel):
    radar_type: str = Field("线性调频0", description="雷达波形类型")
    chirp_type: str = Field("NegativeTriangle", description="Chirp 类型")
    carrier_freq_ghz: float = Field(..., ge=3.0, le=5.0)
    gain_db: float = Field(..., ge=20.0, le=40.0)
    bandwidth_ghz: float = Field(..., ge=0.005, le=0.05)
    pulse_width_ns: float = Field(30000.0, ge=20000.0, le=40000.0)


class WaveformRequest(BaseModel):
    jammer: JammerInput
    radar: RadarInput


def create_app() -> FastAPI:
    app = FastAPI(
        title="Radar Anti-Jamming Demo",
        description="Retrieve+Q 抗干扰雷达参数推荐演示",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _warmup() -> None:
        try:
            load_demo_context()
        except FileNotFoundError as exc:
            print(f"[WARN] 模型未就绪，请先运行 main.py: {exc}")

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        try:
            load_demo_context()
            return {"status": "ok", "models": "loaded"}
        except FileNotFoundError as exc:
            return {"status": "missing_models", "detail": str(exc)}

    @app.get("/api/meta")
    def meta() -> Dict[str, Any]:
        try:
            return get_demo_meta()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/recommend")
    def recommend(body: RecommendRequest) -> Dict[str, Any]:
        try:
            return recommend_jammer(
                body.jammer.model_dump(),
                method=body.method,
                seed=body.seed,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/compare")
    def compare(body: CompareRequest) -> Dict[str, Any]:
        try:
            return compare_methods(body.jammer.model_dump(), seed=body.seed)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/training_viz")
    def training_viz() -> Dict[str, Any]:
        return get_training_viz_meta()

    @app.post("/api/waveform")
    def waveform(body: WaveformRequest) -> Dict[str, Any]:
        try:
            return generate_jammed_waveform_figures(
                body.jammer.model_dump(),
                body.radar.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if OUT_FIGURES.is_dir():
        app.mount("/figures", StaticFiles(directory=OUT_FIGURES), name="figures")

    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
