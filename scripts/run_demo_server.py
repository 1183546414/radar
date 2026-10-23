#!/usr/bin/env python3
"""启动抗干扰决策演示 Web 服务。

用法（项目根目录）:
    PYTHONPATH=src python scripts/run_demo_server.py
    # 浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from radar.web.app import create_app

if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
