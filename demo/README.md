# 抗干扰决策演示 Web

基于 **Retrieve+Q** 的交互演示：输入干扰机参数，输出推荐雷达参数与 SINR。

## 前置条件

已运行主流程并完成模型训练：

- `data/v2/processed/artifacts.pkl`
- `outputs/state_vae/state_ae_or_vae.pt`
- `outputs/q_value/latent_q/q_latent_best.pt`

## 启动

```bash
cd /path/to/radar
pip install fastapi "uvicorn[standard]"
PYTHONPATH=src python scripts/run_demo_server.py
```

浏览器打开：**http://127.0.0.1:8765**

## API

| 接口 | 说明 |
|------|------|
| `GET /api/health` | 服务与模型状态 |
| `GET /api/meta` | 干扰类型、默认值、方法列表 |
| `POST /api/recommend` | 单方法推荐 |
| `POST /api/compare` | Retrieve+Q / Q-only / Retrieval-only 对比 |

## 代码结构

| 路径 | 作用 |
|------|------|
| `src/radar/infer/demo_recommend.py` | 推理封装 |
| `src/radar/web/app.py` | FastAPI 服务 |
| `demo/static/index.html` | 前端页面 |
| `scripts/run_demo_server.py` | 启动脚本 |
