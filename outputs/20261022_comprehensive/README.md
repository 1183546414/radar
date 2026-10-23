# 2026-10-22 全面实验

本目录用于扩充固定 50 场景评估，所有代码与结果均独立保存，不覆盖历史轮次。

实验模块：

1. `core_500_eval.py`：500 场景核心方法评估、前 50/后 450 分解、配对统计、分类型统计。
2. `sweep_eval.py`：悲观系数、Top-K、候选池规模与 JSR 条件扫描。
3. `robust_formal.py`：20/50 随机方向和坐标式最坏扰动搜索。
4. `statistical_report.py`：配对 bootstrap、置换检验、Wilcoxon、Holm 校正与风险指标。

运行环境：`D:\conda\envs\radar\python.exe`。

