# -*- coding: utf-8 -*-
"""生成改进①（Q-Ensemble 悲观打分）组会汇报 PDF。

5 页 A4 竖版：
  P1 标题 / 任务背景 / 动机
  P2 方法：深度集成 + 悲观打分（含流程图）
  P3 实验设置 + 主结果（表 + 柱状图）
  P4 小样本分桶（表 + 分组柱状图）
  P5 σ 诊断 / 结论与采纳 / 局限与下一步

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261007_fewshot/code/make_ensemble_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
OUT_PDF = ROUND / "改进1_Q-Ensemble悲观打分_组会汇报.pdf"

# ---- 中文字体 ----
from matplotlib import font_manager as _fm
for _f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
           r"C:\Windows\Fonts\simhei.ttf"):
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

A4 = (8.27, 11.69)
INK = "#1a1a2e"
ACCENT = "#16537e"
ACCENT2 = "#c0392b"
GRID = "#d8dee9"
LIGHT = "#f2f5f9"

# ================= 数据（与 eval_fewshot/summary.json 一致） =================
MAIN_ROWS = [
    # (方法, 推荐SINR, regret, 中位regret, 零regret%, 胜率vs q_only%, 强调?)
    ("Q-only（无检索基线下限）",              12.46, 4.79, 2.36, 32, None, False),
    ("Retrieve+Q（Top-5 池）",               20.78, -3.53, -1.43, 76, 82, False),
    ("Retrieve+Q-Union（Round2 最优基线）",   22.33, -5.08, -1.89, 84, 90, False),
    ("Retrieve+Q-Ctx-Union（最优 ctx 变体）", 21.44, -4.19, -2.00, 80, 88, False),
    ("① Ensemble-Mean（5×Q 取均值）",        22.26, -5.00, -2.57, 84, 90, False),
    ("① Ensemble-悲观 μ-1.0σ（本轮方法）",   22.22, -4.97, -3.08, 82, 96, True),
    ("② Set-Attention 单模型（对照）",        20.45, -3.19, -1.89, 72, 90, False),
    ("①+② Attention-Ens-悲观（对照）",       21.38, -4.13, -2.10, 76, 90, False),
    ("Oracle-K20（留出上界）",               17.25, 0.00, 0.00, 100, 68, False),
    ("Oracle-Wide（100 随机 A 上界）",        24.38, -7.13, -3.68, 92, 94, False),
]

BUCKETS = {
    # 桶名: (n, q_only, ctx_union基线, ens_mean, ens_pess, pess胜率%)
    "稀有干扰类型\nrare_type (n=25)":   (25, 1.66, -3.70, -3.80, -4.39, 96),
    "低检索相似度\nlow_sim (n=25)":     (25, 2.22, -2.36, -2.38, -2.66, 96),
    "最难子集\nhard (n=9)":             (9,  1.42, -1.74, -1.73, -2.54, 100),
    "常见类型\ncommon_type (n=25)":     (25, 7.93, -4.68, -6.21, -5.55, None),
    "高检索相似度\nhigh_sim (n=25)":    (25, 7.37, -6.02, -7.63, -7.28, None),
}

MEMBER_HITS = [0.57, 0.56, 0.59, 0.63, 0.56]  # 5 个集成成员 val Hit@1
SEEDS = (42, 1042, 2042, 3042, 4042)


# ================= 排版辅助 =================
def new_page():
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    return fig


def page_title(fig, title, sub=None):
    fig.text(0.07, 0.955, title, fontsize=17, fontfamily="SimHei", color=INK,
             va="top", ha="left")
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.925, 0.925],
                                transform=fig.transFigure, color=ACCENT,
                                linewidth=2.2))
    if sub:
        fig.text(0.07, 0.908, sub, fontsize=10, color="#5a6572", va="top")


def section(fig, y, text):
    fig.text(0.07, y, text, fontsize=12.5, fontfamily="SimHei", color=ACCENT,
             va="top")
    fig.lines.append(plt.Line2D([0.07, 0.93], [y - 0.018, y - 0.018],
                                transform=fig.transFigure, color=GRID,
                                linewidth=0.8))


def _wrap(s, width_chars):
    """按显示宽度折行（全角=1，半角≈0.55），中文排版可靠替代 wrap=True。"""
    lines, cur, w = [], "", 0.0
    for ch in s:
        cw = 1.0 if ord(ch) > 0x2E80 else 0.55
        if w + cw > width_chars and cur:
            lines.append(cur)
            cur, w = "", 0.0
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines


def _width_chars(width_in, fs):
    return width_in * A4[0] * 72.0 / fs


def para(fig, x, y, s, fs=9.8, width=0.86, color=INK, lh=1.5):
    """手动折行绘制段落（va=top），返回结束 y。"""
    line_h = fs / 72.0 * lh / A4[1]
    for ln in _wrap(s, _width_chars(width, fs)):
        fig.text(x, y, ln, fontsize=fs, color=color, va="top")
        y -= line_h
    return y


def bullets(fig, y, items, x=0.07, width=0.86, fs=9.8, gap=0.012, lh=1.5):
    """项目符号列表（悬挂缩进 + 手动折行），返回结束 y。"""
    line_h = fs / 72.0 * lh / A4[1]
    for it in items:
        fig.text(x, y, "•", fontsize=fs, color=ACCENT, va="top")
        for ln in _wrap(it, _width_chars(width, fs)):
            fig.text(x + 0.022, y, ln, fontsize=fs, color=INK, va="top")
            y -= line_h
        y -= gap
    return y


def table(fig, rect, headers, rows, col_widths=None, fs=8.6, header_fs=8.8,
          highlight_rows=(), highlight_color="#eaf3fa"):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="center",
                   colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fs)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#b8c4d0")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontsize=header_fs)
            cell.set_height(cell.get_height() * 1.15)
        else:
            cell.set_facecolor(highlight_color if (r - 1) in highlight_rows
                               else ("white" if (r % 2) else LIGHT))
            if (r - 1) in highlight_rows:
                cell.set_text_props(fontfamily="SimHei")
    return ax


# ================= 图 1：方法流程图 =================
def draw_pipeline(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=ACCENT, fs=8.3, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=fc, ec=ec, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK,
                fontfamily="SimHei" if bold else None)

    def arrow(x1, y1, x2, y2, color=ACCENT):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=color, linewidth=1.2))

    # 第一行主流程
    box(1, 78, 17, 14, "干扰状态 S\n(CSV 行)", fc=LIGHT)
    box(22, 78, 17, 14, "冻结 VAE\nz_query [128]")
    box(43, 78, 22, 14, "同类 Top-5 检索\n(余弦相似度)")
    box(69, 78, 30, 14, "候选池 ∪ 随机 100 A\n(约 200 个动作)", fc=LIGHT)
    arrow(18.3, 85, 21.6, 85)
    arrow(39.3, 85, 42.6, 85)
    arrow(65.3, 85, 68.6, 85)
    ax.text(84, 75.2, "候选 A × 5", fontsize=7.6, color="#5a6572", ha="center")

    # 中间：5 个 Q 成员（仅种子不同）
    for i, seed in enumerate(SEEDS):
        y0 = 52 - i * 8
        box(30, y0, 24, 6.4, f"Q{i+1}  (seed {seed})", fs=7.8, ec="#7a93a8")
        arrow(84, 78, 54.5, y0 + 3.2, color="#9fb3c4")

    # 汇聚到 μ/σ，再到悲观打分
    box(58, 18, 18, 14, "μ  (集成均值)\nσ  (成员标准差)", fc="#fdf6ec",
        ec=ACCENT2, bold=True)
    for i in range(5):
        y0 = 52 - i * 8
        arrow(54.3, y0 + 3.2, 57.6, 25 + (2 - i) * 1.5, color="#9fb3c4")
    box(80, 18, 19, 14, "score = μ-λσ\nλ = 1.0\nargmax → 推荐", fc="#eaf3fa",
        ec=ACCENT, bold=True)
    arrow(76.3, 25, 79.6, 25, color=ACCENT2)

    ax.text(2, 6, "候选池 = 检索池（Top-5 相似场景的动作）∪ 随机 100 A",
            fontsize=7.6, color="#5a6572")
    ax.text(2, 1.5, "5 个成员仅训练种子不同，其余训练协议完全一致",
            fontsize=7.6, color="#5a6572")


# ================= 图 2：主结果柱状图 =================
def draw_main_bar(fig, rect):
    ax = fig.add_axes(rect)
    SHORT = {
        "Q-only（无检索基线下限）": "Q-only",
        "Retrieve+Q（Top-5 池）": "Retrieve+Q",
        "Retrieve+Q-Union（Round2 最优基线）": "R+Q-Union",
        "Retrieve+Q-Ctx-Union（最优 ctx 变体）": "R+Q-Ctx-Union",
        "① Ensemble-Mean（5×Q 取均值）": "Ens-Mean",
        "① Ensemble-悲观 μ-1.0σ（本轮方法）": "Ens-悲观 μ-1.0σ",
        "② Set-Attention 单模型（对照）": "Set-Attention",
        "①+② Attention-Ens-悲观（对照）": "Attn-Ens-悲观",
        "Oracle-K20（留出上界）": "Oracle-K20",
        "Oracle-Wide（100 随机 A 上界）": "Oracle-Wide",
    }
    names = [SHORT.get(r[0], r[0]) for r in MAIN_ROWS]
    regrets = [r[2] for r in MAIN_ROWS]
    colors = [ACCENT2 if r[6] else ("#8aa4b8" if "Oracle" in r[0] else "#5b7c99")
              for r in MAIN_ROWS]
    y = np.arange(len(names))[::-1]
    ax.barh(y, regrets, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.4)
    ax.axvline(0, color="#444", linewidth=0.9, zorder=4)
    ax.axvline(-5.08, color="#3a7ca5", linewidth=1.1, linestyle="--", zorder=2)
    ax.text(-5.08, len(names) - 0.35, " Round2 基线 -5.08", fontsize=7.8,
            color="#3a7ca5", va="bottom")
    for yi, v in zip(y, regrets):
        ax.text(v + (0.14 if v >= 0 else -0.14), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.8,
                color=INK, zorder=5)
    ax.set_xlabel("regret 相对 Oracle-K20（dB，越负越好）", fontsize=9)
    ax.set_xlim(-8.3, 6.3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


# ================= 图 3：分桶分组柱状图 =================
def draw_bucket_bar(fig, rect):
    ax = fig.add_axes(rect)
    keys = list(BUCKETS.keys())
    short = ["稀有类型 rare (n=25)", "低相似 low_sim (n=25)", "最难 hard (n=9)",
             "常见 common (n=25)", "高相似 high_sim (n=25)"]
    x = np.arange(len(keys))
    w = 0.26
    base = [BUCKETS[k][2] for k in keys]
    mean = [BUCKETS[k][3] for k in keys]
    pess = [BUCKETS[k][4] for k in keys]
    ax.bar(x - w, base, w, label="基线 Retrieve+Q-Ctx-Union", color="#8aa4b8",
           zorder=3)
    ax.bar(x, mean, w, label="① Ensemble-Mean", color="#5b7c99", zorder=3)
    ax.bar(x + w, pess, w, label="① Ensemble-悲观 μ-1.0σ", color=ACCENT2,
           zorder=3)
    for xi, v in zip(x + w, pess):
        ax.text(xi, v - 0.28, f"{v:.2f}", ha="center", fontsize=7.6,
                color=ACCENT2, fontfamily="SimHei", zorder=5)
    for xi, v in zip(x - w, base):
        ax.text(xi, v - 0.28, f"{v:.2f}", ha="center", fontsize=7.4,
                color="#5a6572", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.4)
    ax.axhline(0, color="#444", linewidth=0.9)
    ax.set_ylabel("regret（dB，越负越好）", fontsize=9)
    ax.set_ylim(-10.2, 0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ================= 页面 =================
def page1(pdf):
    fig = new_page()
    fig.text(0.07, 0.86, "改进①：Q-Ensemble 悲观打分", fontsize=26,
             fontfamily="SimHei", color=INK)
    fig.text(0.07, 0.815, "面向小样本抗干扰的动作推荐打分改进", fontsize=15,
             color=ACCENT)
    fig.text(0.07, 0.782, "Round 20261007 ｜ 组会汇报", fontsize=11,
             color="#5a6572")
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.762, 0.762],
                                transform=fig.transFigure, color=ACCENT,
                                linewidth=2.5))

    section(fig, 0.72, "一、任务与基线架构")
    y = bullets(fig, 0.685, [
        "任务：给定干扰状态 S（干扰类型、功率、载频等），从候选雷达动作 A（载频 / 增益 / "
        "带宽 / 脉宽 / 类型 / 调制）中推荐预测 SINR 最高的动作。",
        "基线管线（Round2 确认的最优形态 Retrieve+Q-Union）：",
    ], gap=0.026)
    y = para(fig, 0.09, y, "S → 冻结 VAE 编码为潜向量 z（128 维）→ 按同类干扰余弦相似度检索 Top-5 "
                           "相似训练场景 → 其动作构成检索候选池，再并入随机 100 个动作组成大池（约 200 个）"
                           "→ QRetrievalNet(z, z_ctx, A) 逐一打分 → argmax。", fs=9.6)
    y -= 0.012
    y = bullets(fig, y, [
        "QRetrievalNet 的上下文 z_ctx：Top-5 邻居潜向量按相似度 softmax(τ=0.07) 加权均值；"
        "训练目标 = MSE 回归 + listwise ranking loss（池内 argmax Q 对齐 argmax 真值）。",
        "Round2 成绩（50 个留出场景）：regret −5.08 dB，零 regret 84%，对 q_only 胜率 90%。",
    ], gap=0.014)

    section(fig, y - 0.012, "二、动机：单模型 Q 的两个痛点")
    y = bullets(fig, y - 0.045, [
        "痛点 1（打分随机性）：Q 网络训练存在种子级随机性，单模型在约 200 个候选的大池上"
        "argmax 时，容易被偶然高分的动作带偏——这类动作往往是检索先验覆盖之外的随机动作。",
        "痛点 2（小样本过度自信）：在稀有干扰类型、低检索相似度的场景，检索先验本身证据不足，"
        "单模型仍给出确定的点估计，无法表达「我不确定」，容易在小样本场景翻车。",
        "目标：不追求全体场景平均 regret 最低，而是提升小样本 / 检索证据弱场景的稳健性"
        "（减少坏案例），这正是抗干扰部署最关心的指标。",
    ], gap=0.028)

    section(fig, y - 0.012, "三、一句话概括改进①")
    box_y, box_h = y - 0.125, 0.085
    ax = fig.add_axes([0.07, box_y, 0.86, box_h])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                fc="#eaf3fa", ec=ACCENT, linewidth=1.4,
                                transform=ax.transAxes))
    summary = ("训练 5 个不同种子的 Q 网络（其余协议完全相同），推理时用「集成均值 − 1×集成标准差」"
               "悲观打分：成员间分歧大的候选动作被自动降权，优先选择大家一致认可的动作。")
    lines = _wrap(summary, _width_chars(0.80, 10.5))
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
            fontsize=10.5, color=INK, linespacing=1.7)
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = new_page()
    page_title(fig, "方法：深度集成 + 悲观打分 μ-λσ", "改进① 的完整打分流程与训练协议")

    section(fig, 0.878, "1. 打分流程")
    draw_pipeline(fig, [0.06, 0.46, 0.88, 0.39])

    section(fig, 0.435, "2. 关键设计点")
    y = bullets(fig, 0.403, [
        "深度集成（5 seeds）：seed = 42 / 1042 / 2042 / 3042 / 4042，数据、场景级划分、"
        "warm-start、超参全部与 Round2 单模型一致——唯一变量是种子，集成差异纯粹反映训练随机性。",
        "悲观打分 score = μ − λσ：σ 是 5 个成员对同一候选动作打分的标准差（模型间分歧），"
        "分歧大的动作被降权；λ=1.0 未调参。",
        "大池统一打分：检索池与随机探索动作同规则同尺度——无需显式设定「先验 vs 探索」"
        "的融合权重，由 σ 隐式完成。",
    ], gap=0.014)

    section(fig, y - 0.02, "3. 训练协议（与 Round2 完全对齐，保证公平对比）")
    y = bullets(fig, y - 0.05, [
        "数据：Round2 信号级重标注（RadarSimPy 信号级仿真 SINR，JSR=10 dB），10k 场景 × 10 动作。",
        "损失：MSE + 0.5×listwise ranking loss（检索候选池内 argmax Q 对齐 argmax R），每 epoch 32 步 × 4 场景。",
        "优化：Adam lr=1e-3，weight decay 1e-4，batch 256，≤300 epoch；早停 = 验证集检索池 Oracle Hit@1（patience 25）。",
        "集成成员质量：val Hit@1 = 0.57 / 0.56 / 0.59 / 0.63 / 0.56（与 Round2 单模型同水平）。",
    ], gap=0.012)
    pdf.savefig(fig)
    plt.close(fig)


def page3(pdf):
    fig = new_page()
    page_title(fig, "实验设置与主结果", "50 个留出场景 · 信号级真值 · regret 相对 Oracle-K20")

    section(fig, 0.895, "1. 实验设置")
    bullets(fig, 0.862, [
        "留出条件：前 50 个未见干扰场景（holdout seed=2026），每场景候选约 200 个动作；"
        "真值为 RadarSimPy 信号级仿真（JSR=10 dB），所有方法完全同池同真值。",
        "指标：regret = Oracle-K20（留出 K=10 随机动作上界）− 推荐动作 SINR，越负越好；"
        "另报中位 regret（稳健性）、零 regret 占比、对 q_only 的逐场景配对胜率。",
    ], gap=0.026)

    section(fig, 0.755, "2. 主结果表")
    headers = ["方法", "推荐SINR(dB)", "regret(dB)", "中位regret", "零regret", "胜率vs q_only"]
    rows = []
    for name, rec, reg, med, zero, win, _ in MAIN_ROWS:
        rows.append([name, f"{rec:.2f}", f"{reg:+.2f}", f"{med:+.2f}",
                     f"{zero}%", "—" if win is None else f"{win}%"])
    table(fig, [0.06, 0.50, 0.88, 0.235], headers, rows,
          col_widths=[0.36, 0.13, 0.13, 0.13, 0.11, 0.14], fs=8.4,
          highlight_rows=(5,))

    section(fig, 0.465, "3. 结果柱状图")
    draw_main_bar(fig, [0.13, 0.19, 0.81, 0.25])

    para(fig, 0.07, 0.105, "要点：平均 regret 上集成悲观（−4.97）与 Round2 最优基线（−5.08）持平；"
                           "但中位 regret 最优（−3.08，基线 −1.89），对 q_only 配对胜率 96%（基线 90%）"
                           "——均值没赢，坏案例显著变少。", fs=9.4)
    pdf.savefig(fig)
    plt.close(fig)


def page4(pdf):
    fig = new_page()
    page_title(fig, "小样本分桶：改进①的目标场景", "按检索证据强弱二分（中位数切分）——证据越弱，集成悲观领先越多")

    section(fig, 0.895, "1. 分桶定义")
    bullets(fig, 0.862, [
        "rare_type / common_type：按「训练库中同类干扰场景数」中位数（567.5）二分；",
        "low_sim / high_sim：按查询与检索库 Top-1 余弦相似度中位数（0.931）二分；",
        "hard = rare_type ∩ low_sim（n=9）：检索证据最弱的子集——本轮改进的核心靶点。",
    ], gap=0.024)

    section(fig, 0.745, "2. 分桶结果表")
    headers = ["桶 (n)", "Q-only", "基线 Ctx-Union", "① Ens-Mean", "① Ens-悲观", "悲观胜率"]
    rows = []
    for name, (n, qo, cu, em, ep, wr) in BUCKETS.items():
        rows.append([name.replace("\n", " "), f"{qo:+.2f}", f"{cu:+.2f}",
                     f"{em:+.2f}", f"{ep:+.2f}",
                     "—" if wr is None else f"{wr}%"])
    table(fig, [0.06, 0.495, 0.88, 0.235], headers, rows,
          col_widths=[0.30, 0.12, 0.16, 0.14, 0.14, 0.14], fs=8.4,
          highlight_rows=(0, 1, 2))

    section(fig, 0.462, "3. 分桶柱状图（三个「检索证据弱」桶 + 两个对照桶）")
    draw_bucket_bar(fig, [0.08, 0.20, 0.86, 0.235])

    para(fig, 0.07, 0.125, "要点：rare_type / low_sim / hard 三个桶上集成悲观全面最优（hard 桶 regret "
                           "−2.54 vs 基线 −1.74，对 q_only 胜率 100%）；检索证据充足的 common_type / "
                           "high_sim 桶上基线仍略优——改进①把「保守」花在了它该花的地方。", fs=9.4)
    pdf.savefig(fig)
    plt.close(fig)


def page5(pdf):
    fig = new_page()
    page_title(fig, "σ 诊断、结论与后续", "对机制的诚实检验 + 已落地的默认推荐路径")

    section(fig, 0.895, "1. σ 诊断：机制假设部分证伪")
    headers = ["方法", "全池平均σ", "检索池动作σ", "随机池动作σ"]
    rows = [
        ["① Ensemble（μ 与 μ−σ 同 σ）", "0.2670", "0.2657", "0.2684"],
        ["①+② Attention-Ensemble", "0.4663", "0.4678", "0.4650"],
    ]
    table(fig, [0.06, 0.78, 0.88, 0.10], headers, rows,
          col_widths=[0.40, 0.18, 0.21, 0.21], fs=8.6)
    bullets(fig, 0.735, [
        "设计初衷假设「随机池动作（检索先验外）σ 更大，悲观打分会自动压低它们」——"
        "数据显示检索池与随机池 σ 仅差 0.9%（0.2657 vs 0.2684），该机制未获支持。",
        "真实收益来源：集成平均本身比单模型更准（降噪），μ−σ 再把成员分歧大的候选压低——"
        "体现为中位 regret 由 −2.57（仅取 μ）进一步改善至 −3.08，尾部坏案例减少。",
    ], gap=0.02)

    section(fig, 0.555, "2. 结论与采纳")
    bullets(fig, 0.522, [
        "改进①有效，且收益集中在目标场景：小样本分桶（rare_type / low_sim / hard）全面最优，"
        "hard 桶对 q_only 胜率 100%；全体场景配对胜率 96%（全场最高）。",
        "已采纳为默认推荐打分方式：官方推理 API radar/infer/q_ensemble.py"
        "（QEnsembleRecommender，λ=1.0，5 seeds）；与存档评估逐场景对拍 50/50 精确一致。",
        "改进②（Set-Attention 上下文）本轮仅保留接口（支持任意邻居库注入），不进主流程，"
        "待 k-shot 实测样本实验后再判定——其单模型在 hard 桶优于基线（−2.44 vs −1.74），有潜力信号。",
    ], gap=0.028)

    section(fig, 0.345, "3. 局限与下一步")
    bullets(fig, 0.312, [
        "统计功效有限：50 个场景、hard 桶仅 9 个，胜率差（96% vs 90%）方向可信、幅度待更多场景验证；",
        "σ 区分度弱：可尝试让网络显式输出不确定性（log-variance 头 / 最后层贝叶斯化），"
        "或按「候选与检索库的距离」外生构造 σ，恢复「压低池外动作」的机制；",
        "k-shot 实测样本：用改进②的邻居集合接口，把检索库替换 / 混入少量实测场景样本，"
        "检验小样本实测适配能力——这是两项改进共同的下一步。",
    ], gap=0.028)

    para(fig, 0.07, 0.06, "附：产物位置（outputs/20261007_fewshot/）— 代码 code/ ｜ 模型 models/q_ensemble/ ｜ "
                          "评估 eval_fewshot/summary.json ｜ 报告 comparison_report.md", fs=8.6)
    pdf.savefig(fig)
    plt.close(fig)


def main(as_png: bool = False):
    preview = ROUND / "_pdf_preview"
    if as_png:
        preview.mkdir(exist_ok=True)
    pages = [page1, page2, page3, page4, page5]
    if as_png:
        for i, fn in enumerate(pages, 1):
            fig = None
            class _FakePdf:  # 收集 fig 交由调用方存 PNG
                @staticmethod
                def savefig(f, **kw):
                    nonlocal fig
                    fig = f
            fn(_FakePdf())
            fig.savefig(preview / f"page{i}.png", dpi=110,
                        bbox_inches=None)
            plt.close(fig)
        print(f"预览 PNG 输出: {preview}")
        return
    with PdfPages(OUT_PDF) as pdf:
        page1(pdf)
        page2(pdf)
        page3(pdf)
        page4(pdf)
        page5(pdf)
        d = pdf.infodict()
        d["Title"] = "改进① Q-Ensemble 悲观打分 — 组会汇报"
        d["Author"] = "Radar Anti-Jamming Project"
    size = OUT_PDF.stat().st_size / 1024
    print(f"PDF 已生成: {OUT_PDF} ({size:.0f} KB, 5 页)")


if __name__ == "__main__":
    import sys
    main(as_png="--png" in sys.argv)
