from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]
RES = BASE / 'results'
OUT = BASE / 'paper_assets'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.unicode_minus': False})

true = pd.read_csv(RES / 'true_regret_fresh250.csv')
labels = ['Q-only', 'Retrieve+Q', 'Union', 'Ens-Mean', 'Ens-Pess.']
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = np.arange(len(true))
bars = ax.bar(x, true.mean_true_regret, color=['#7F7F7F','#ED7D31','#1F4E79','#548235','#70AD47'])
ax.set_xticks(x, labels); ax.set_ylabel('Mean true regret (dB)'); ax.set_title('True regret on 250 fresh scenarios')
ax.grid(axis='y', alpha=.25); ax.set_axisbelow(True)
for b, v in zip(bars, true.mean_true_regret): ax.text(b.get_x()+b.get_width()/2, v+.25, f'{v:.2f}', ha='center')
fig.tight_layout(); fig.savefig(OUT / 'fig_true_regret.png', dpi=220); plt.close(fig)

sweep = pd.read_csv(RES / 'sweeps' / 'summary.csv')
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
for ax, kind, col, title, xlabel, colname in [
    (axs[0], 'top_k', '#1F4E79', 'Retrieval depth sweep', 'Top-K', 'top_k'),
    (axs[1], 'pool', '#ED7D31', 'Exploration pool sweep', 'Random candidate count', 'pool')]:
    d = sweep[sweep.sweep == kind].sort_values(colname)
    ax.plot(d[colname], d['fresh450.mean'], marker='o', color=col, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean regret vs K10 (dB)'); ax.set_title(title); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'fig_candidate_sweeps.png', dpi=220); plt.close(fig)

rob = pd.read_csv(RES / 'robust_formal' / 'summary_by_radius.csv')
fig, ax1 = plt.subplots(figsize=(8.0, 4.2)); x = rob.rho * 100
ax1.plot(x, rob.mean_worst_degradation, marker='o', color='#1F4E79', label='Mean worst-case degradation')
ax1.plot(x, rob.p95_worst_degradation, marker='s', color='#ED7D31', label='p95 worst-case degradation')
ax1.set_xlabel('Perturbation radius (%)'); ax1.set_ylabel('Degradation (dB)'); ax1.grid(alpha=.25)
ax2 = ax1.twinx(); ax2.plot(x, rob.any_flip_rate*100, marker='^', color='#548235', label='Any decision flip rate'); ax2.set_ylabel('Decision flip rate (%)')
lines = ax1.get_lines() + ax2.get_lines(); ax1.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
ax1.set_title('Robustness under jammer-parameter perturbations'); fig.tight_layout(); fig.savefig(OUT / 'fig_robustness.png', dpi=220); plt.close(fig)

jsr = sweep[sweep.sweep == 'jsr'].sort_values('jsr_db')
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(jsr.jsr_db, jsr['fresh450.mean'], marker='o', color='#1F4E79', linewidth=2, label='Mean regret')
ax.plot(jsr.jsr_db, jsr['fresh450.p95'], marker='s', color='#ED7D31', linewidth=2, label='p95 regret')
ax.set_xlabel('JSR (dB)'); ax.set_ylabel('Regret vs K10 (dB)'); ax.set_title('Cross-condition evaluation with frozen policy'); ax.grid(alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(OUT / 'fig_jsr.png', dpi=220); plt.close(fig)
