"""Standalone research figure from independently recomputed CSV results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/small_seed/paired_contrasts.csv');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/figures');args=p.parse_args()
ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
with args.input.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
seeds = [6060, 6061, 6062, 6063]
labels = ['6060 (original), alpha=0.5', '6061, alpha=1.0', '6062*, alpha=0.5', '6063, alpha=1.0']
colors = ['#69727A', '#27628E', '#AD6531', '#27628E']
for ax, mode, title in zip(axes, ['zero', 'delayed'], ['(a) Zero compute delay', '(b) Original timing replay']):
    for i, seed in enumerate(seeds):
        row = next(x for x in rows if int(x['seed']) == seed and x['mode'] == mode)
        value, low, high = [float(row[k]) * 1e4 for k in ['small_minus_shared', 'conditional_episode_ci_low', 'conditional_episode_ci_high']]
        ax.errorbar(value, i, xerr=np.array([[value - low], [high - value]]), fmt='o',
                    color=colors[i], mfc='white' if seed == 6062 else colors[i], mec=colors[i],
                    markersize=7, capsize=4, elinewidth=1.5)
        ax.text(value, i - .16, f'{float(row["small_relative_to_shared_pct"]):+.3f}%', ha='center', va='bottom', fontsize=9)
    ax.axvline(0, color='#30363D', linestyle='--', linewidth=1)
    ax.set_title(title, loc='left', fontsize=12, pad=15)
    ax.set_xlim(-2.65, 1.05); ax.set_xticks([-2, -1, 0, 1])
    ax.set_xlabel(r'Mean cost difference: Small $-$ Shared ($\times 10^{-4}$)', labelpad=10)
    ax.xaxis.grid(True, alpha=.17); ax.set_axisbelow(True)
    ax.set_ylim(3.5, -.6)
axes[0].set_yticks(range(4), labels)
axes[1].tick_params(axis='y', left=False)
fig.suptitle('Small initialization repeats against one fixed Shared checkpoint', x=.05, ha='left', fontsize=14, y=.98)
fig.text(.05, .112, 'Points: mean paired cost difference; bars: conditional 95% episode-bootstrap intervals (24 episodes, 20,000 resamples).', fontsize=9)
fig.text(.05, .076, 'Percent labels use the corresponding Shared mean as denominator. Negative values indicate lower Small cost.', fontsize=9)
fig.text(.05, .04, '* Seed 6062 failed the validation regret criterion and is retained. Intervals exclude uncertainty across training seeds and timing streams.', fontsize=8.6)
fig.subplots_adjust(left=.255, right=.985, bottom=.255, top=.835, wspace=.13)
fig.savefig(ROOT / 'small_seed_cost_comparison.png', dpi=200, facecolor='white')
fig.savefig(ROOT / 'small_seed_cost_comparison.pdf', facecolor='white')
plt.close(fig)
print('Saved small_seed_cost_comparison.png and .pdf')
