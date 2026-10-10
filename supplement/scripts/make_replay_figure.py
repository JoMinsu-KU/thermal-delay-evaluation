"""Plot independently audited saved replay/fallback contrasts only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
R=P/'data/replay'
F=P/'outputs/figures';F.mkdir(parents=True,exist_ok=True)
rows=json.loads((R/'replay_results.json').read_text(encoding='utf-8'))['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.3,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(3,1,figsize=(3.15,5.45),layout='constrained')
items=[('FV_zero_change_pct','FV_zero_ci','(a) FV: zero fallback','#D55E00'),
 ('FV_hold_change_pct','FV_hold_ci','(b) FV: hold fallback','#0072B2'),
 ('Shared_minus_Small_pct','Shared_minus_Small_ci','(c) Shared - Small under replay','#009E73')]
for ax,(key,ci,title,color) in zip(axes,items):
 mean=np.array([r[key] for r in rows]);bounds=np.array([r[ci] for r in rows])
 ax.errorbar(np.arange(7),mean,yerr=np.stack([mean-bounds[:,0],bounds[:,1]-mean]),fmt='o',color=color,
             markersize=3.4,elinewidth=.85,capsize=2)
 ax.axhline(0,color='.4',lw=.6);ax.axvline(.5,color='.75',lw=.6,ls=':')
 ax.set_xticks(np.arange(7),[r['label'] if r['sequence']!='legacy' else 'Orig.' for r in rows],rotation=35,ha='right')
 ax.set_xlim(-.5,6.5);ax.set_title(title,loc='left');ax.set_ylabel('Mean-cost change (%)');ax.grid(axis='y',alpha=.14)
axes[0].set_ylim(-3,100);axes[1].set_ylim(-.3,10);axes[2].set_ylim(-.03,.85)
for ext in ('pdf','png'):fig.savefig(F/f'replay_fallback_sensitivity.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
print('Wrote replay_fallback_sensitivity')
