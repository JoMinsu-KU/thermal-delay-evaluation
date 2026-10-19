"""Plot all 64 conditional Monte Carlo repetitions; no smoothing or inferred CI."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];D=P/'data/monte_carlo';O=P/'outputs/figures';O.mkdir(parents=True,exist_ok=True)
with (D/'replicate_summary.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
def values(policy,key):return np.array([float(r[key]) for r in sorted(rows,key=lambda r:int(r['replicate'])) if r['method']=='fv_mpc' and r['fallback_policy']==policy])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),layout='constrained');x=np.arange(1,65)
for p,label,color,marker in [('zero','Zero fallback','#c66a23','o'),('hold','Hold','#287399','s')]:
    axs[0].scatter(x,values(p,'relative_change_pct'),label=label,c=color,s=15,marker=marker,linewidths=0)
axs[0].set(ylabel='Cost increase from zero delay (%)',ylim=(-1,58),title='(a) FV-MPC mean cost increase')
axs[0].legend(frameon=False,fontsize=8,loc='upper left')
reduction=100*(1-values('hold','mean_cost')/values('zero','mean_cost'))
axs[1].scatter(x,reduction,s=16,c='#315b55',linewidths=0);axs[1].axhline(0,c='#777777',lw=.7)
axs[1].set(ylabel='Cost reduction under hold (%)',ylim=(-1,36),title='(b) Paired policy effect')
for ax in axs:ax.set(xlabel='Monte Carlo repetition',xlim=(0,65),xticks=[1,16,32,48,64]);ax.grid(axis='y',alpha=.2)
fig.savefig(O/'monte_carlo_policy_effect.pdf',metadata={'Author':'','CreationDate':None,'ModDate':None})
fig.savefig(O/'monte_carlo_policy_effect.png',dpi=220)
plt.close(fig)
print('PASS: Monte Carlo policy figure, all 64 repetitions.')
