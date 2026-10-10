"""Check saved command increments and plot the first expiration (Supplement S1, Fig. S2)."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
R=P/'data/minor_revision_r2'
F=P/'outputs/figures';F.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((R/'deadline_commands.csv').open(encoding='utf-8')))
info=json.loads((R/'deadline_command_evidence.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(6.65,4.4),sharex=True,sharey=True,layout='constrained')
start,end=info['plot_window_s'];checks={}
for policy,color,style,label in [('zero','#D55E00','-','Zero fallback'),('hold','#0072B2','--','Hold fallback')]:
 rs=[r for r in rows if r['policy']==policy];assert len(rs)==592
 ts=np.array([int(r['time_ns'])/1e9 for r in rs])
 us=np.array([[float(r[f'u{i}']) for i in range(4)] for r in rs])
 delta=np.diff(np.vstack([np.zeros(4),us]),axis=0)
 computed=float(.02/600*np.sum(np.mean(delta**2,axis=1)))
 assert np.isclose(computed,info['policies'][policy]['reconstructed_J_increment'],atol=1e-14,rtol=0)
 first=next(r for r in rs if r['event']=='deadline_fallback');assert int(first['time_ns'])==6_800_000_000
 checks[policy]={'nominal_slots':600,'stored_application_events':592,'J_increment':computed}
 for i,ax in enumerate(axes.flat):
  ax.step(np.r_[ts,60.],np.r_[us[:,i],us[-1,i]],where='post',color=color,ls=style,lw=1.6,label=label)
for i,ax in enumerate(axes.flat):
 for deadline in [6.8,7.0]:ax.axvline(deadline,color='.4',ls=':',lw=.8)
 ax.axvspan(6.8,7.177863943,color='.6',alpha=.14,lw=0)
 ax.set(xlim=(start,end),ylim=(-.025,1.025),title=f'({chr(97+i)}) Heater {i+1}')
 ax.set_xticks([6.5,7,7.5,8,8.5,9]);ax.grid(axis='y',alpha=.18)
 if i>=2:ax.set_xlabel('Time (s)')
 if i%2==0:ax.set_ylabel('Applied command')
axes[0,0].legend(loc='center right',frameon=False)
fig.suptitle('FV-MPC commands around the first deadline miss',fontsize=11)
for ext in ['pdf','png']:fig.savefig(F/f'deadline_command_response.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
print('PASS: fixed N=600 command penalties; wrote supplementary deadline_command_response')
