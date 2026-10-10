"""Portable figures from compact saved analysis inputs; no experiment execution."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P=Path(__file__).resolve().parents[1]
R=P/'data/revision'
M=P/'data/minor_revision'
F=P/'outputs/figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 for ext in ('pdf','png'):fig.savefig(F/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

rows=list(csv.DictReader((R/'cost_components.csv').open(encoding='utf-8')))
episodes=sorted({r['episode'] for r in rows})
cost={(r['episode'],r['method'],r['mode']):float(r['episode_cost']) for r in rows}
methods=['pi','fv_mpc','pino_mpc','shared_physics','small_physics'];labels=['PI','FV','PINO','Shared','Small']
fig,axes=plt.subplots(3,1,figsize=(3.15,5.55),layout='constrained')
for ax,subset,title,scale in [(axes[0],methods,'(a) Delay effects: all five systems',1e3),(axes[1],[m for m in methods if m!='fv_mpc'],'(b) Expanded view: non-FV systems',1e5)]:
 for k,m in enumerate(subset):
  v=np.array([cost[e,m,'delayed']-cost[e,m,'zero'] for e in episodes])*scale
  ax.scatter(k+(np.arange(24)%7-3)*.03,v,s=9,color='#0072B2',alpha=.65)
  ax.plot([k-.22,k+.22],[v.mean()]*2,color='#D55E00',lw=1.6)
 ax.set_xticks(range(len(subset)),[labels[methods.index(m)] for m in subset]);ax.axhline(0,color='.5',lw=.6)
 ax.set_ylabel('Replay - zero cost\n'+('$10^{-3}$' if scale==1e3 else '$10^{-5}$'));ax.set_title(title,loc='left')
for e in episodes:
 v=[(cost[e,'shared_physics',mode]-cost[e,'small_physics',mode])*1e3 for mode in ('zero','delayed')]
 axes[2].plot([0,1],v,color='#879ead',lw=.65,alpha=.75);axes[2].scatter([0,1],v,color=['#0072B2','#D55E00'],s=10)
axes[2].axhline(0,color='.5',ls='--',lw=.6);axes[2].set_xticks([0,1],['Zero delay','Measured replay']);axes[2].set_xlim(-.15,1.15)
axes[2].set_ylabel('Shared - Small cost\n($10^{-3}$)');axes[2].set_title('(c) Paired residual-system comparison',loc='left')
save(fig,'paired_test_results')

info=json.loads((R/'physical_results.json').read_text(encoding='utf-8'))
temp=list(csv.DictReader((R/'representative_temperature.csv').open(encoding='utf-8')))
temp+=list(csv.DictReader((M/'hold_temperature.csv').open(encoding='utf-8')))
def curve(method,mode,field):return np.array([float(r[field]) for r in temp if r['method']==method and r['mode']==mode])
time=curve('fv_mpc','zero','time_ns')/1e9
fig,axes=plt.subplots(2,1,figsize=(3.15,4.3),layout='constrained')
for mode,color,label in [('zero','#0072B2','Zero delay'),('delayed','#D55E00','Replay (zero fallback)'),('hold','#009E73','Replay (hold fallback)')]:
 axes[0].plot(time,curve('fv_mpc',mode,'mean_temperature_c'),color=color,label=label,lw=1.1)
 axes[0].plot(time,curve('fv_mpc',mode,'max_temperature_c'),color=color,ls='--',lw=.9)
 if mode!='hold':
  axes[1].plot(time,curve('shared_physics',mode,'mean_temperature_c')-curve('small_physics',mode,'mean_temperature_c'),color=color,label=label,lw=1)
axes[0].plot(time,curve('fv_mpc','zero','goal_c'),color='.3',ls=':',lw=1,label='Target')
for start,end in info['FV_deadline_fallback_intervals_s']:axes[0].axvspan(start,end,color='#777777',alpha=.22,lw=0)
axes[0].set_ylabel(r'Temperature ($^\circ$C)');axes[0].set_title('(a) FV-MPC: solid mean, dashed peak',loc='left');axes[0].legend(loc='lower right',frameon=False,fontsize=6.5)
axes[1].axhline(0,color='.5',ls=':',lw=.6);axes[1].set_ylabel('Shared - Small\nmean temperature (K)');axes[1].legend(frameon=False)
axes[1].set_title('(b) Shared - Small mean temperature',loc='left')
for ax in axes:ax.set_xlabel('Time (s)');ax.set_xlim(0,60);ax.grid(axis='y',alpha=.15)
save(fig,'temperature_response')

fig,ax=plt.subplots(figsize=(6.65,2.7));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,color='#edf3f7',size=8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035',facecolor=color,edgecolor='#526678',lw=.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.9,'color':'#526678'})
box(.1,2.65,2.2,1.1,'Numerical inputs\n4 modality encoders\n22 x 128 features',size=7.3)
box(2.8,2.65,3,1.1,'Shared: Qwen + LoRA\nInput + task + summary\nSmall: 3-layer Transformer',size=7.3)
box(6.35,2.65,1.5,1.1,'Summary\nProjection\n256-d context',size=7.3)
arrow((2.35,3.2),(2.75,3.2));arrow((5.85,3.2),(6.3,3.2))
box(.1,.85,2.2,1.05,'9 candidates\n20 x 4 commands\n32 physics features',size=7.7)
box(2.8,.85,3,1.05,'Action: 80 to 64\nPhysics: 32 to 64\nOne shared 256-d state',size=7.5)
box(6.35,.85,1.5,1.05,'Concat: 384\nMLP: 128 to 1\nScalar output',size=7.3)
box(8.35,.85,1.55,1.05,'PINO correction\nFeasibility gate\nCommand',size=7.3)
arrow((2.35,1.38),(2.75,1.38));arrow((5.85,1.38),(6.3,1.38));arrow((7.9,1.38),(8.3,1.38));arrow((7.1,2.6),(7.1,1.95))
ax.text(5,.22,'One state per request, shared by all nine candidates. No autoregressive generation.',ha='center',fontsize=8)
save(fig,'residual_architecture')
# Schematic: present evaluation and execution roles without claiming a new algorithm.
fig, ax = plt.subplots(figsize=(3.15,3.65))
ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
boxes=[(.91,'Observation and command history','Causal sensors, frames, and process context'),
       (.735,'Nine candidate heater commands','Hold, or change one heater by +/-0.04'),
       (.56,'Learned score and feasibility gate','PINO screening; zero-command fallback'),
       (.385,'Asynchronous command execution','Path-matched delay; 100-ms deadline\nOn expiration: zero or hold command'),
       (.21,'Finite-volume thermal plant','60 s trajectory; state evolves during delay')]
for y,title,sub in boxes:
    ax.add_patch(FancyBboxPatch((.055,y-.068),.89,.13,boxstyle='round,pad=.006,rounding_size=.015',
                 facecolor='#f4f7fa',edgecolor='#6e8293',linewidth=.7))
    ax.text(.5,y+.036,title,ha='center',va='center',fontsize=7.8,fontweight='bold')
    ax.text(.5,y-.025,sub,ha='center',va='center',fontsize=6.7)
for a,b in zip(boxes,boxes[1:]):
    ax.annotate('',xy=(.5,b[0]+.07),xytext=(.5,a[0]-.075),arrowprops={'arrowstyle':'->','lw':.8})
ax.annotate('',xy=(.035,.91),xytext=(.035,.21),arrowprops={'arrowstyle':'->','lw':.75,
            'connectionstyle':'bar,angle=180,fraction=-0.02'})
ax.text(.5,.05,'Readouts: FV cost difference / trajectory cost / events',ha='center',fontsize=7.0)
save(fig,'evaluation_flow')

print('Wrote four revised figures')
