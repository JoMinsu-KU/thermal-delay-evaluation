"""Regenerate the four paper figures using public relative paths (MIT)."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def render(root, out):
    F=out/'figures'; F.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('pdf','png'):
            fig.savefig(F/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.035)
        plt.close(fig)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':8.5,'xtick.labelsize':7.2,'ytick.labelsize':7.2,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'axes.linewidth':.6,'lines.linewidth':1.,'legend.fontsize':7})
    # Schematic: present evaluation and execution roles without claiming a new algorithm.
    fig, ax = plt.subplots(figsize=(3.15,3.65))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    boxes=[(.91,'Observation and command history','Causal sensors, frames, and process context'),
           (.735,'Nine candidate heater commands','Hold, or change one heater by +/-0.04'),
           (.56,'Learned score and feasibility gate','PINO screening; zero-command fallback'),
           (.385,'Asynchronous command execution','Path-matched delay; 100 ms deadline'),
           (.21,'Finite-volume thermal plant','60 s trajectory; state evolves during delay')]
    for y,title,sub in boxes:
        ax.add_patch(FancyBboxPatch((.055,y-.068),.89,.13,boxstyle='round,pad=.006,rounding_size=.015',
                     facecolor='#f4f7fa',edgecolor='#6e8293',linewidth=.7))
        ax.text(.5,y+.018,title,ha='center',va='center',fontsize=7.8,fontweight='bold')
        ax.text(.5,y-.024,sub,ha='center',va='center',fontsize=7)
    for a,b in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(.5,b[0]+.07),xytext=(.5,a[0]-.075),arrowprops={'arrowstyle':'->','lw':.8})
    ax.annotate('',xy=(.035,.91),xytext=(.035,.21),arrowprops={'arrowstyle':'->','lw':.75,
                'connectionstyle':'bar,angle=180,fraction=-0.02'})
    ax.text(.5,.05,'Readouts: teacher regret / trajectory cost / events',ha='center',fontsize=7.4)
    save(fig,'evaluation_flow')


    rows=list(csv.DictReader((out/'paired_episode_differences.csv').open(encoding='utf-8')))
    fig,axes=plt.subplots(2,1,figsize=(3.15,4.2),layout='constrained')
    ax=axes[0]
    names=['pi','fv_mpc','pino_mpc','shared_physics','small_physics']
    labels=['PI','FV','PINO','Shared','Small']
    for k,m in enumerate(names):
     vals=np.array([float(r['difference']) for r in rows if r['comparison']==m+': replay-zero'])*1000
     offsets=(np.arange(len(vals))%7-3)*.033
     ax.scatter(k+offsets,vals,s=11,color='#23679b',alpha=.7)
     ax.plot([k-.23,k+.23],[vals.mean()]*2,color='#c06424',lw=1.8)
    ax.axhline(0,color='#555555',lw=.6);ax.set_xticks(range(5),labels)
    ax.set_ylabel('Replay minus zero cost ($10^{-3}$)')
    ax.set_title('(a) Paired delay effects: 24 episodes',loc='left')
    ax=axes[1]
    for i,e in enumerate(sorted({r['episode'] for r in rows})):
     vals=[float(next(r['difference'] for r in rows if r['episode']==e and r['comparison']==mode+': shared-small_physics'))*1000 for mode in ('zero','replay')]
     ax.plot([0,1],vals,color='#94a9b9',alpha=.8,lw=.65)
     ax.scatter([0,1],vals,color=['#23679b','#d47730'],s=13,zorder=3)
    ax.axhline(0,color='#555555',lw=.7,ls='--');ax.set_xlim(-.2,1.2)
    ax.set_xticks([0,1],['Zero delay','Measured replay'])
    ax.set_ylabel('Shared minus Small cost ($10^{-3}$)')
    ax.set_title('(b) Complete residual-system comparison',loc='left')
    save(fig,'paired_test_results')



    rows=list(csv.DictReader((root/'data/selected_C/applied_commands.csv').open(encoding='utf-8')))
    traces={d:[x for x in rows if float(x['delay_ms'])==d] for d in (13,38,50)}
    fig,axes=plt.subplots(2,2,figsize=(3.15,3.05),layout='constrained')
    for ax,heater,span,pair in zip(axes.flat,[1,3,2,4],[(1.15,1.65)]*2+[(2.45,2.95)]*2,[(13,38)]*2+[(50,38)]*2):
     for d,color,ls in zip(pair,['#d47730','#23679b'],['-','--']):
      ts=np.array([0.]+[float(x['time_s']) for x in traces[d]])
      ys=np.array([0.]+[float(x[f'heater_{heater}']) for x in traces[d]])
      ax.step(ts,ys,where='post',color=color,ls=ls,label=f'{d} ms',lw=1.2)
     ax.set_xlim(*span)
     vals=[float(x[f'heater_{heater}']) for d in pair for x in traces[d] if span[0]-.1 <= float(x['time_s']) <= span[1]+.1]
     ax.set_ylim(min(vals)-.025,max(vals)+.025)
     ax.axvline(1.4 if heater in (1,3) else 2.7,color='#888888',lw=.6,ls=':')
     ax.set_title(f'Heater {heater}',loc='left');ax.set_xlabel('Application time (s)')
     if heater in (1,2):ax.set_ylabel('Applied command')
     ax.legend(loc='upper left',fontsize=6.2,handlelength=1.3,frameon=False)
    save(fig,'first_command_divergence')

    a=json.loads((root/'data/numerical/independent_audit.json').read_text(encoding='utf-8'))
    c=json.loads((root/'data/numerical/initial_C_audit.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
                         'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    fig,axs=plt.subplots(2,1,figsize=(3.22,4.65),layout='constrained')
    colors=['#0072B2','#D55E00','#009E73']
    markers=['o','s','^']
    for delay,color,marker in zip((13.,38.,50.),colors,markers):
        rows=[next(r for r in a['fixed_replay'] if r['delay_ms']==delay and r['coarse_ms']==coarse and r['fine_ms']==fine)
              for coarse,fine in ((10.,5.),(5.,2.5),(2.5,1.25))]
        axs[0].plot(range(3),[r['field_rmse_k_1hz'] for r in rows],marker=marker,ms=3.5,lw=1.2,color=color,label=f'{delay:g} ms')
        costs=[next(r['episode_cost'] for r in c['case_metrics'] if r['delay_ms']==delay and r['step_ms']==h) for h in (10.,5.,2.5)]
        axs[1].plot(range(3),costs,marker=marker,ms=3.5,lw=1.2,color=color,label=f'{delay:g} ms')
    axs[0].set_yscale('log');axs[0].set_xticks(range(3),['10 / 5','5 / 2.5','2.5 / 1.25'])
    axs[0].set_xlabel('Compared maximum plant steps (ms)')
    axs[0].set_ylabel('Field RMSE (K)')
    axs[0].set_title('(a) Fixed applied-command sequences',loc='left',pad=7)
    axs[0].legend(loc='upper right',frameon=False,ncol=1)
    axs[1].set_xticks(range(3),['10','5','2.5']);axs[1].set_xlabel('Maximum plant integration step (ms)')
    axs[1].set_ylabel('Episode cost');axs[1].set_title('(b) Recomputed closed-loop decisions',loc='left',pad=7)
    axs[1].legend(loc='center',bbox_to_anchor=(.52,.49),frameon=False,ncol=1)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    for ext in ('pdf','png'):fig.savefig(F/('numerical_case_C.'+ext))
    plt.close(fig)
