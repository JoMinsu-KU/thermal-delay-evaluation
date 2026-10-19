#!/usr/bin/env python3
"""Plot seven fixed expiration schedules beside conditional cost increases."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root,out):
    with (root/'setting_summary.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    audit=json.loads((root/'extraction_audit.json').read_text(encoding='utf-8'))
    labels=['Original' if r['sequence']=='legacy' else f"R{int(r['sequence'][1])+1}B{int(r['sequence'][-1])+1}" for r in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.05),gridspec_kw={'width_ratios':[1.05,1]},layout='constrained')
    yy=np.arange(len(rows))
    for i,r in enumerate(rows):
        times=audit['settings'][r['sequence']]['deadline_s']
        axs[0].scatter(times,[i]*len(times),marker='|',s=100,c='#292929',linewidth=1.4)
    axs[0].set(yticks=yy,yticklabels=labels,xlim=(0,60),xticks=np.arange(0,61,10),xlabel='Expiration time (s)',title='(a) Saved expiration schedules')
    axs[0].invert_yaxis();axs[0].grid(axis='x',alpha=.25)
    axs[1].barh(yy-.17,[float(r['zero_increase_pct']) for r in rows],height=.30,label='Zero fallback',color='#D55E00')
    axs[1].barh(yy+.17,[float(r['hold_increase_pct']) for r in rows],height=.30,label='Hold fallback',color='#0072B2')
    axs[1].set(yticks=yy,yticklabels=labels,xlim=(0,87),xlabel='Mean FV cost increase (%)',title='(b) Cost relative to zero delay')
    axs[1].invert_yaxis();axs[1].grid(axis='x',alpha=.25);axs[1].set_axisbelow(True)
    axs[1].legend(loc='lower right',fontsize=7.2,frameon=False)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/f'deadline_schedule_cost.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path)
    args=p.parse_args();main(args.input,args.output or args.input/'figures')
