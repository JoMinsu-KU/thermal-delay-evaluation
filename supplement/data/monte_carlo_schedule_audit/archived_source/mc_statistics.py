"""Conditional replicate-level Monte Carlo summaries; no episode bootstrap."""
import math
import numpy as np
from plan import METHODS
from validation_io import require
TOL = 1e-10


def wilson(successes, n, z=1.959963984540054):
    require(n > 0 and 0 <= successes <= n, 'Invalid binomial count')
    p = successes/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'successes':int(successes), 'replicates':int(n), 'proportion':p,
            'wilson95_low':max(0.,center-half), 'wilson95_high':min(1.,center+half)}


def rank_summary(means):
    require(set(means)==set(METHODS), 'Complete five-method comparison required')
    order = sorted(METHODS,key=lambda m:(means[m],METHODS.index(m)))
    return dict(ascending_mean_cost=order,
                ranks={m:1+sum(means[v]<means[m]-TOL for v in METHODS) for m in METHODS},
                ties=[[a,b] for i,a in enumerate(METHODS) for b in METHODS[i+1:] if abs(means[a]-means[b])<=TOL],
                means=means)


def summarize(records, episodes, replicates):
    """records keys: (sequence, policy, mode, method, episode) -> metrics."""
    def vals(seq,policy,mode,method,field='episode_cost'):
        keys=[(seq,policy,mode,method,e) for e in episodes]
        return np.array([records[k][field] for k in keys],dtype=float) if all(k in records for k in keys) else None
    zero = {m:vals('legacy','zero','zero',m) for m in METHODS}
    if not all(v is not None for v in zero.values()):
        return {'rankings':[], 'probabilities':[], 'cells':[], 'effects':[], 'distributions':[], 'complete_replicates':0}
    baseline = rank_summary({m:float(v.mean()) for m,v in zero.items()})
    rankings=[];cells=[];effects=[];events={};draws={};completed=0
    for rep in range(replicates):
        seq=f'mc{rep:04d}'
        arrays={m:vals(seq,'zero','delayed',m) for m in METHODS}
        hold=vals(seq,'hold','delayed','fv_mpc')
        if hold is None or not all(v is not None for v in arrays.values()):
            continue
        completed+=1
        for m in METHODS:
            for policy in (('zero','hold') if m=='fv_mpc' else ('zero',)):
                x=hold if policy=='hold' else arrays[m]
                mean=float(x.mean());base=float(zero[m].mean());effect=mean-base
                ratio=(mean/base-1)*100
                cells.append(dict(replicate=rep,method=m,fallback_policy=policy,n_episodes=len(episodes),mean_cost=mean,baseline_mean_cost=base,delta_cost=effect,relative_change_pct=ratio))
                draws.setdefault((m,policy),[]).append((mean,effect,ratio))
        for policy in ('zero','hold'):
            means={m:float(v.mean()) for m,v in arrays.items()}
            if policy=='hold':means['fv_mpc']=float(hold.mean())
            ranking=rank_summary(means)
            rankings.append(dict(replicate=rep,fv_fallback=policy,**ranking))
            def collect(event,value):events.setdefault((policy,event),[]).append(bool(value))
            collect('FV_strictly_highest',all(means['fv_mpc']>means[m]+TOL for m in METHODS if m!='fv_mpc'))
            collect('FV_rank_worse_than_zero_delay',ranking['ranks']['fv_mpc']>baseline['ranks']['fv_mpc'])
            collect('Shared_strictly_worse_than_Small',means['shared_physics']>means['small_physics']+TOL)
            reversals=[]
            for i,a in enumerate(METHODS):
                for b in METHODS[i+1:]:
                    before=baseline['means'][a]-baseline['means'][b];after=means[a]-means[b]
                    rev=abs(before)>TOL and abs(after)>TOL and before*after<0
                    collect(f'pair_reversal_{a}_vs_{b}',rev);reversals.append(rev)
            collect('any_strict_pair_reversal',any(reversals))
            for m in METHODS:collect(m+'_rank1_including_ties',ranking['ranks'][m]==1)
        a=arrays['fv_mpc'];delta=hold-a
        effects.append(dict(replicate=rep,mean_hold_minus_zero=float(delta.mean()),relative_hold_vs_zero_pct=float((hold.mean()/a.mean()-1)*100),hold_wins=int((delta < -TOL).sum()),ties=int((abs(delta)<=TOL).sum()),hold_losses=int((delta>TOL).sum())))
        events.setdefault(('paired','FV_hold_lower_cost'),[]).append(float(delta.mean()) < -TOL)
    probs=[dict(fv_fallback=p,event=e,**wilson(sum(v),len(v))) for (p,e),v in events.items()]
    distributions=[]
    for (method,policy),data in draws.items():
        matrix=np.asarray(data)
        for j,metric in enumerate(('mean_cost','delta_cost','relative_change_pct')):
            v=matrix[:,j];quant=np.quantile(v,[.025,.5,.975])
            distributions.append(dict(method=method,fallback_policy=policy,metric=metric,replicates=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else None,min=float(v.min()),q025=float(quant[0]),median=float(quant[1]),q975=float(quant[2]),max=float(v.max())))
    return dict(baseline=baseline,rankings=rankings,probabilities=probs,cells=cells,effects=effects,distributions=distributions,complete_replicates=completed)
