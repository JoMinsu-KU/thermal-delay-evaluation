"""Recompute Monte Carlo summaries from saved episode metrics, without simulation."""
from pathlib import Path
import csv, itertools, json, math
from collections import defaultdict
import numpy as np

P=Path(__file__).resolve().parents[1]; D=P/'data/monte_carlo'; O=P/'outputs/monte_carlo'
O.mkdir(parents=True,exist_ok=True)
def read(name):
    with (D/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
    with (O/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare(name,rows,keys):
    expected={tuple(r[k] for k in keys):r for r in read(name)}
    assert len(expected)==len(rows),(name,len(expected),len(rows))
    for r in rows:
        e=expected[tuple(str(r[k]) for k in keys)]
        for k,v in r.items():
            if k not in keys:assert math.isclose(float(v),float(e[k]),rel_tol=1e-9,abs_tol=1e-12),(name,k,v,e[k])
    save(name,rows)
rows=read('episode_metrics.csv')
assert len(rows)==9456 and len({r['key'] for r in rows})==9456
assert all(r['completed']=='True' and float(r['duration_s'])==60 and float(r['plant_max_step_ms'])==10 for r in rows)
error=max(abs(float(r['episode_cost'])-sum(float(r[k]) for k in ('J_tracking','J_terminal','J_command','J_increment'))) for r in rows)
assert error<1e-12
methods=['pi','fv_mpc','pino_mpc','shared_physics','small_physics']
TOL=1e-10
conds=[(m,'zero') for m in methods]+[('fv_mpc','hold')]
baseline={m:{r['episode']:float(r['episode_cost']) for r in rows if r['phase']=='baseline' and r['mode']=='zero' and r['method']==m} for m in methods}
episodes=set(baseline['pi']);assert len(episodes)==24 and all(set(x)==episodes for x in baseline.values())
assert sum(r['phase']=='baseline' for r in rows)==240
groups=defaultdict(dict)
for r in rows:
    if r['phase']!='baseline':groups[(int(r['replicate']),r['method'],r['fallback_policy'])][r['episode']]=r
assert set(groups)=={(i,*c) for i in range(64) for c in conds}
assert all(set(g)==episodes for g in groups.values())
means={m:float(np.mean(list(baseline[m].values()))) for m in methods}
summary=[]; costs={}
for (i,m,p),g in sorted(groups.items()):
    mean=float(np.mean([float(r['episode_cost']) for r in g.values()]));costs[i,m,p]=mean
    summary.append(dict(replicate=i,method=m,fallback_policy=p,n_episodes=24,mean_cost=mean,baseline_mean_cost=means[m],delta_cost=mean-means[m],relative_change_pct=100*(mean/means[m]-1)))
compare('replicate_summary.csv',summary,['replicate','method','fallback_policy'])
dist=[]
for m,p in conds:
    for metric in ['mean_cost','delta_cost','relative_change_pct']:
        x=np.array([r[metric] for r in summary if r['method']==m and r['fallback_policy']==p])
        dist.append(dict(method=m,fallback_policy=p,metric=metric,replicates=64,mean=x.mean(),sd=x.std(ddof=1),min=x.min(),q025=np.quantile(x,.025),median=np.median(x),q975=np.quantile(x,.975),max=x.max()))
compare('cost_effect_distribution.csv',dist,['method','fallback_policy','metric'])
pairs=[]
for i in range(64):
    z=groups[i,'fv_mpc','zero'];h=groups[i,'fv_mpc','hold']
    delta=np.array([float(h[e]['episode_cost'])-float(z[e]['episode_cost']) for e in sorted(episodes)])
    pairs.append(dict(replicate=i,mean_hold_minus_zero=delta.mean(),relative_hold_vs_zero_pct=100*(costs[i,'fv_mpc','hold']/costs[i,'fv_mpc','zero']-1),hold_wins=int((delta < -TOL).sum()),ties=int((abs(delta)<=TOL).sum()),hold_losses=int((delta>TOL).sum())))
compare('paired_fv_effects.csv',pairs,['replicate'])
def wilson(k,n=64):
    z=1.959963984540054;q=k/n;den=1+z*z/n
    center=(q+z*z/(2*n))/den;half=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)
prob=[]
for policy in ['zero','hold']:
    events=defaultdict(list)
    for i in range(64):
        c={m:costs[i,m,policy if m=='fv_mpc' else 'zero'] for m in methods}
        events['FV_strictly_highest'].append(all(c['fv_mpc']>c[m]+TOL for m in methods if m!='fv_mpc'))
        events['FV_rank_worse_than_zero_delay'].append(sum(v<c['fv_mpc']-TOL for v in c.values())>sum(v<means['fv_mpc']-TOL for v in means.values()))
        events['Shared_strictly_worse_than_Small'].append(c['shared_physics']>c['small_physics']+TOL)
        reversals=[]
        for a,b in itertools.combinations(methods,2):
            flag=abs(c[a]-c[b])>TOL and abs(means[a]-means[b])>TOL and (c[a]-c[b])*(means[a]-means[b])<0
            events[f'pair_reversal_{a}_vs_{b}'].append(flag);reversals.append(flag)
        events['any_strict_pair_reversal'].append(any(reversals))
        for m in methods:events[f'{m}_rank1_including_ties'].append(c[m]<=min(c.values())+TOL)
    for event,flags in events.items():
        k=sum(flags);lo,hi=wilson(k)
        prob.append(dict(fv_fallback=policy,event=event,successes=k,replicates=64,proportion=k/64,wilson95_low=lo,wilson95_high=hi))
k=sum(r['mean_hold_minus_zero'] < -TOL for r in pairs);lo,hi=wilson(k)
prob.append(dict(fv_fallback='paired',event='FV_hold_lower_cost',successes=k,replicates=64,proportion=k/64,wilson95_low=lo,wilson95_high=hi))
compare('ranking_probabilities.csv',prob,['fv_fallback','event'])
counts=[]
for m,p in conds:
    rs=[r for (i,mm,pp),g in groups.items() if (mm,pp)==(m,p) for r in g.values()]
    counts.append(dict(method=m,fallback_policy=p,runs=len(rs),missed_deadlines=sum(int(r['missed_deadlines']) for r in rs),thermal_violation_runs=sum(r['violation']=='True' for r in rs)))
save('completion_deadline_counts.csv',counts)
checks=dict(status='PASS',saved_cases=9456,baseline_cases=240,new_cases=9216,replicates=64,unique_nominal_episodes=24,paired_fv_cases=1536,max_saved_component_error=error,hold_episode_wins=sum(r['hold_wins'] for r in pairs),hold_episode_ties=sum(r['ties'] for r in pairs),hold_episode_losses=sum(r['hold_losses'] for r in pairs),scope='Saved-metric arithmetic only. Raw quadrature, request schedules, sampled draw plans and source seals are covered by author audit receipts, not re-audited here. Inference-failure counters are inactive and not interpreted.',percentile_interpretation='Empirical replicate spread, not a confidence interval for the mean; Wilson intervals concern conditional replicate events.')
(O/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print('PASS: 9,456 saved costs, 384 group means, 18 distributions, 39 rank events, and 64 paired effects.')
