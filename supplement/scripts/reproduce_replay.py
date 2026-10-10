"""Portable reanalysis of 1,128 compact saved cases; no inference/simulation."""
from pathlib import Path
import csv,json
import numpy as np
P=Path(__file__).resolve().parents[1];D=P/'data/replay';O=P/'outputs';O.mkdir(exist_ok=True)
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def rows(name):return list(csv.DictReader((D/name).open(encoding='utf-8')))
protocol=read('protocol.json')['design'];episodes=protocol['normal_episodes']
assert len(set(episodes))==24
records=rows('episode_metrics.csv');assert len(records)==1128
by={(r['sequence'],r['fallback_policy'],r['mode'],r['method'],r['episode']):r for r in records};assert len(by)==1128
for r in records:
 assert np.isclose(sum(float(r[k]) for k in ('J_tracking','J_terminal','J_command','J_increment')),float(r['episode_cost']),atol=1e-12,rtol=1e-10)
 assert r['new_violation']=='False' and r['inference_failures']=='0' and r['delayed_slew_rejections']=='0'
def vector(seq,policy,mode,method):return np.array([float(by[seq,policy,mode,method,e]['episode_cost']) for e in episodes])
idx=np.random.default_rng(20261008).integers(0,24,(20000,24))
verified=[]
for expected in rows('paired_contrasts.csv'):
 seq=expected['sequence'];pol=expected['fallback_policy'];c=expected['comparison']
 if c=='Shared_minus_Small_zero_delay':a=vector('legacy','zero','zero','small_physics');b=vector('legacy','zero','zero','shared_physics')
 elif c=='Shared_minus_Small_replay':a=vector(seq,'zero','delayed','small_physics');b=vector(seq,'zero','delayed','shared_physics')
 elif c=='FV_hold_minus_zero_fallback':a=vector(seq,'zero','delayed','fv_mpc');b=vector(seq,'hold','delayed','fv_mpc')
 else:
  method=c.removesuffix('_replay_minus_zero');a=vector('legacy','zero','zero',method);b=vector(seq,pol,'delayed',method)
 d=b-a;ci=np.quantile(d[idx].mean(1),[.025,.975]);rci=np.quantile((b[idx].mean(1)/a[idx].mean(1)-1)*100,[.025,.975])
 result={'mean_A':float(a.mean()),'mean_B':float(b.mean()),'mean_B_minus_A':float(d.mean()),'difference_ci_low':float(ci[0]),'difference_ci_high':float(ci[1]),
  'ratio_of_means_pct':float((b.mean()/a.mean()-1)*100),'ratio_ci_low':float(rci[0]),'ratio_ci_high':float(rci[1]),
  'B_wins':int((d < -1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'B_losses':int((d>1e-10).sum())}
 for k,v in result.items():assert np.isclose(v,float(expected[k]),atol=1e-12,rtol=1e-10),(seq,c,k)
 verified.append({'sequence':seq,'comparison':c,'fallback_policy':pol,**result})
methods=protocol['jobs'][0:10:2];methodnames=[x['method'] for x in methods];assert len(methodnames)==5
for rank in read('method_rankings.json'):
 seq=rank['sequence'];means={m:vector(seq,'zero','delayed',m).mean() for m in methodnames}
 assert sorted(means,key=means.get)==rank['ascending_mean_cost']
 means['fv_mpc']=vector(seq,'hold','delayed','fv_mpc').mean()
 assert sorted(means,key=means.get)==rank['ranking_with_FV_hold']
figdata=read('replay_results.json')['rows']
for r in figdata:
 for prefix,c,pol in [('FV_zero','fv_mpc_replay_minus_zero','zero'),('FV_hold','fv_mpc_replay_minus_zero','hold'),('Shared_minus_Small','Shared_minus_Small_replay','zero')]:
  x=next(x for x in verified if x['sequence']==r['sequence'] and x['comparison']==c and x['fallback_policy']==pol)
  k=prefix+('_change_pct' if prefix.startswith('FV') else '_pct')
  assert np.isclose(r[k],x['ratio_of_means_pct'],atol=1e-12,rtol=1e-10)
  assert np.allclose(r[prefix+'_ci'],[x['ratio_ci_low'],x['ratio_ci_high']],atol=1e-12,rtol=1e-10)
out={'passed':True,'saved_cases':1128,'unique_episodes':24,'paired_contrasts_recomputed':len(verified),'all_rankings_and_figure_inputs_checked':True,
 'new_simulation_or_inference':False,'raw_quadrature_and_timing_audit_repeated':False,
 'scope':'Compact-record reanalysis. Original raw-input audit is documented separately; raw journals and quadrature arrays are not bundled.'}
(O/'replay_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
(O/'replay_paired_results.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
