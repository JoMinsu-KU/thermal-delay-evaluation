"""Reanalyse compact saved Small initialization outcomes; no models or simulator.

MIT code. Data/figures: CC BY 4.0 in the public distribution.
Inputs retain all three new seeds, including the failed validation-quality flag.
"""
from pathlib import Path
import argparse,csv,json,math
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-10)
def main(data,out):
    out.mkdir(parents=True,exist_ok=True)
    records=csvrows(data/'episode_metrics.csv')
    lookup={(int(r['seed']),r['mode'],r['episode']):r for r in records}
    assert len(records)==len(lookup)==192
    design=read(data/'design.json');roster=design['normal_episodes'];assert len(roster)==24
    fixed=read(data/'fixed_reference.json');original=read(data/'original_training.json')
    refs=read(data/'validation_reference.json');training=[];validation=[];alpha={6060:.5};quality={6060:True}
    for seed in design['initialization_seeds']:
        t=read(data/f'training_{seed}.json')
        assert len(t['history'])==t['optimizer_steps']==450
        assert t['batch_sequence']==original['batch_sequence'] and t['training_data_hash']==original['training_data_hash']
        assert sum(x['phase']=='warmup' for x in t['history'])==50 and all(x['global_decisions']==16 for x in t['history'])
        assert t['test_accessed'] is False and t['selection']['split']=='validation'
        options=[(t['baseline']['selection_score'],['initial',0])]+[(x['validation']['selection_score'],[x['phase'],x['step']]) for x in t['history'] if 'validation' in x]
        assert min(options,key=lambda x:x[0])[1]==t['selected_step']
        for c in t['selection']['candidates']:
            actual={(x['delay'],x['episode']):x for x in c['rows']};assert len(actual)==32
            assert not ({x['episode'] for x in c['rows']}&set(roster))
            adverse=0;ratios=[]
            for ref in refs.values():
                expected={(x['delay'],x['episode']):x for x in ref};assert set(actual)==set(expected)
                for k,a in actual.items():
                    b=expected[k];assert a['completed'] and math.isfinite(a['episode_cost'])
                    adverse+=int(a['new_violation']>b['new_violation'] or a['violation_s']>b['violation_s']+1e-8 or a['inference_failures']>b['inference_failures'])
                for delay in sorted({x['delay'] for x in c['rows']}):
                    for cohort in ['all','normal']:
                        keys=[k for k,x in actual.items() if k[0]==delay and (cohort=='all' or x['stratum']==cohort)]
                        ratios.append(np.mean([actual[k]['episode_cost'] for k in keys])/max(1e-8,np.mean([expected[k]['episode_cost'] for k in keys])))
            k=[adverse,max(ratios),np.mean(ratios)];assert all(close(a,b) for a,b in zip(k,c['key']))
            validation.append({'seed':seed,'alpha':c['alpha'],'rollouts':32,'adverse_comparisons':adverse,'worst_ratio':float(k[1]),'mean_ratio':float(k[2])})
        alpha[seed]=min(t['selection']['candidates'],key=lambda c:(*c['key'],c['alpha']))['alpha']
        assert alpha[seed]==t['selection']['selected_alpha']
        q=t['selected'];b=t['baseline'];quality[seed]=q['mean_regret']<b['mean_regret'] and q['infeasible']==0 and q['unexpected_fallback']==0
        assert quality[seed]==t['quality_pass']
        training.append({'seed':seed,'selected_phase':t['selected_step'][0],'selected_step':t['selected_step'][1],'alpha':alpha[seed],
                         'quality_pass':quality[seed],'optimizer_steps':450,'initial_candidate_cost_difference':b['mean_regret'],
                         'selected_candidate_cost_difference':q['mean_regret'],'change_pct':100*(q['mean_regret']/b['mean_regret']-1)})
    idx=np.random.default_rng(20261013).integers(0,24,(20000,24));contrasts=[];effects=[]
    for seed in [6060,*design['initialization_seeds']]:
        for mode in ['zero','delayed']:
            small=np.array([float(lookup[seed,mode,e]['episode_cost']) for e in roster])
            shared=np.array([fixed['shared_physics'][mode][e]['episode_cost'] for e in roster])
            d=small-shared;ci=np.quantile(d[idx].mean(1),[.025,.975])
            contrasts.append({'seed':seed,'mode':mode,'alpha':alpha[seed],'validation_quality_pass':quality[seed],
              'paired_episodes':24,'small_mean_cost':float(small.mean()),'fixed_shared_mean_cost':float(shared.mean()),
              'small_minus_shared':float(d.mean()),'small_relative_to_shared_pct':100*(float(small.mean()/shared.mean())-1),
              'conditional_episode_ci_low':float(ci[0]),'conditional_episode_ci_high':float(ci[1]),
              'small_wins':int((d< -1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'small_losses':int((d>1e-10).sum())})
        z,d=contrasts[-2:]
        effects.append({'seed':seed,'small_replay_penalty_pct':100*(d['small_mean_cost']/z['small_mean_cost']-1),
          'shared_replay_penalty_pct':100*(d['fixed_shared_mean_cost']/z['fixed_shared_mean_cost']-1),
          'rank_reversal':z['small_minus_shared']>0 and d['small_minus_shared']<0})
    for expected in csvrows(data/'expected_contrasts.csv'):
        now=next(x for x in contrasts if x['seed']==int(expected['seed']) and x['mode']==expected['mode'])
        for k in ['small_mean_cost','fixed_shared_mean_cost','small_minus_shared','small_relative_to_shared_pct','conditional_episode_ci_low','conditional_episode_ci_high','small_wins','small_losses']:
            assert close(now[k],expected[k]),(k,now[k],expected[k])
    writecsv(out/'paired_contrasts.csv',contrasts);writecsv(out/'training_checks.csv',training)
    writecsv(out/'validation_selection.csv',validation);writecsv(out/'delay_effects.csv',effects)
    summary={'passed':True,'test_runs':192,'baseline_runs':48,'new_test_runs':144,'validation_rollouts':288,'unique_test_episodes':24,
      'new_seeds':3,'fixed_shared_checkpoints':1,'quality_pass_count':sum(quality[s] for s in design['initialization_seeds']),
      'new_replay_lower_cost':sum(x['small_minus_shared']<0 for x in contrasts if x['seed']!=6060 and x['mode']=='delayed'),
      'new_rank_reversals':sum(x['rank_reversal'] for x in effects if x['seed']!=6060),
      'new_test_limit_violation_episodes':sum(r['new_violation']=='True' for r in records if r['seed']!='6060'),
      'scope':'Saved episode and validation metrics only. Raw quadrature/field, source-seal, reload and remote-input checks are separate author audit receipts.'}
    (out/'checks.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=root/'data/small_seed');p.add_argument('--output',type=Path,default=root/'outputs/small_seed')
    args=p.parse_args();main(args.data_dir,args.output)
