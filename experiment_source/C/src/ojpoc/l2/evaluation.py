"""Locked held-out evaluation and episode-level, paired summaries."""
from collections import defaultdict
import numpy as np
from ..common import read_json,write_json,require,digest,setup_torch
from ..data import RawEpisode
from ..evaluation import Journal,quantiles
from .contracts import entries,modality,assert_frozen,ABLATIONS
from .cache import make_split,operator,Rows
from .engine import Engine


def cohort(entry):
    return 'ID' if entry['split']=='test' else 'OOD-'+entry['ood_group']


def prepare_test(run,cfg,output):
    setup_torch(cfg);assert_frozen(run)
    require((run/'16_calibration/COMPLETE').exists(),'Calibration must be sealed before test access')
    op=operator(run);result={}
    for mode in ['both','sensor','thermal']:
        for split in ['test','ood']:result[mode+'/'+split]=make_split(run,cfg,output/'cache'/mode/split,mode,split,op)
    return result


def evaluate_method(root,run,cfg,output,method):
    setup_torch(cfg);assert_frozen(run);engine=Engine(root,run,method)
    identity={'method':method,'freeze':digest(run/'15_freeze/frozen.json'),'calibration':digest(run/'16_calibration/calibration.json')}
    journal=Journal(output/(method+'.jsonl'),identity);seen={(r['episode'],r['step']) for r in journal.rows}
    manifest={e['code']:e for e in entries(run,['test','ood'])};raw=None;old=None
    try:
        for split in ['test','ood']:
            rows=Rows(run/'17_evaluation/cache'/modality(method)/split)
            try:
                for i,(code,step) in enumerate(rows.arrays['index']):
                    e=manifest[int(code)];step=int(step)
                    if (e['episode_id'],step) in seen:continue
                    if old!=int(code):
                        if raw:raw.close()
                        raw=RawEpisode(run/'02_generate/data/raw'/e['episode_id']);old=int(code)
                    decision=engine.decide(lambda:raw.snapshot(step));choice=decision['choice']
                    eligible=rows.arrays['eligible'][i];cost=rows.arrays['cost'][i]
                    valid_choice=0<=choice<9;available=bool(eligible.any());regret=None;correct=None
                    if valid_choice and available:
                        regret=float(cost[choice]-cost[eligible].min());correct=bool(eligible[choice] and regret<=1e-6)
                    failure=bool((valid_choice and not eligible[choice]) or (available and not valid_choice))
                    target=np.array(rows.arrays['target'][i]);prob=decision['probabilities'];nll=None
                    if prob is not None and target.sum()>0:
                        nll=float(-np.dot(target,np.log(np.maximum(prob,1e-12))))
                    journal.append({'episode':e['episode_id'],'step':step,'cohort':cohort(e),'stratum':e['stratum'],
                                    'ood_condition':e['ood_condition'],'method':method,'choice':choice,'regret':regret,
                                    'expert_feasible':available,'failure':failure,'infeasible':bool(valid_choice and not eligible[choice]),
                                    'unexpected_fallback':bool(available and not valid_choice),'fallback':not valid_choice,
                                    'correct':correct,'confidence':float(max(prob)) if prob is not None else None,
                                    'nll':nll,'fallback_reason':decision['fallback_reason'],'tokens':decision['tokens'],
                                    'diagnostic_ablation':method in ABLATIONS})
            finally:rows.close()
        result={key:summarize_rows([r for r in journal.rows if r['cohort']==key],cfg) for key in sorted({r['cohort'] for r in journal.rows})}
        write_json(output/(method+'.summary.json'),result);return result
    finally:
        if raw:raw.close()
        engine.close()


def summarize_rows(rows,cfg):
    points=[r for r in rows if r['correct'] is not None and r['confidence'] is not None]
    ece=0.;bins=[]
    for i in range(cfg['l2']['probability_bins']):
        lo=i/cfg['l2']['probability_bins'];hi=(i+1)/cfg['l2']['probability_bins']
        group=[r for r in points if lo<=r['confidence'] and (r['confidence']<hi or (i==cfg['l2']['probability_bins']-1 and r['confidence']<=hi))]
        if group:
            confidence=float(np.mean([r['confidence'] for r in group]));accuracy=float(np.mean([r['correct'] for r in group]))
            ece+=len(group)/len(points)*abs(confidence-accuracy);bins.append({'lo':lo,'hi':hi,'n':len(group),'confidence':confidence,'accuracy':accuracy})
    return {'rows':len(rows),'episodes':len({r['episode'] for r in rows}),
            'regret':quantiles([r['regret'] for r in rows if r['regret'] is not None]),
            **{k:sum(bool(r[k]) for r in rows) for k in ['expert_feasible','failure','infeasible','unexpected_fallback','fallback']},
            'confidence_rows':len(points),'ece':ece if points else None,'bins':bins,
            'nll':quantiles([r['nll'] for r in rows if r['nll'] is not None]),'confidence_is_not_safety_probability':True}


def paired(left,right,cfg,metric='episode_cost'):
    lookup={r['episode']:r for r in right};differences=[];ids=[]
    for row in left:
        if row['episode'] not in lookup:continue
        a=row[metric];b=lookup[row['episode']][metric]
        if a is None or b is None:continue
        ids.append(row['episode']);differences.append(a-b)
    out={'episodes':len(ids),'episode_ids':ids,'difference':None,'ci95':None,'interpretation':'INCONCLUSIVE'}
    if differences:
        values=np.array(differences);out['difference']=float(values.mean())
        if len(values)>=2:
            rng=np.random.default_rng(cfg['seed']+991);samples=values[rng.integers(len(values),size=(cfg['l2']['bootstrap_resamples'],len(values)))].mean(1)
            lo,hi=map(float,np.quantile(samples,[.025,.975]));out['ci95']=[lo,hi]
            out['interpretation']='POSITIVE' if hi<0 else 'NEGATIVE' if lo>0 else 'INCONCLUSIVE'
    return out


def regret_pairs(run,cfg):
    rows={}
    for method in cfg['l2']['methods']+ABLATIONS:
        path=run/'17_evaluation'/(method+'.jsonl')
        import json
        rows[method]=[json.loads(line)['row'] for line in path.open()]
    results={}
    for group in sorted({r['cohort'] for r in rows['shared_physics']}):
        results[group]={}
        for reference in ['shared_no','pino_mpc','small_physics','pi','native_physics','shared_sensor','shared_thermal',*ABLATIONS]:
            left={(r['episode'],r['step']):r for r in rows['shared_physics'] if r['cohort']==group}
            right={(r['episode'],r['step']):r for r in rows[reference] if r['cohort']==group}
            values=defaultdict(list);excluded=0
            for key,a in left.items():
                b=right.get(key)
                if b is None or a['regret'] is None or b['regret'] is None:excluded+=1;continue
                values[key[0]].append(a['regret']-b['regret'])
            paired_rows=[{'episode':k,'regret':float(np.mean(v))} for k,v in values.items()]
            baseline=[{'episode':k,'regret':0.} for k in values]
            results[group][reference]={**paired(paired_rows,baseline,cfg,'regret'),'excluded_undefined_rows':excluded,
                                       'planned_rows':len(left),'common_rows':sum(map(len,values.values()))}
    return results
