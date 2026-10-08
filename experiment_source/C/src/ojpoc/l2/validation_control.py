"""Pre-calibration validation audit: reject poor repeated control before costly test work."""
import time
import numpy as np
from ..common import write_json,read_json,require,setup_torch
from .contracts import entries
from .data import Plant
from .engine import Engine
from .control import simulate


def audit_method(root,run,cfg,output,method):
    setup_torch(cfg);engine=Engine(root,run,method,calibrated=False);rows=[];started=time.perf_counter()
    try:
        for entry in entries(run,['validation']):
            plant=Plant(run/'02_generate/data',entry,cfg['budget']['duration_s'])
            metric,trace=simulate(plant,engine,'zero')
            accepted=[r for r in trace if r['event']=='request' and r['fallback_reason'] is None]
            rows.append({**metric,'episode':entry['episode_id'],'stratum':entry['stratum'],
                         'accepted_requests':len(accepted),'all_heaters_zero':sum(not any(r['command']) for r in accepted)})
            print(f'L2 validation control {method} {len(rows)} cost={metric["episode_cost"]:.6g}',flush=True)
        result={'method':method,'rows':rows,'seconds':time.perf_counter()-started,
                'scope':'validation only, zero compute delay, pre-calibration margin=0; not G5/test evidence'}
        write_json(output/('validation_control_'+method+'.json'),result);return result
    finally:engine.close()


def decide(output,cfg):
    values={m:read_json(output/('validation_control_'+m+'.json'))['rows'] for m in ['shared_physics','pi','pino_mpc']}
    comparisons={};checks={};fraction=cfg['l2']['noninferiority_fraction']
    for stratum in ['all','normal','stress']:
        selected={m:{r['episode']:r for r in rows if stratum=='all' or r['stratum']==stratum} for m,rows in values.items()}
        ids=set(selected['shared_physics']);require(all(set(v)==ids for v in selected.values()),'Unpaired validation control cohorts')
        if not ids:continue
        a=selected['shared_physics'];cost=float(np.mean([a[i]['episode_cost'] for i in ids]))
        for m in ['pi','pino_mpc']:
            b=selected[m];base=float(np.mean([b[i]['episode_cost'] for i in ids]));key=stratum+'/'+m
            checks[key+'/cost']=cost<=base*(1+fraction)+1e-12
            checks[key+'/violation']=all(a[i]['violation_s']<=b[i]['violation_s']+cfg['l2']['violation_time_tolerance_s'] for i in ids)
            comparisons[key]={'episodes':len(ids),'policy_cost':cost,'baseline_cost':base,'allowed_cost':base*(1+fraction)}
    primary={k:v for k,v in checks.items() if k.startswith('all/')}
    result={'passed':all(primary.values()),'checks':primary,
            'stratum_diagnostics':{k:v for k,v in checks.items() if not k.startswith('all/')},'comparisons':comparisons,
            'scope':'development audit on whole validation cohort, with separate stratum diagnostics; final G5 paired bootstrap criteria are unchanged'}
    write_json(output/'validation_control_gate.json',result);return result
