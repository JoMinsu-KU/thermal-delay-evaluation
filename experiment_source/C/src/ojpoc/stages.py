"""L1 stage implementations and inspectable, versioned artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .common import require, read_json, write_json, save_array, digest, seal, object_hash
from .data import split_plan, generate_episode, build_index, validate_dataset, EpisodeCache, bilinear_observer, initial_field
from .physics import (advance, branch, restrict, resize, candidates, costs, order_accuracy,
                      soft_targets, PROBES, CANDIDATES)


STAGES=[('00_preflight','Environment and pinned model checks'),('01_generate','Synthetic thermal episodes'),
        ('02_validate_index','Solver acceptance, data validation and causal index'),
        ('03_observer_labels','O-bilinear states and independent FV64 candidate labels'),
        ('04_operator_data','Operator windows and paired counterfactuals'),
        ('05_fno_data','FNO-data training and validation'),('06_pino_soft','PINO-soft training and validation'),
        ('07_pino_balance','PINO-balance training and validation'),('08_physics_cache','Frozen PINO feature cache'),
        ('09_native_no_physics','Original Open-Jev adaptation with observation text'),
        ('10_reload_no_physics','Fresh-process M0-no-physics reload'),
        ('11_native_physics_text','Original Open-Jev adaptation with PINO text'),
        ('12_reload_physics_text','Fresh-process M0-physics-text reload'),('13_report','Validation gates and L1 report'),
        ('14_fno_balance','Field-only FNO with identical balance projection'),
        ('15_eval_data','Preregistered held-out pilot evaluation episodes and FV labels'),
        ('16_operator_ablation','PV1 balance-only versus neural operators, initial/best/last'),
        ('17_policy_regret','PV2 paired native policy regret, baseline/best/last'),
        ('18_closed_loop','PV3 zero-compute-delay closed loop with five controllers'),
        ('19_latency','PV4 online local E2E and GPU latency, five controllers'),
        ('20_delayed_loop','PV4 measured-trace delayed closed loop'),
        ('21_report','Combined L1 and PV1--PV4 evidence report')]


def preflight(root,run,config,output):
    import shutil, subprocess
    require(Path(sys.prefix).resolve()==Path('/home/jovyan/.conda/envs/jev').resolve(),'Use the jev environment')
    sys.path.insert(0,str(root/'stages/L0'))
    spec=importlib.util.spec_from_file_location('l0_preflight',root/'stages/L0/00_preflight/run.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from common import Context
    result=module.run(Context(root,run,output,True))
    require(shutil.disk_usage(root).free/2**30>8,'At least 8 GiB visible free disk required')
    manifest=read_json(root/'locks/models-2B.json')
    for name,item in manifest['files'].items():
        label,relative=name.split('/',1)
        source=Path(manifest['adapter_snapshot'] if label=='adapter_package' else manifest['base_snapshot'])/relative
        require(source.is_file() and digest(source)==item['sha256'],f'Pinned model cache changed: {name}')
    previous=[]
    for path in (root/'runs/L0').glob('*/summary.json'):
        summary=read_json(path)
        if summary['status'] in ['PASS','DIAGNOSTIC_PASS']:
            previous.append(summary)
    require(previous,'Complete a full L0 before L1')
    result.update({'l0_run':max(previous,key=lambda x:x['started_at'])['run_id'],
                   'profile':config['profile'],'model_download':False,'actuation_enabled':False})
    return result


def generate(root,run,config,output):
    base=output/'data'; (base/'manifests').mkdir(parents=True,exist_ok=True)
    entries=split_plan(config)
    write_json(base/'manifests/split.json',{'episodes':entries,'group_split_before_generation':True,'seed':config['seed']})
    write_json(base/'manifests/generation-config.json',config)
    write_json(base/'manifests/sensors.json',{'probes_xy_m':PROBES.tolist(),'array_axes':'y,x',
               'interpolation':'cell-centred bilinear, nearest boundary extension','heater_order':
               ['low_y_low_x','low_y_high_x','high_y_low_x','high_y_high_x'],
               'temperature':'Celsius','time':'int64 nanoseconds','initial_u_p':0,
               'command_apply':'synchronous zero-compute-delay; no delayed SIL claim'})
    config={**config,'openjev_commit':(root/'locks/openjev.commit').read_text().strip()}
    rows=[]
    for i,entry in enumerate(entries):
        rows.append(generate_episode(base,entry,config))
        print(f'generated {i+1}/{len(entries)} {entry["split"]} {entry["episode_id"]}',flush=True)
    return {'episodes':len(rows),'duration_s':60,'counts':config['poc']['data']['profiles'][config['profile']],
            'normal_and_stress_retained':True,'control':'causal PI with saturation/anti-windup, perturbation and stress mix',
            'test_and_calibration_generated_but_not_used_for_fitting':True}


def solver_acceptance(config):
    theta=np.array([12000.,.75,25.,25.,40000.,.1,.1,.1,.1]); zero=np.zeros(4)
    field=np.full((64,64),80.); cooled,_=advance(field,zero,zero,theta,duration=1.)
    cooling=float(abs(cooled-(25+55*np.exp(-25/12000))).max())
    conservative=theta.copy(); conservative[2]=0
    initial=initial_field(np.random.default_rng(9),64,65)
    evolved,_=advance(initial,zero,zero,conservative,duration=1.)
    conservation=float(abs(initial.mean()-evolved.mean()))
    ambient=np.full((64,64),25.); unchanged,_=advance(ambient,zero,zero,theta,duration=.2)
    require(np.max(abs(unchanged-ambient))<1e-10,'Ambient equilibrium failed')
    command,valid=candidates(np.full(4,.3)); p=np.full(4,.3)
    fine=branch(initial,p,command,theta); refined=branch(initial,p,command,theta,grid=128,max_dt=.005)
    temporal=branch(initial,p,command,theta,max_dt=.005)
    c1=costs(fine,command,np.full(4,.3),80); c2=costs(refined,command,np.full(4,.3),80)
    correct,total=order_accuracy(c1,c2,valid,config['acceptance']['solver']['significant_cost_gap'])
    result={'cooling_abs_k':cooling,'conservation_abs_k':conservation,
            'refinement_rmse_k':float(np.sqrt(np.mean((fine-restrict(refined,64))**2))),
            'refinement_peak_k':float(abs(fine.max((2,3))-refined.max((2,3))).max()),
            'time_refinement_rmse_k':float(np.sqrt(np.mean((fine-temporal)**2))),
            'refinement_order_accuracy':correct/total if total else None,'significant_pairs':total}
    gates=config['acceptance']['solver']
    require(cooling<=gates['cooling_abs_k'] and conservation<=gates['conservation_abs_k'],'Analytic solver gate failed')
    require(result['refinement_rmse_k']<=gates['refinement_rmse_k'] and result['refinement_peak_k']<=gates['refinement_peak_k'],
            'FV discretization error too large')
    require(total>0 and correct/total>=gates['refinement_order_accuracy'],'FV cost order not stable under refinement')
    return result


def validate_index(root,run,config,output):
    base=run/'01_generate/data'; entries=read_json(base/'manifests/split.json')['episodes']
    result={'solver':solver_acceptance(config),'data':validate_dataset(base,entries),'index':build_index(base,entries,config)}
    entry=next(e for e in entries if e['split']=='validation'); eid=entry['episode_id']
    meta=read_json(base/'raw'/eid/'meta.json'); theta=np.array(meta['passport'])
    initial=np.load(base/'truth'/eid/'initial_T64.npy'); coarse=initial.copy(); refined=resize(initial,128)
    p64=np.zeros(4); p128=np.zeros(4); errors=[]; peaks=[]
    commands=np.load(base/'raw'/eid/'commands.npy',mmap_mode='r')
    for command in commands[:,2]:
        coarse,p64=advance(coarse,p64,command,theta)
        refined,p128=advance(refined,p128,command,theta,max_dt=.005)
        errors.append(float(np.mean((coarse-restrict(refined,64))**2)))
        peaks.append(float(abs(coarse.max()-refined.max())))
    commands._mmap.close()
    refinement={'episode':eid,'duration_s':meta['duration_s'],'field_rmse_k':float(np.sqrt(np.mean(errors))),
                'maximum_peak_difference_k':max(peaks),'commands':'same recorded applied sequence; no policy re-optimization'}
    result['validation_episode_refinement']=refinement
    require(refinement['field_rmse_k']<=config['acceptance']['solver']['refinement_rmse_k'] and
            refinement['maximum_peak_difference_k']<=config['acceptance']['solver']['refinement_peak_k'],'Episode grid refinement gate failed')
    write_json(output/'validation.json',result)
    result['external_artifacts']={str(p.relative_to(run)):digest(p) for p in
                                 [*sorted((base/'index').glob('*')),base/'manifests/normalization.json'] if p.is_file()}
    # Index/normalizer are owned by this stage; raw/truth artifacts remain immutable.
    return result


def labels(root,run,config,output):
    base=run/'01_generate/data'; entries=read_json(base/'manifests/split.json')['episodes']; cache=EpisodeCache(base,entries)
    diagnostics=[]; gaps=[]; fallback={}; observer_hash=object_hash({'observer':'bilinear-probe-v1','timestamp':'capture_receive_ready_le_decision'})
    for split in ['train','validation']:
        index=np.load(base/'index'/(split+'_windows.npy'),mmap_mode='r'); n=len(index)
        shapes={'field':(n,32,32),'p0':(n,4),'passport':(n,9),'commands':(n,9,20,4),'previous':(n,4),
                'goal':(n,),'age':(n,),'cost':(n,9),'valid':(n,9),'eligible':(n,9),'target':(n,9)}
        arrays={key:np.lib.format.open_memmap(output/(split+'_'+key+'.npy'),mode='w+',
                  dtype=bool if key in ['valid','eligible'] else np.float32,shape=shape) for key,shape in shapes.items()}
        fallbacks=[]
        for i,(code,step,_) in enumerate(index):
            obs=cache.get(code).snapshot(step); estimated=bilinear_observer(obs)
            command,valid=candidates(obs.previous); future=branch(estimated,obs.actuator,command,obs.passport)
            cost=costs(future,command,obs.previous,obs.goal); eligible=valid&(future.max((1,2,3))<=120)
            arrays['field'][i]=estimated; arrays['p0'][i]=obs.actuator; arrays['passport'][i]=obs.passport
            arrays['commands'][i]=command; arrays['previous'][i]=obs.previous; arrays['goal'][i]=obs.goal
            arrays['age'][i]=obs.thermal_age[-1]; arrays['cost'][i]=cost; arrays['valid'][i]=valid; arrays['eligible'][i]=eligible
            if not eligible.any(): fallbacks.append({'row':i,'reason':'no_expert_feasible_candidate'})
            if split=='train':
                active=cost[eligible]; gaps.extend(abs(active[a]-active[b]) for a in range(len(active)) for b in range(a+1,len(active)))
            if i==0:  # One observation state per fitting split: refinement is separate from true-state diagnostics.
                refined=branch(estimated,obs.actuator,command,obs.passport,grid=128,max_dt=.005)
                refined_cost=costs(refined,command,obs.previous,obs.goal)
                ok,total=order_accuracy(cost,refined_cost,valid,config['acceptance']['solver']['significant_cost_gap'])
                metric={'split':split,'field_rmse_k':float(np.sqrt(np.mean((future-restrict(refined,64))**2))),
                        'peak_error_k':float(abs(future.max((2,3))-refined.max((2,3))).max()),
                        'order_accuracy':ok/total if total else None,'significant_pairs':total}
                diagnostics.append(metric)
                gate=config['acceptance']['solver']
                require(metric['field_rmse_k']<=gate['refinement_rmse_k'] and metric['peak_error_k']<=gate['refinement_peak_k'],
                        'Observed-state FV label refinement failed')
                require(total>0 and ok/total>=gate['refinement_order_accuracy'],'Observed-state label order unstable')
            print(f'FV64 labels {split} {i+1}/{n}',flush=True)
        for a in arrays.values(): a.flush(); a._mmap.close()
        save_array(output/(split+'_index.npy'),np.array(index)); index._mmap.close(); fallback[split]=fallbacks
    equivalence=config['acceptance']['solver']['significant_cost_gap']
    positive=np.array([v for v in gaps if v>equivalence]); temperature=float(np.median(positive)) if len(positive) else equivalence
    for split in ['train','validation']:
        cost=np.load(output/(split+'_cost.npy'),mmap_mode='r'); eligible=np.load(output/(split+'_eligible.npy'),mmap_mode='r')
        target=np.load(output/(split+'_target.npy'),mmap_mode='r+')
        for i in range(len(cost)): target[i]=soft_targets(cost[i],eligible[i],temperature)
        target.flush()
        for a in [cost,eligible,target]: a._mmap.close()
    cache.close()
    write_json(output/'label_temperature.json',{'value':temperature,'source_split':'train','method':'median significant absolute pairwise cost gap',
                'equivalence_gap':equivalence,'validation_adjusted':False})
    write_json(output/'fallback.json',fallback)
    metadata={'teacher':'FV64-MPC_obs','observer_hash':observer_hash,'candidate_ids':CANDIDATES,
              'dataset_split_sha256':digest(base/'manifests/split.json'),'normalization_sha256':digest(base/'manifests/normalization.json'),
              'cost_version':'thermal_poc_v1','candidate_library':'axis9_poc_v1','timestamp_policy':'capture_receive_ready_le_decision_v1',
              'truth_used_in_teacher_initial_state':False,'future_disturbances_used':False,
              'refinement':diagnostics,'fallback_counts':{k:len(v) for k,v in fallback.items()},'test_accessed':False}
    write_json(output/'provenance.json',metadata)
    return metadata


def report(root,run,config,output):
    ops={name:read_json(run/stage/'result.json') for name,stage in
         [('fno_data','05_fno_data'),('pino_soft','06_pino_soft'),('pino_balance','07_pino_balance')]}
    policies={name:read_json(run/stage/'result.json') for name,stage in
              [('M0-no-physics','09_native_no_physics'),('M0-physics-text','11_native_physics_text')]}
    quality=ops['pino_balance']['quality_pass'] and policies['M0-physics-text']['quality_pass']
    rows=['# L1 validation report','',f'Profile: `{config["profile"]}`; seed: `{config["seed"]}`',
          f'Quality gates met: **{quality}**','',
          'This report uses train/validation only. Calibration, test/OOD performance, delayed SIL and p99 are not certified.',
          'Smoke budgets verify executable connections; a smoke pass is not an L1 scientific-quality pass.','',
          '| Operator | H RMSE (K) | Candidate ΔT RMSE (K) | Order accuracy | G3 |','|---|---:|---:|---:|---|']
    for name,result in ops.items():
        r=result['candidate_validation']; rows.append(f'| {name} | {result["validation"]["horizon_rmse_k"]:.4f} | {r["candidate_delta_rmse_k"]:.4f} | {r["significant_order_accuracy"]} | {result["quality_pass"]} |')
    rows+=['','| Policy | Original regret | Adapted regret | Failure/fallback fraction | G4 improvement |','|---|---:|---:|---:|---|']
    for name,result in policies.items():
        rows.append(f'| {name} | {result["baseline_validation"]["mean_regret"]} | {result["selected_validation"]["mean_regret"]} | {result["selected_validation"]["failure_or_fallback_fraction"]} | {result["quality_pass"]} |')
    from .common import atomic_text
    atomic_text(output/'report.md','\n'.join(rows)+'\n')
    return {'quality_pass':quality,'operators':{k:v['quality_pass'] for k,v in ops.items()},
            'policies':{k:v['quality_pass'] for k,v in policies.items()},'profile':config['profile'],
            'next':'pilot only after reviewing validation and resolving failed gates; no automatic L2 claim',
            'test_accessed':False}


def execute_stage(name,root,run,config,output):
    if name=='00_preflight': return preflight(root,run,config,output)
    if name=='01_generate': return generate(root,run,config,output)
    if name=='02_validate_index': return validate_index(root,run,config,output)
    if name=='03_observer_labels': return labels(root,run,config,output)
    if name=='04_operator_data':
        from .training import prepare_operator_data
        return prepare_operator_data(run,config)
    if name in ['05_fno_data','06_pino_soft','07_pino_balance']:
        from .training import train_operator
        return train_operator(run,config,name[3:],output)
    if name=='08_physics_cache':
        from .common import setup_torch
        setup_torch(config)
        from .training import cache_physics
        return cache_physics(run,config,output)
    if name in ['09_native_no_physics','11_native_physics_text']:
        from .policy import train_native
        return train_native(root,run,config,output,physics=name.startswith('11'))
    if name in ['10_reload_no_physics','12_reload_physics_text']:
        from .policy import reload_native
        return reload_native(root,run,config,'09_native_no_physics' if name.startswith('10') else '11_native_physics_text')
    if name=='13_report': return report(root,run,config,output)
    if name=='14_fno_balance':
        from .training import train_operator
        return train_operator(run,config,'fno_balance',output)
    if name=='15_eval_data':
        from .evaluation import prepare_evaluation
        return prepare_evaluation(root,run,config,output)
    if name=='16_operator_ablation':
        from .evaluation import operator_ablation
        return operator_ablation(root,run,config,output)
    if name in ['17_policy_regret','18_closed_loop','19_latency','20_delayed_loop','21_report']:
        from .pilot import policy_regret,closed_loop,latency_benchmark,final_report
        if name=='17_policy_regret': return policy_regret(root,run,config,output)
        if name=='18_closed_loop': return closed_loop(root,run,config,output)
        if name=='19_latency': return latency_benchmark(root,run,config,output)
        if name=='20_delayed_loop': return closed_loop(root,run,config,output,mode='delayed')
        return final_report(root,run,config,output)
    raise ValueError(f'Unknown L1 stage: {name}')
