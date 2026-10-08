#!/usr/bin/env python3
"""Offline evidence audit and paired summaries; no research model is loaded."""
from __future__ import annotations
import argparse
import csv
import gzip
import math
from pathlib import Path
import numpy as np
from validation_io import completed_case, read_json, read_trace, require, sha256, write_json
from plan import METHODS, MODES, D_STEPS, C_STEPS, DELAYS, LABELS, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES


def csv_file(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def truth(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        return [{k:int(v) if k=='time_ns' else float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def command_integrals(trace, duration):
    last=0.; u=np.zeros(4); fallback=True; effort=seconds=delta=0.; effective=0
    for row in trace:
        if row['event'] not in ('apply','deadline_fallback'):continue
        t=row['time_ns']/1e9
        require(last<=t<=duration,'Nonmonotone command trace')
        effort+=float(np.mean(u*u))*(t-last);seconds+=fallback*(t-last)
        v=np.asarray(row['command'],dtype=np.float64)
        require(v.shape==(4,) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all(),'Invalid command')
        same=bool(np.array_equal(u,v) and fallback==row['fallback'])
        require(same==row['no_change'],'No-op flag is inconsistent')
        require(row['fallback'] or np.max(np.abs(v-u))<=.04000001,'Applied nonfallback slew violation')
        if not same:
            delta+=float(np.mean((v-u)**2));effective+=1
        u=v;fallback=row['fallback'];last=t
    effort+=float(np.mean(u*u))*(duration-last);seconds+=fallback*(duration-last)
    return {'action_effort':effort/duration,'fallback_s':seconds,
            'action_delta_mean_square':delta/round(duration*10),'effective_applications':effective}


def audit_case(folder, metrics, trace, rows):
    duration=metrics['duration_s']; end=round(duration*1e9)
    require(metrics['completed'],'Incomplete metric marked sealed')
    require(rows and rows[0]['time_ns']==0 and rows[-1]['time_ns']==end,'Incomplete truth log')
    commands=command_integrals(trace,duration)
    with np.load(folder/'quadrature.npz',allow_pickle=False) as q:
        dt=q['dt_ns'].astype(np.int64); ends=q['end_ns']
        require(len(dt)>0 and (dt>0).all() and np.array_equal(np.cumsum(dt),ends) and ends[-1]==end,'Quadrature clock coverage changed')
        for key in ('mse_k2','command_mse','peak_c'):
            require(np.isfinite(q[key]).all(),f'Nonfinite {key}')
        require((q['mse_k2']>=0).all(),'Negative mean squared error')
        seconds=dt/1e9
        tracking=float(np.sum(q['mse_k2']*seconds)/duration)
        peak=max(rows[0]['max_temperature_c'],float(q['peak_c'].max()))
        qeffort=float(np.sum(q['command_mse']*seconds)/duration)
        qfallback=float(np.sum(q['fallback']*seconds))
        violation=float(np.sum((q['peak_c']>120)*seconds))
    field=np.load(folder/'final_T64.npy',allow_pickle=False)
    require(np.isfinite(field).all(),'Nonfinite final field')
    terminal=float(np.mean(((field-rows[0]['goal_c'])/100)**2))
    cost=tracking/10000+.5*terminal+.002*commands['action_effort']+.02*commands['action_delta_mean_square']
    expected={'episode_cost':cost,'tracking_rmse_k':math.sqrt(tracking),
              'max_temperature_c':peak,'violation_s':violation,
              **{k:v for k,v in commands.items() if k!='effective_applications'}}
    errors=[]
    for key,value in expected.items():
        if not math.isclose(value,metrics[key],abs_tol=1e-10,rel_tol=1e-8):
            errors.append({'metric':key,'reconstructed':value,'saved':metrics[key]})
    for key,a,b in (('quadrature_effort',qeffort,commands['action_effort']),('quadrature_fallback',qfallback,commands['fallback_s'])):
        if not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-8):errors.append({'metric':key,'quadrature':a,'trace':b})
    if 'requests' in metrics:
        requests=[r for r in trace if r['event']=='request']
        countchecks={'requests':len(requests),'missed_deadlines':sum(r['expired'] for r in requests),
                     'busy_skipped_ticks':sum(r['event']=='busy_skip' for r in trace),
                     'filter_interventions':sum(bool(r.get('filter_intervention')) for r in requests)}
        for key,value in countchecks.items():
            if metrics[key]!=value:errors.append({'metric':key,'reconstructed':value,'saved':metrics[key]})
        ticks=[r['snapshot_ns'] for r in requests]+[r['time_ns'] for r in trace if r['event']=='busy_skip']
        if sorted(ticks)!=list(range(0,end,100_000_000)):errors.append({'request_tick_coverage':'incomplete or duplicated'})
    return {'passed':not errors,'cost_abs_error':abs(cost-metrics['episode_cost']),'errors':errors}


def paired(a,b,indices):
    """B minus A, with episodes paired before bootstrapping."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);d=b-a
    require(a.shape==b.shape==(indices.shape[1],),'Pair count differs')
    samples=d[indices].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    relative=(b/a-1)*100
    rlo,rhi=np.quantile(relative[indices].mean(axis=1),[.025,.975])
    # Ratio of cohort means is distinct from the mean paired relative change.
    ratios=(b[indices].mean(axis=1)/a[indices].mean(axis=1)-1)*100
    ratio_ci=np.quantile(ratios,[.025,.975])
    return {'n':len(a),'mean_A':float(a.mean()),'mean_B':float(b.mean()),'mean_B_minus_A':float(d.mean()),
            'difference_ci_low':float(lo),'difference_ci_high':float(hi),
            'mean_paired_relative_pct':float(relative.mean()),'relative_ci_low':float(rlo),'relative_ci_high':float(rhi),
            'ratio_of_means_pct':float((b.mean()/a.mean()-1)*100),
            'ratio_ci_low':float(ratio_ci[0]),'ratio_ci_high':float(ratio_ci[1]),
            **{f'B_{kind}_tol_{tol:g}':int(func(d,tol).sum())
               for tol in (1e-10,1e-8,1e-6)
               for kind,func in (('wins',lambda x,t:x < -t),('ties',lambda x,t:abs(x)<=t),('losses',lambda x,t:x>t))}}


def direction(value,tol=1e-10):
    return 'positive' if value>tol else 'negative' if value < -tol else 'within_tolerance'


def analyze(output):
    output=Path(output); protocol=read_json(output/'protocol.json');ph=sha256(output/'protocol.json')
    jobs=protocol['design']['jobs']; episodes=protocol['design']['normal_episodes']
    require(len(episodes)==24 and len(set(episodes))==24,'Expected 24 unique normal episodes')
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    records={};flat=[];audit=[];missing=[];problems=[];fixed={}
    for job in jobs:
        for case in job['cases']:
            key=case['key']
            try:
                folder=completed_case(output,key,ph)
                if folder is None:missing.append(key);continue
                condition=read_json(folder/'condition.json')
                require(condition=={**{k:v for k,v in job.items() if k!='cases'},**case},'Condition differs from protocol')
                m=read_json(folder/'metrics.json');trace=read_trace(folder/'trace.jsonl.gz');rows=truth(folder/'truth.csv.gz')
                check=audit_case(folder,m,trace,rows);audit.append({'case':key,**check})
                require(check['passed'],'Independent metric reconstruction failed')
                flat.append({'case':key,'source':job['source'],'h_ms':job['h_ms'],'episode':case['episode'],
                             'method':job.get('method','fixed_C'),'mode':job.get('mode','fixed'),
                             'delay_ms':case.get('delay_ms'),**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}})
                if job['source']=='D':
                    records[(job['h_ms'],job['method'],job['mode'],case['episode'])]=m
                else:
                    with np.load(folder/'fields_1hz.npz',allow_pickle=False) as f:
                        times=f['time_ns'];fields=f['temperature_c']
                    require(np.array_equal(times,np.arange(61)*1_000_000_000),'C field sampling changed')
                    fixed[(job['h_ms'],case['delay_ms'])]=(m,rows,fields)
            except Exception as exc:
                problems.append({'case':key,'error':repr(exc)})
    fixed_diffs=[]
    for delay in DELAYS:
        for coarse,fine in zip(C_STEPS,C_STEPS[1:]):
            if (coarse,delay) not in fixed or (fine,delay) not in fixed:continue
            a,ar,af=fixed[(coarse,delay)];b,br,bf=fixed[(fine,delay)]
            require([r['time_ns'] for r in ar]==[r['time_ns'] for r in br],'C truth timestamps differ')
            d=bf-af; md=np.array([r['mean_temperature_c'] for r in br])-np.array([r['mean_temperature_c'] for r in ar])
            fixed_diffs.append({'delay_ms':delay,'coarse_ms':coarse,'fine_ms':fine,
                                'cost_coarse':a['episode_cost'],'cost_fine':b['episode_cost'],
                                'cost_change':b['episode_cost']-a['episode_cost'],
                                'cost_relative_pct':(b['episode_cost']/a['episode_cost']-1)*100,
                                'field_trajectory_rmse_k_1hz':float(np.sqrt(np.mean(d*d))),
                                'field_trajectory_max_abs_k_1hz':float(abs(d).max()),
                                'final_field_rmse_k':float(np.sqrt(np.mean(d[-1]**2))),
                                'mean_temperature_rmse_k_100hz':float(np.sqrt(np.mean(md*md)))})
    idx=np.random.default_rng(BOOTSTRAP_SEED).integers(0,24,size=(BOOTSTRAP_RESAMPLES,24))
    contrasts=[];pairs=[];cells=[];refinement=[];interaction=[]
    def values(h,method,mode):
        keys=[(h,method,mode,e) for e in episodes]
        return np.array([records[k]['episode_cost'] for k in keys]) if all(k in records for k in keys) else None
    def add_contrast(h,label,a,b):
        if a is None or b is None:return
        contrasts.append({'h_ms':h,'contrast_B_minus_A':label,**paired(a,b,idx)})
        pairs.extend({'h_ms':h,'contrast':label,'episode':e,'A':float(x),'B':float(y),'B_minus_A':float(y-x)}
                     for e,x,y in zip(episodes,a,b))
    for h in D_STEPS:
        for method in METHODS:
            z=values(h,method,'zero');d=values(h,method,'delayed')
            add_contrast(h,f'{method}: delayed - zero',z,d)
            for mode in MODES:
                v=values(h,method,mode)
                if v is None:continue
                ms=[records[(h,method,mode,e)] for e in episodes]
                cells.append({'h_ms':h,'method':method,'label':LABELS[method],'mode':mode,'n':24,'mean_cost':float(v.mean()),
                              'std_cost':float(v.std(ddof=1)),
                              **{k+'_total':sum(m[k] for m in ms) for k in
                                 ('missed_deadlines','busy_skipped_ticks','inference_failures','new_violation','violation_s','fallback_s')},
                              'max_temperature_c':max(m['max_temperature_c'] for m in ms)})
        for mode in MODES:
            shared=values(h,'shared_physics',mode)
            for comparator in ('small_physics','pino_mpc'):
                add_contrast(h,f'{mode}: Shared - {comparator}',values(h,comparator,mode),shared)
    for coarse,fine in zip(D_STEPS,D_STEPS[1:]):
        for method in METHODS:
            for mode in MODES:
                a=values(coarse,method,mode);b=values(fine,method,mode)
                if a is not None and b is not None:
                    refinement.append({'coarse_ms':coarse,'fine_ms':fine,'method':method,'mode':mode,**paired(a,b,idx)})
        # Difference of paired contrasts, in cost units. Avoid relative ratios
        # whose denominators (contrasts near zero) would be unstable.
        for label,parts in [('FV-MPC delay',('fv_mpc','delayed','fv_mpc','zero')),
                            ('Shared-Small zero',('shared_physics','zero','small_physics','zero')),
                            ('Shared-Small delayed',('shared_physics','delayed','small_physics','delayed'))]:
            bm,bmode,am,amode=parts
            arrays=[values(h,m,mode) for h in (coarse,fine) for m,mode in ((bm,bmode),(am,amode))]
            if any(x is None for x in arrays):continue
            c=arrays[0]-arrays[1];f=arrays[2]-arrays[3];diff=f-c
            lo,hi=np.quantile(diff[idx].mean(axis=1),[.025,.975])
            interaction.append({'contrast':label,'coarse_ms':coarse,'fine_ms':fine,'n':24,
                                'coarse_mean':float(c.mean()),'fine_mean':float(f.mean()),
                                'fine_minus_coarse':float(diff.mean()),'ci_low':float(lo),'ci_high':float(hi),
                                'direction_changed':direction(c.mean())!=direction(f.mean())})
    gates={name:(read_json(output/name).get('passed',False) if (output/name).exists() else False)
           for name in ('fixed_baseline_gate.json','D_baseline_gate.json','INPUT_INTEGRITY.json')}
    execution=read_json(output/'STATUS.json').get('state') if (output/'STATUS.json').exists() else 'UNKNOWN'
    complete=len(flat)==732 and not missing and not problems and all(gates.values()) and execution=='EXECUTION_COMPLETE'
    result={'status':'COMPLETE_VERIFIED' if complete else 'PARTIAL_OR_UNVERIFIED','execution_state':execution,
            'completed_audited_cases':len(flat),'expected_cases':732,'D_cases':len(records),'C_cases':len(fixed),
            'missing_cases':missing,'problems':problems,'gates':gates,
            'independent_metric_audit_passed':not problems and all(x['passed'] for x in audit),
            'statistics':protocol['design']['statistics'],'fixed_replay_comparisons':fixed_diffs,
            'D_cells':cells,'D_contrasts':contrasts,'D_refinement':refinement,'primary_contrast_changes':interaction,
            'interpretation':'Post hoc numerical robustness of frozen systems. C has one episode; D has 24 paired normal episodes. Repeated settings are not independent samples. No automatic convergence or publication-acceptance claim.'}
    if complete:
        result['primary_direction_changes_observed']=any(x['direction_changed'] for x in interaction)
    write_json(dest/'summary.json',result);write_json(dest/'metric_audit.json',audit)
    csv_file(dest/'all_cases.csv',flat);csv_file(dest/'fixed_replay.csv',fixed_diffs)
    csv_file(dest/'D_cells.csv',cells);csv_file(dest/'D_contrasts.csv',contrasts)
    csv_file(dest/'D_paired_episodes.csv',pairs);csv_file(dest/'D_refinement.csv',refinement)
    csv_file(dest/'primary_contrast_changes.csv',interaction)
    lines=['# Extended numerical validation', '',f"Status: **{result['status']}**; audited {len(flat)}/732 runs (D {len(records)}/720, C {len(fixed)}/12).",'',
           'This is a post hoc sensitivity analysis of already selected models. The independent unit is the episode (D n=24, C n=1), not a run, control tick, or integration step.', '',
           'Only actual-plant integration is refined. Controller prediction, sensor clocks, and original endpoint cost/peak sampling remain fixed. Latencies are archived conditional replay, not newly measured system E2E latency.', '',
           '## Reproduction and integrity','',*[f'- {k}: {v}' for k,v in gates.items()],
           f"- Independent metric audit: {result['independent_metric_audit_passed']}",'',
           '## Fixed-command replay','',
           '| Delay ms | h coarse → fine ms | Cost change % | Field trajectory RMSE K (1 Hz) | Final field RMSE K |',
           '|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['delay_ms']:g} | {r['coarse_ms']:g} → {r['fine_ms']:g} | {r['cost_relative_pct']:.6g} | {r['field_trajectory_rmse_k_1hz']:.6g} | {r['final_field_rmse_k']:.6g} |" for r in fixed_diffs)
    lines+=['','## Paired D comparisons','','Rows are shown only for complete 24-episode pairs. Intervals are percentile bootstrap 95% intervals conditional on these frozen systems and latency replay. They are descriptive, without multiple-comparison correction. Negative B−A means B has lower cost.','',
            '| h ms | Contrast B−A | Mean difference | 95% interval | Ratio of means change % |',
            '|---:|:---|---:|:---|---:|']
    lines.extend(f"| {r['h_ms']:g} | {r['contrast_B_minus_A']} | {r['mean_B_minus_A']:.6g} | [{r['difference_ci_low']:.6g}, {r['difference_ci_high']:.6g}] | {r['ratio_of_means_pct']:.5g} |" for r in contrasts)
    lines+=['','## Interpretation limits','',
            'Assess fixed-command trajectory differences and closed-loop contrast changes together. Small fixed-command errors can still change candidate choices in feedback. Preserved contrast signs alone do not establish numerical convergence. All adverse outcomes remain in the tables.', '',
            'The original 10 ms/event cost quadrature and peak sampling were held fixed; this study does not establish continuous-time peak bounds or convergence of that quadrature. No spatial-grid, new-training-seed, OOD, or hardware generalization claim follows.', '',
            'See D_refinement.csv for per-method h sensitivity, primary_contrast_changes.csv for changes of the primary paired effects, D_cells.csv for deadlines/safety outcomes, and D_paired_episodes.csv for each episode.', '',
            f'Missing cases: {len(missing)}. Audit problems: {len(problems)}.']
    if not complete:lines+=['','**The full protocol is incomplete or unverified. Do not use these partial tables as a complete-cohort result.**']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Analysis: {dest/'REPORT.md'} ({result['status']})",flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    analyze(parser.parse_args().output)
