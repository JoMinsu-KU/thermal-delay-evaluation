"""PV2--PV4 orchestration: isolated models, live inference, latency and closed loops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .common import ROOT, require, read_json, write_json, digest, seal, verify, setup_torch, StopRequest, atomic_text
from .evaluation import EvaluationData, Journal, decision_metrics, summarize_decisions, paired, quantiles
from .control import DecisionEngine, LivePlant, freeze_request, run_episode, METHODS


def identity(run,config):
    return {'plan':digest(run/'15_eval_data/plan.json'),'protocol':config['validation']}


def policy_regret(root,run,config,output,device='cuda'):
    setup_torch(config,device); journals={}; stop=StopRequest()
    try:
        for method in ['no_physics','physics']:
            for checkpoint in config['validation'].get('policy_checkpoints',['baseline','selected','last']):
                key=method+'_'+checkpoint
                journal=Journal(output/(key+'.jsonl'),{**identity(run,config),'method':key}); journals[key]=journal
                data=EvaluationData(run); engine=None
                try:
                    expected=len(data.entries)*len(data.plan['anchor_steps'])
                    if len(journal.rows)==expected: continue
                    engine=DecisionEngine(root,run,method,checkpoint,device)
                    for i,(entry,step,obs,labels) in enumerate(data.rows()):
                        if i<len(journal.rows): continue
                        result=engine.decide(lambda:obs)
                        metric=decision_metrics(result['choice'],obs,labels,config['acceptance']['solver']['significant_cost_gap'])
                        journal.append({'episode':entry['episode_id'],'stratum':entry['stratum'],
                                        'cohort':entry.get('cohort','pilot_eval'),'step':step,
                                        'time_band':'early' if step<100 else ('transient' if step<300 else 'late'),
                                        'methods':{key:metric},'tokens':result['tokens']})
                        if (i+1)%32==0: print(f'PV2 {key} {i+1}/{expected}',flush=True)
                        if stop.signal: raise InterruptedError('PV2 committed decision; resume supported')
                finally:
                    data.close()
                    if engine: engine.close()
    finally: stop.close()
    keys=list(journals); rows=[]
    for i,base in enumerate(journals[keys[0]].rows):
        row={k:v for k,v in base.items() if k!='methods'}; row['methods']={}
        for key in keys:
            other=journals[key].rows[i]
            require((base['episode'],base['step'])==(other['episode'],other['step']),'Unpaired evaluation rows')
            row['methods'].update(other['methods'])
        rows.append(row)
    summaries={key:summarize_decisions(rows,key) for key in keys}
    comparison=paired(rows,'physics_selected','no_physics_selected','regret',config)
    a=summaries['physics_selected']; b=summaries['no_physics_selected']
    expected=sum(r['methods']['physics_selected']['expert_feasible'] for r in rows)
    complete=comparison['common_rows']==expected
    safe=a['infeasible']==0 and a['failure']<=b['failure'] and complete
    status=comparison['interpretation'] if safe else ('NEGATIVE' if a['failure']>b['failure'] else 'INCONCLUSIVE')
    stratified={}
    for attribute in ['stratum','time_band','cohort']:
        for value in sorted({r[attribute] for r in rows}):
            subset=[r for r in rows if r[attribute]==value]
            stratified[attribute+':'+value]={'paired':paired(subset,'physics_selected','no_physics_selected','regret',config),
                                             'methods':{k:summarize_decisions(subset,k) for k in keys}}
    denominator=b['regret']['mean']; numerator=a['regret']['mean']
    result={'PV2':status,'paired':comparison,'complete_common_expert_feasible_coverage':complete,'methods':summaries,
            'relative_mean_regret_reduction':1-numerator/denominator if complete and denominator is not None and denominator>1e-6 and numerator is not None else None,
            'stratified':stratified,'rows':len(rows),'test_accessed':False,
            'note':'Same predicted PINO safety gate; only scorer physics text differs. CIs use episodes, not decisions.'}
    write_json(output/'regret_comparison.json',result); return result


def system_counters():
    result={}
    for name in ['cpu.stat','memory.events','memory.current']:
        path=Path('/sys/fs/cgroup')/name
        if path.exists(): result[name]=path.read_text().strip()
    return result


def thermal_is_fresh(age_s,limit_s=.1):
    # Ages are FP32 seconds. 100ms is stored as 0.10000000149, not exactly 0.1.
    return float(age_s)<=limit_s+1e-8


def violation_nonworsening(left,right,time_tolerance_s=1e-8):
    return (all(left[k]<=right[k] for k in ['violation_episodes','new_violation_episodes','inference_failures'])
            and left['violation_s']<=right['violation_s']+time_tolerance_s)


def make_ring(run,config):
    data=EvaluationData(run); ring=[]
    try:
        candidates_list=[(entry,int(step)) for entry in data.entries for step in data.plan['anchor_steps']]
        take=np.linspace(0,len(candidates_list)-1,min(len(candidates_list),config['validation']['latency_ring_size']),dtype=int)
        for i in take:
            entry,step=candidates_list[int(i)]; raw,_=data.episode(entry)
            frozen=freeze_request(raw,step)
            ring.append((frozen,step,entry['stratum'],entry['episode_id']))
    finally: data.close()
    return ring


def latency_benchmark(root,run,config,output,device='cuda',cold_process=True):
    setup_torch(config,device); cfg=config['validation']; ring=make_ring(run,config); summaries={}; stop=StopRequest()
    try:
        for method in METHODS:
            reports=[]
            for repeat in range(cfg['latency_repeats']):
                journal=Journal(output/f'{method}_{repeat}.jsonl',{**identity(run,config),'method':method,'repeat':repeat})
                session_path=output/f'{method}_{repeat}_sessions.json'; sessions=read_json(session_path) if session_path.exists() else []
                require(len(journal.rows)<=cfg['latency_requests'],'Too many latency samples')
                engine=None
                if len(journal.rows)<cfg['latency_requests']:
                    begin=time.perf_counter(); engine=DecisionEngine(root,run,method,device=device)
                    load_s=time.perf_counter()-begin
                    sessions.append({'start_request':len(journal.rows),'system_before':system_counters(),'model_load_s':load_s,
                                     'warmup_requests':cfg['warmup_requests'],'distinct_inputs':len(ring)})
                    write_json(session_path,sessions)
                    try:
                        for i in range(cfg['warmup_requests']):
                            raw,step,_,_=ring[i%len(ring)]; engine.reset()
                            engine.decide(lambda:raw.snapshot_ns(step*100_000_000))
                            if stop.signal: raise InterruptedError('Latency warmup interrupted')
                        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
                        for i in range(len(journal.rows),cfg['latency_requests']):
                            raw,step,stratum,episode=ring[i%len(ring)]; engine.reset()
                            started=time.perf_counter_ns()
                            try:
                                result=engine.decide(lambda:raw.snapshot_ns(step*100_000_000)); result['error']=None
                            except (ValueError,RuntimeError) as exc:
                                if isinstance(exc,torch.cuda.OutOfMemoryError): raise
                                result={'error':str(exc),'e2e_ms':(time.perf_counter_ns()-started)/1e6,'gpu_compute_ms':None,
                                        'fallback_reason':'inference_error','tokens':0}
                            journal.append({'request':i,'episode':episode,'step':step,'stratum':stratum,
                                            'session':len(sessions)-1,'deadline_miss':result['e2e_ms']>cfg['deadline_ms'],**result})
                            if (i+1)%100==0: print(f'PV4 {method} repeat={repeat} {i+1}/{cfg["latency_requests"]}',flush=True)
                            require(not (len(journal.rows)>=3 and all(r['error'] for r in journal.rows[-3:])), 'Three consecutive inference failures')
                            if stop.signal: raise InterruptedError('Latency request committed; resumed segment repeats warmup')
                        sessions[-1]['system_after']=system_counters()
                        if device.startswith('cuda'):
                            sessions[-1]['peak_cuda_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                            sessions[-1]['peak_cuda_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
                        write_json(session_path,sessions)
                    finally: engine.close()
                rows=journal.rows; good=[r for r in rows if not r['error']]
                report={'requests':len(rows),'successful_requests':len(good),'failures':len(rows)-len(good),
                        'local_e2e_ms':quantiles([r['e2e_ms'] for r in good]),'gpu_compute_ms':quantiles([r['gpu_compute_ms'] for r in good]),
                        'deadline_miss_count':sum(r['deadline_miss'] or bool(r['error']) for r in rows),
                        'deadline_miss_fraction':sum(r['deadline_miss'] or bool(r['error']) for r in rows)/len(rows),
                        'native_scored_requests':sum(r['tokens']>0 for r in good),
                        'scored_e2e_ms':quantiles([r['e2e_ms'] for r in good if r['tokens']>0]),
                        'fallback_e2e_ms':quantiles([r['e2e_ms'] for r in good if r['fallback_reason']]),
                        'sessions':sessions,'quantile_method':'numpy linear; successful completed calls; errors reported separately',
                        'distinct_inputs':len(ring),'thermal_fresh_e2e_ms':quantiles([r['e2e_ms'] for r in good if thermal_is_fresh(r['thermal_age_s'])]),
                        'thermal_older_e2e_ms':quantiles([r['e2e_ms'] for r in good if not thermal_is_fresh(r['thermal_age_s'])])}
                reports.append(report)
            summaries[method]=reports
            # True new-process start-to-first-decision measurement, separate from warm latency.
            cold_file=output/(method+'_cold.json')
            if cold_process and device.startswith('cuda') and method in ['no_physics','physics'] and not cold_file.exists():
                started=time.perf_counter_ns()
                command=[sys.executable,'-m','ojpoc.pilot','--cold','--root',str(root),'--run',str(run),
                         '--output',str(cold_file),'--method',method]
                subprocess.run(command,check=True,timeout=240)
                cold=read_json(cold_file); cold['process_start_to_first_result_ms']=(cold['first_result_monotonic_ns']-started)/1e6
                cold['process_start_to_exit_ms']=(time.perf_counter_ns()-started)/1e6
                write_json(cold_file,cold)
    finally: stop.close()
    result={'methods':summaries,'deadline_ms':cfg['deadline_ms'],'warmup_requests':cfg['warmup_requests'],
            'requested_repeats':cfg['latency_repeats'],'repeat_variability_measured':cfg['latency_repeats']>1,
            'request_batch':1,'candidate_batch':9,'online_features':True,'token_cache':False,'prefix_cache':False,
            'disk_and_network_in_timed_region':False,'profile':config['profile'],
            'full_latency_budget_measured':config['profile']!='smoke','hard_realtime_guarantee':False,
            'hard_request_timeout_ms':None,'deadline_is_not_cancellation':True,
            'pi_integrator_reset_per_benchmark_request':True,
            'test_accessed':False,'scope':'warm isolated local inference; cold process measurements in *_cold.json'}
    write_json(output/'latency_summary.json',result); return result


def closed_loop(root,run,config,output,mode='zero',device='cuda'):
    setup_torch(config,device); data=EvaluationData(run)
    entries=[e for e in data.entries if e['closed_loop']]; data.close(); rows_by_episode={}
    stop=StopRequest()
    try:
        for method in METHODS:
            engine=None
            try:
                latency=None
                if mode=='delayed':
                    if config['validation'].get('strict_latency_matching'):
                        from .recheck_latency import load_trace
                        latency=load_trace(run,config,method)
                    else:
                        latency=[]
                        for repeat in range(config['validation']['latency_repeats']):
                            journal=Journal(run/'19_latency'/f'{method}_{repeat}.jsonl',{**identity(run,config),'method':method,'repeat':repeat})
                            latency.extend(journal.rows)
                        require(len(latency)==config['validation']['latency_requests']*config['validation']['latency_repeats'],'Incomplete latency trace')
                for entry in entries:
                    directory=output/method/entry['episode_id']
                    if (directory/'COMPLETE').exists(): verify(directory); metrics=read_json(directory/'metrics.json')
                    else:
                        if engine is None: engine=DecisionEngine(root,run,method,device=device)
                        directory.mkdir(parents=True,exist_ok=True)
                        plant=LivePlant(run/'15_eval_data/data',entry,config['validation']['duration_s'])
                        started=time.perf_counter()
                        metrics,trace=run_episode(plant,engine,mode,latency,stop,config['validation']['deadline_ms'],
                                                  config['validation'].get('strict_latency_matching',False))
                        gaps=[r['trace_token_gap'] for r in trace if r['event']=='request' and r['trace_token_gap'] is not None]
                        metrics['trace_token_gap']=quantiles(gaps)
                        metrics['wall_seconds']=time.perf_counter()-started
                        write_json(directory/'metrics.json',metrics)
                        atomic_text(directory/'control_trace.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in trace))
                        seal(directory,{'method':method,'episode':entry['episode_id'],'mode':mode,**identity(run,config)})
                    row=rows_by_episode.setdefault(entry['episode_id'],{'episode':entry['episode_id'],'stratum':entry['stratum'],
                                                                       'cohort':entry.get('cohort','pilot_eval'),'methods':{}})
                    row['methods'][method]=metrics
                    if stop.signal: raise InterruptedError('Completed SIL episode committed')
            finally:
                if engine: engine.close()
    finally: stop.close()
    result=summarize_closed_loop(list(rows_by_episode.values()),config,mode)
    write_json(output/('closed_loop_'+mode+'.json'),result); return result


def summarize_closed_loop(rows,config,mode):
    require(bool(rows),'No completed closed-loop episodes')
    summary={}
    for method in METHODS:
        metrics=[row['methods'][method] for row in rows]
        summary[method]={'episodes':len(metrics),'mean_episode_cost':float(np.mean([r['episode_cost'] for r in metrics])),
                         'mean_tracking_rmse_k':float(np.mean([r['tracking_rmse_k'] for r in metrics])),
                         'max_overshoot_k':max(r['overshoot_k'] for r in metrics),
                         'violation_episodes':sum(r['violation'] for r in metrics),'violation_s':sum(r['violation_s'] for r in metrics),
                         'new_violation_episodes':sum(r['new_violation'] for r in metrics),
                         'fallback_s':sum(r['fallback_s'] for r in metrics),'inference_failures':sum(r['inference_failures'] for r in metrics),
                         'filter_interventions':sum(r['filter_interventions'] for r in metrics),
                         'missed_deadlines':sum(r['missed_deadlines'] for r in metrics)}
    comparison=paired(rows,'physics','no_physics','episode_cost',config)
    safe=violation_nonworsening(summary['physics'],summary['no_physics'])
    status=comparison['interpretation'] if safe else 'NEGATIVE'
    baseline=summary['no_physics']['mean_episode_cost']; margin=.05*baseline
    references={}
    for reference in ['pi','fv_mpc','pino_mpc','no_physics']:
        diff=paired(rows,'physics',reference,'episode_cost',config)
        guard=violation_nonworsening(summary['physics'],summary[reference])
        allowance=.05*summary[reference]['mean_episode_cost']
        references[reference]={'paired_cost':diff,'margin_absolute':allowance,'violation_failure_nonworsening':guard,
                               'noninferior_5pct':bool(diff['ci95'] and diff['ci95'][1]<=allowance and guard)}
    result={'PV3':status,'mode':mode,'methods':summary,'paired_episode_cost':comparison,'noninferior_5pct':bool(
            comparison['ci95'] and comparison['ci95'][1]<=margin and safe),
            'noninferiority_reference':'no_physics','reference_controller_comparisons':references,
            'noninferiority_margin_absolute':margin,'normal_stress':{s:{m:quantiles([r['methods'][m]['episode_cost'] for r in rows if r['stratum']==s]) for m in METHODS} for s in ['normal','stress']},
            'episodes':rows,'test_accessed':False,'cost_sampling':'fine FV substeps; full horizon only',
            'delayed_scope':'serial worker, measured empirical trace matched by fallback/token length; busy ticks skip requests; late results discarded' if mode=='delayed' else 'zero compute delay; actuator dynamics retained'}
    return result


def final_report(root,run,config,output):
    base=read_json(run/'13_report/result.json'); op=read_json(run/'16_operator_ablation/operator_ablation.json')
    policy=read_json(run/'17_policy_regret/regret_comparison.json')
    zero=read_json(run/'18_closed_loop/closed_loop_zero.json'); latency=read_json(run/'19_latency/latency_summary.json')
    delayed=read_json(run/'20_delayed_loop/closed_loop_delayed.json')
    pv1=op['PV1']['pino_balance_selected']['interpretation']
    deadline_met=all(r['failures']==0 and r['local_e2e_ms']['p99'] is not None and r['local_e2e_ms']['p99']<=config['validation']['deadline_ms'] for r in latency['methods']['physics'])
    pv4='POSITIVE' if deadline_met and delayed['PV3']=='POSITIVE' and config['profile']!='smoke' else 'INCONCLUSIVE'
    if not deadline_met or delayed['PV3']=='NEGATIVE': pv4='NEGATIVE'
    findings={'PV1':pv1,'PV2':policy['PV2'],'PV3':zero['PV3'],'PV4':pv4}
    result={'validation_quality_pass':base['quality_pass'],'extended_validation_completed':True,'findings':findings,
            'all_hypotheses_supported':all(v=='POSITIVE' for v in findings.values()),'profile':config['profile'],
            'test_accessed':False,'L2_complete':False,'latency_deadline_met':deadline_met}
    lines=['# L1 + pilot validation report','',f'Profile: {config["profile"]}; seed: {config["seed"]}',
           f'Execution completed: True; original validation quality: {base["quality_pass"]}',
           '', '| Question | Evidence | Interpretation |','|---|---|---|',
           f'| PV1 neural learning beyond balance | `16_operator_ablation/operator_ablation.json` | {pv1} |',
           f'| PV2 physics regret on {policy["rows"]} decisions | `17_policy_regret/regret_comparison.json` | {policy["PV2"]} |',
           f'| PV3 zero-delay closed loop | `18_closed_loop/closed_loop_zero.json` | {zero["PV3"]} |',
           f'| PV4 latency plus delayed control quality | `19_latency/latency_summary.json`, `20_delayed_loop/closed_loop_delayed.json` | {pv4} |',
           '', '| Operator (selected) | H RMSE K | Candidate delta RMSE K | Spatial deviation MSE | Peak error K |',
           '|---|---:|---:|---:|---:|']
    for name in ['balance_only','fno_balance_selected','pino_balance_selected']:
        m=op['observed_state'][name]
        lines.append(f'| {name} | {m["horizon_rmse_k"]} | {m["candidate_delta_rmse_k"]} | {m["spatial_deviation_mse"]} | {m["peak_error_k"]} |')
    lines+=['','| Policy | Mean regret | Failures | Fallbacks |','|---|---:|---:|---:|']
    for name,m in policy['methods'].items(): lines.append(f'| {name} | {m["regret"]["mean"]} | {m["failure"]} | {m["fallback"]} |')
    for label,loop in [('Zero compute delay',zero),('Measured trace delay',delayed)]:
        lines+=['',f'## {label}','','| Method | Episode cost | Tracking RMSE K | Max overshoot K | Violation episodes | Violation seconds | Fallback seconds |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for name,m in loop['methods'].items():
            lines.append(f'| {name} | {m["mean_episode_cost"]} | {m["mean_tracking_rmse_k"]} | {m["max_overshoot_k"]} | {m["violation_episodes"]} | {m["violation_s"]} | {m["fallback_s"]} |')
    lines+=['','## Warm local end-to-end latency','','| Method / repeat | p50 ms | p95 ms | p99 ms | Deadline miss fraction | Failures |','|---|---:|---:|---:|---:|---:|']
    for name,repeats in latency['methods'].items():
        for i,m in enumerate(repeats):
            q=m['local_e2e_ms']; lines.append(f'| {name} / {i} | {q["p50"]} | {q["p95"]} | {q["p99"]} | {m["deadline_miss_fraction"]} | {m["failures"]} |')
    lines+=['','All comparisons use held-out pilot development episodes, not calibration/test/OOD.',
            'Execution completion does not imply all hypotheses are positive. Smoke budgets do not certify p99 or scientific quality.',
            'CIs resample episodes; checkpoints were chosen on separate validation. One training seed; exploratory pilot inference.',
            'PINO vs FNO balance compares PDE + paired-delta objectives together. No-physics retains the same independent PINO safety gate.',
            'Latency includes online observer/PINO/tokenization/filter; excludes model loading, external network, input disk reads and logging.',
            'A single latency repeat does not measure run-to-run p99 variability. Delayed SIL is an empirical trace approximation, not real-time equipment validation.',
            'Stress initial violations and fallback duration remain in episode results. No L2 completion or probability calibration claim.']
    atomic_text(output/'report.md','\n'.join(lines)+'\n'); write_json(output/'validation_summary.json',result); return result


def cold_main():
    p=argparse.ArgumentParser(); p.add_argument('--cold',action='store_true',required=True)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--method',choices=['no_physics','physics'],required=True)
    args=p.parse_args(); config=read_json(args.run/'resolved_config.json'); setup_torch(config)
    ring=make_ring(args.run,config); raw,step,_,_=next(r for r in ring if r[0].snapshot_ns(r[1]*100_000_000).usable)
    engine=DecisionEngine(args.root,args.run,args.method)
    result=engine.decide(lambda:raw.snapshot_ns(step*100_000_000))
    write_json(args.output,{'first_decision':result,'new_process':True,'first_result_monotonic_ns':time.perf_counter_ns()})


if __name__=='__main__': cold_main()
