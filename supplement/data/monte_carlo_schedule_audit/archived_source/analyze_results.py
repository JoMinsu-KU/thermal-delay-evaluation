#!/usr/bin/env python3
"""Independent saved-record audits and conditional Monte Carlo summaries."""
from pathlib import Path
import argparse
import gzip
import json
import math
import numpy as np
from validation_io import read_json, read_trace, require, sha256, completed_case, write_json, object_hash, verify_seal
from metric_audit import truth, audit_case, csv_file
from latency_replay import ReplaySampler
from mc_statistics import summarize


def final_input_integrity(output):
    """A complete case set cannot override a missing or failed final inventory."""
    path=Path(output)/'INPUT_INTEGRITY.json'
    if not path.is_file():
        return {'passed':False,'available':False,'reason':'Final input-integrity receipt is missing'}
    try:
        receipt=read_json(path)
        passed=(receipt.get('passed') is True and receipt.get('failures')==[]
                and isinstance(receipt.get('checked_files'),int) and receipt['checked_files']>0)
        return {'passed':passed,'available':True,'receipt':receipt}
    except (OSError,ValueError,TypeError,AttributeError) as exc:
        return {'passed':False,'available':True,'reason':f'Invalid final input-integrity receipt: {exc}'}


def final_state(verified_cases,expected_cases,problems,complete_replicates,expected_replicates,integrity):
    complete=(verified_cases==expected_cases and not problems
              and complete_replicates==expected_replicates and integrity['passed'])
    return 'COMPLETE_VERIFIED' if complete else 'INCOMPLETE_OR_FAILED'


def load_pools(output):
    pools={}
    for folder in sorted((output/'timing_pools').glob('*')):
        if not (folder/'COMPLETE').exists():continue
        meta=verify_seal(folder)['metadata']
        with gzip.open(folder/'rows.json.gz','rt',encoding='utf-8') as stream:rows=json.load(stream)
        require(meta['method']==folder.name and meta['pool_object_sha256']==object_hash(rows),'Timing pool content mismatch')
        pools[folder.name]=rows
    return pools


def replay_audit(folder, condition, trace, pool, seed):
    if condition['mode']=='zero':return None
    with gzip.open(folder/'replay_rows.json.gz','rt',encoding='utf-8') as stream:selected=json.load(stream)
    requests=[r for r in trace if r['event']=='request']
    require(len(selected)==len(requests),'Replay identity row count mismatch')
    sampler=None
    if condition['sequence']!='legacy':
        sampler=ReplaySampler(pool,condition['sequence'],None,seed=seed,method=condition['method'],episode=condition['episode'],replicate=condition['replicate'])
        plan=sampler.preflight()
        require(read_json(folder/'draw_plan.json')==plan,'Potential draw plan is not reproducible from frozen inputs')
    for request,item in zip(requests,selected):
        require(item['request']==request['request'] and item['sequence']==condition['sequence'],'Replay request identity mismatch')
        require(item['source_request']==request['trace_request'] and item['e2e_ms']==request['delay_ms'],'Consumed latency differs')
        if sampler is not None:
            sampler.choose(request,request['request'])
            require(item==sampler.records[-1],'Resampled source row, pool match, or hash differs from deterministic plan')
            require(item['token_gap']==request['trace_token_gap'],'Token gap differs')
    return {'plan_sha256':plan['draw_plan_sha256'], 'selected':selected, 'requests':requests} if sampler else None


def analyze(output):
    output=Path(output);protocol=read_json(output/'protocol.json');ph=sha256(output/'protocol.json')
    design=protocol['design'];episodes=design['normal_episodes'];require(len(set(episodes))==24,'Episode roster changed')
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    pools=load_pools(output)
    records={};flat=[];audits=[];missing=[];problems=[];events=[];pairs={}
    for job in design['jobs']:
        for case in job['cases']:
            key=case['key']
            try:
                folder=completed_case(output,key,ph)
                if folder is None:missing.append(key);continue
                condition={**{k:v for k,v in job.items() if k!='cases'},**case}
                require(read_json(folder/'condition.json')==condition,'Condition differs from protocol')
                m=read_json(folder/'metrics.json');trace=read_trace(folder/'trace.jsonl.gz');rows=truth(folder/'truth.csv.gz')
                check=audit_case(folder,m,trace,rows);require(check['passed'],json.dumps(check))
                replay=replay_audit(folder,condition,trace,pools.get(job['method']),design['seed'])
                if job['phase']=='baseline':require(read_json(folder/'baseline_comparison.json')['passed'],'Baseline failed')
                if job['phase']=='monte_carlo' and job['method']=='fv_mpc':
                    pairs.setdefault((job['replicate'],case['episode']),{})[job['fallback_policy']]=replay
                with np.load(folder/'quadrature.npz',allow_pickle=False) as q:
                    tracking=float(np.sum(q['mse_k2']*q['dt_ns'].astype(float)/1e9)/60/10000)
                field=np.load(folder/'final_T64.npy',allow_pickle=False)
                comp=dict(J_tracking=tracking,J_terminal=.5*float(np.mean(((field-rows[0]['goal_c'])/100)**2)),J_command=.002*m['action_effort'],J_increment=.02*m['action_delta_mean_square'])
                require(math.isclose(sum(comp.values()),m['episode_cost'],abs_tol=1e-10,rel_tol=1e-8),'Cost components differ')
                record={**condition,**m,**comp}
                previous=np.zeros(4)
                for event in trace:
                    if event['event'] not in ('apply','deadline_fallback'):continue
                    command=np.asarray(event['command'])
                    if event['event']=='deadline_fallback':
                        target=np.zeros(4) if job['fallback_policy']=='zero' else previous
                        require(np.array_equal(command,target),'Deadline command differs from policy')
                        events.append(dict(case=key,episode=case['episode'],replicate=job['replicate'],method=job['method'],fallback_policy=job['fallback_policy'],request=event['request'],time_s=event['time_ns']/1e9,previous_u_sum=float(previous.sum()),applied_u_sum=float(command.sum())))
                    previous=command
                records[(job['sequence'],job['fallback_policy'],job['mode'],job['method'],case['episode'])]=record
                flat.append(record);audits.append({'case':key,**check,'replay_identity_checked':True})
            except Exception as exc:problems.append({'case':key,'error':repr(exc)})
    coupling=[]
    for (rep,episode),p in pairs.items():
        if set(p)!={'zero','hold'}:continue
        a,b=p['zero'],p['hold']
        sameplan=a['plan_sha256']==b['plan_sha256']
        if not sameplan:problems.append(dict(replicate=rep,episode=episode,error='FV potential draw plan differs'))
        common=min(len(a['selected']),len(b['selected']))
        divergent=[i for i in range(common) if a['selected'][i]!=b['selected'][i]]
        timing_keys=('snapshot_ns','ready_ns','compute_ready_ns','deadline_ns','expired','delay_ms')
        clock_equal=len(a['requests'])==len(b['requests']) and all(all(x[k]==y[k] for k in timing_keys) for x,y in zip(a['requests'],b['requests']))
        coupling.append(dict(replicate=rep,episode=episode,potential_draw_plan_equal=sameplan,zero_requests=len(a['selected']),hold_requests=len(b['selected']),common_request_path_or_token_divergences=len(divergent),first_divergent_request=divergent[0] if divergent else None,consumed_clocks_equal=clock_equal))
    stats=summarize(records,episodes,design['replicates'])
    for name,rows in [('episode_metrics',flat),('deadline_events',events),('coupling_audit',coupling),('replicate_summary',stats['cells']),('ranking_probabilities',stats['probabilities']),('cost_effect_distribution',stats['distributions']),('paired_fv_effects',stats['effects'])]:csv_file(dest/(name+'.csv'),rows)
    write_json(dest/'method_rankings.json',{'baseline':stats.get('baseline'),'replicates':stats['rankings']})
    write_json(dest/'metric_audit.json',dict(passed=not problems,checked_cases=len(audits),cases=audits,problems=problems))
    integrity=final_input_integrity(output)
    state=final_state(len(flat),design['total_runs'],problems,stats['complete_replicates'],design['replicates'],integrity)
    summary=dict(state=state,expected_cases=design['total_runs'],completed_verified_cases=len(flat),missing_cases=missing,problems=problems,independent_metric_audit_passed=not problems,final_input_integrity_passed=integrity['passed'],final_input_integrity=integrity,complete_replicates=stats['complete_replicates'],expected_replicates=design['replicates'],fv_pairs_with_identical_consumed_clocks=sum(c['consumed_clocks_equal'] for c in coupling),fv_complete_pairs=len(coupling),scientific_positive_effect_required=False,statistics_scope=design['statistics'])
    write_json(dest/'summary.json',summary)
    text=['# Empirical latency Monte Carlo results','',f"State: **{summary['state']}**",'',f"Verified cases: {len(flat)}/{design['total_runs']}; complete replicates: {stats['complete_replicates']}/{design['replicates']}.",'','One replicate comprises the same 24 fixed nominal episodes with independent latency draws per episode and method. FV zero and hold share potential draw plans; coupling_audit.csv reports whether the realized timing paths remained identical.','All probabilities and Wilson intervals are conditional on the fixed timing pools, IID assumption, episodes and checkpoints. IID destroys recorded temporal correlation. Intervals quantify Monte Carlo sampling precision, not uncertainty over hardware, new plants, training seeds or the empirical timing distribution.','Partial-run statistics are provisional and must not replace the frozen complete design.','See ranking_probabilities.csv (including strict pairwise reversals), method_rankings.json (five-method rankings and ties), cost_effect_distribution.csv (replicate spread, not confidence intervals), paired_fv_effects.csv, and every raw episode in episode_metrics.csv.','No safety guarantee or positive-result requirement is imposed.']
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    result=analyze(parser.parse_args().output);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['state']=='COMPLETE_VERIFIED' else 1)
