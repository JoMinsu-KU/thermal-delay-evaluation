"""Causal, serial SIL with measured delay and explicit OOD compute/apply injection."""
import json
import time
import numpy as np
import torch
from ..common import require,read_json,write_json,atomic_text,seal,verify,setup_torch,StopRequest
from ..control import choose_latency
from .contracts import entries,assert_frozen,ABLATIONS
from .data import Plant
from .engine import Engine
from .latency import load_trace
from .evaluation import cohort


def injection(entry):
    if entry.get('ood_group')!='delay':return 0.,0.
    parts=entry['ood_condition'].split('_');return float(parts[1]),float(parts[3])


def selected_entries(run,cfg,method):
    rows=entries(run,['test','ood'])
    limit=cfg['budget']['ablation_loop_per_split'] if method in ABLATIONS else cfg['budget']['original_loop_per_split'] if method=='native_original' else None
    if limit is None:return rows
    chosen=[];counts={}
    for row in rows:
        key=cohort(row);counts.setdefault(key,0)
        if counts[key]<limit:chosen.append(row);counts[key]+=1
    return chosen


def simulate(plant,engine,mode,latency=None,stop=None):
    engine.reset();trace=[];pending=None;busy=0;requests=missed=skipped=failed=slew=0
    extra_compute,extra_apply=injection(plant.entry)
    def until(end):
        nonlocal pending,slew
        if pending is not None and pending['event_ns']<=end:
            if pending['expired']:command=np.zeros(4);fallback=True
            else:
                command=np.array(pending['decision']['command']);fallback=pending['decision']['fallback_reason'] is not None
                if not fallback and np.any(abs(command-plant.u)>.04000001):command=np.zeros(4);fallback=True;slew+=1
            unchanged=bool(np.array_equal(command,plant.u) and fallback==plant.fallback)
            if not unchanged:plant.advance_to(pending['event_ns']);plant.apply(command,fallback)
            trace.append({'event':'deadline_fallback' if pending['expired'] else 'apply','time_ns':pending['event_ns'],
                          'request':pending['request'],'fallback':fallback,'command':command.tolist(),'no_change':unchanged})
            pending=None
        plant.advance_to(end)
    for step in range(round(plant.duration_s*10)):
        now=step*100_000_000;until(now)
        if now<busy:
            skipped+=1;trace.append({'event':'busy_skip','time_ns':now,'ready_ns':busy});continue
        decision=engine.decide(lambda:plant.raw().snapshot_ns(now))
        if mode=='delayed':delay,ref,gap=choose_latency(latency,decision,requests,strict=True)
        else:delay=0.;ref=None;gap=None
        compute_ready=now+int(np.ceil((delay+extra_compute)*1e6))
        ready=compute_ready+int(extra_apply*1e6);deadline=now+100_000_000;expired=ready>deadline
        missed+=int(expired);pending={'request':requests,'event_ns':min(ready,deadline),'expired':expired,'decision':decision}
        busy=compute_ready
        trace.append({'event':'request','request':requests,'snapshot_ns':now,'ready_ns':ready,'compute_ready_ns':compute_ready,
                      'deadline_ns':deadline,'expired':expired,'delay_ms':delay,'injected_compute_ms':extra_compute,
                      'injected_apply_ms':extra_apply,'trace_request':ref,'trace_token_gap':gap,**decision})
        requests+=1;until(min((step+1)*100_000_000,plant.end_ns))
        if stop is not None and stop.signal:raise InterruptedError('Current episode restarts; completed episodes are retained')
    until(plant.end_ns)
    return {**plant.metrics(),'requests':requests,'missed_deadlines':missed,'busy_skipped_ticks':skipped,
            'inference_failures':failed,'delayed_slew_rejections':slew,'mode':mode,'deadline_ms':100,
            'injected_compute_ms':extra_compute,'injected_apply_ms':extra_apply,
            'trace_token_gap_max':max((r['trace_token_gap'] for r in trace if r['event']=='request' and r['trace_token_gap'] is not None),default=None),
            'filter_interventions':sum(bool(r.get('filter_intervention')) for r in trace)},trace


def run_method(root,run,cfg,output,method,mode):
    setup_torch(cfg);assert_frozen(run);rows=[];engine=None;stop=StopRequest()
    latency=load_trace(run,cfg,method) if mode=='delayed' else None
    try:
        for entry in selected_entries(run,cfg,method):
            folder=output/method/entry['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE').exists():verify(folder);metric=read_json(folder/'metrics.json')
            else:
                if engine is None:engine=Engine(root,run,method)
                plant=Plant(run/'02_generate/data',entry,cfg['budget']['duration_s'])
                started=time.perf_counter();metric,trace=simulate(plant,engine,mode,latency,stop)
                metric['wall_seconds']=time.perf_counter()-started
                write_json(folder/'metrics.json',metric)
                atomic_text(folder/'control_trace.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in trace))
                seal(folder,{'method':method,'episode':entry['episode_id'],'mode':mode})
            rows.append({'episode':entry['episode_id'],'cohort':cohort(entry),'stratum':entry['stratum'],
                         'ood_condition':entry['ood_condition'],'method':method,**metric})
            print(f'L2 {mode} {method} {len(rows)}/{len(selected_entries(run,cfg,method))}',flush=True)
        write_json(output/(method+'.json'),rows);return {'method':method,'mode':mode,'episodes':len(rows)}
    finally:
        if engine is not None:engine.close()
        stop.close()
