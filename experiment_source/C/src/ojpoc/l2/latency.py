"""Warm, path-conditional E2E in fresh processes, with request-level resume."""
import os
import time
from collections import Counter
import numpy as np
import torch
from ..common import read_json, write_json, require, digest, seal, verify, setup_torch, StopRequest
from ..data import RawEpisode
from ..control import MemoryEpisode, freeze_request
from ..evaluation import Journal, quantiles
from ..pilot import system_counters
from .contracts import entries, anchors, modality, assert_frozen
from .engine import Engine
from .observer import Observer

PATHS={'normal':None,'missing_observation':'cold_start_or_missing_observation',
       'no_predicted_feasible_candidate':'no_predicted_feasible_candidate'}


def path_of(result):
    for key,value in PATHS.items():
        if result['fallback_reason']==value:return key
    raise ValueError('Unexpected inference path')


def prepare_rings(root,run,cfg,output):
    setup_torch(cfg);assert_frozen(run);b=cfg['budget'];info={}
    candidates=[(e,int(s)) for e in entries(run,['validation']) for s in anchors(cfg)]
    for mode in ['both','sensor','thermal']:
        folder=output/'rings'/mode;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'COMPLETE').exists():verify(folder);info[mode]=read_json(folder/'ring.json')['counts'];continue
        engine=Engine(root,run,'pi');engine.mode=mode;engine.observer=Observer(run,mode)
        engine.margin=read_json(run/'16_calibration/calibration.json')['margins'][mode]['margin_k']
        groups={k:[] for k in PATHS};raw=None;previous=None
        try:
            points=np.linspace(0,len(candidates)-1,min(len(candidates),2*b['ring_size']),dtype=int)
            for index in points:
                entry,step=candidates[index]
                if previous!=entry['episode_id']:
                    if raw:raw.close()
                    raw=RawEpisode(run/'02_generate/data/raw'/entry['episode_id']);previous=entry['episode_id']
                frozen=freeze_request(raw,step);result=engine.decide(lambda:frozen.snapshot(step))
                groups[path_of(result)].append((frozen,step,entry,'natural'))
            require(groups['normal'],'No feasible validation inputs; latency normal path cannot be certified')
            base,step,entry,_=groups['normal'][0]
            for e in entries(run,['validation'])[:max(1,b['fallback_ring_size']//2)]:
                r=RawEpisode(run/'02_generate/data/raw'/e['episode_id']);cold=freeze_request(r,0);r.close()
                groups['missing_observation'].append((cold,0,e,'natural_cold_start'))
            arrays={k:v.copy() for k,v in base.arrays.items()};arrays['sensor_valid'][:,12:16]=False
            masked=MemoryEpisode(dict(base.meta),arrays)
            groups['missing_observation'].append((masked,step,entry,'synthetic_missing_feedback'))
            if not groups['no_predicted_feasible_candidate']:
                arrays={k:v.copy() for k,v in base.arrays.items()};arrays['sensor'][:,:8]=140;arrays['thermal'][:]=140
                groups['no_predicted_feasible_candidate'].append((MemoryEpisode(dict(base.meta),arrays),step,entry,'synthetic_140C_fault'))
            rows=[]
            for key,values in groups.items():
                limit=b['ring_size'] if key=='normal' else b['fallback_ring_size']
                take=np.linspace(0,len(values)-1,min(len(values),limit),dtype=int)
                for i in take:
                    raw_input,step,e,origin=values[i]
                    assert path_of(engine.decide(lambda:raw_input.snapshot(step)))==key
                    filename=f'input_{len(rows):04d}.npz';np.savez(folder/filename,**raw_input.arrays)
                    rows.append({'id':len(rows),'file':filename,'sha256':digest(folder/filename),'step':step,
                                 'episode':e['episode_id'],'meta':raw_input.meta,'path':key,'origin':origin})
            manifest={'entries':rows,'mode':mode,'counts':dict(Counter(r['path'] for r in rows)),
                      'source_split':'validation','mixture_not_production_prevalence':True}
            write_json(folder/'ring.json',manifest);seal(folder,manifest);info[mode]=manifest['counts']
        finally:
            if raw:raw.close()
            engine.close()
    return info


def identity(run,cfg,method,repeat):
    return {'frozen':digest(run/'15_freeze/frozen.json'),'calibration':digest(run/'16_calibration/calibration.json'),
            'ring':digest(run/'18_latency/rings'/modality(method)/'ring.json'),'method':method,'repeat':repeat,
            'budget':cfg['budget']}


def load_ring(run,method):
    folder=run/'18_latency/rings'/modality(method);verify(folder)
    groups={k:[] for k in PATHS}
    for row in read_json(folder/'ring.json')['entries']:
        with np.load(folder/row['file'],allow_pickle=False) as arrays:
            groups[row['path']].append((MemoryEpisode(row['meta'],{k:arrays[k] for k in arrays.files}),row))
    return groups


def summarize(rows):
    good=[r for r in rows if not r.get('error')]
    return {'requests':len(rows),'failures':len(rows)-len(good),'e2e_ms':quantiles([r['e2e_ms'] for r in good]),
            'gpu_event_ms':quantiles([r['gpu_compute_ms'] for r in good]),
            'deadline_misses':sum(bool(r.get('error')) or r['e2e_ms']>100 for r in rows),
            'distinct_inputs':len({r['input_id'] for r in rows}),
            'fresh_e2e_ms':quantiles([r['e2e_ms'] for r in good if r['thermal_age_s']<=.1+1e-8]),
            'older_e2e_ms':quantiles([r['e2e_ms'] for r in good if r['thermal_age_s']>.1+1e-8])}


def trial(root,run,cfg,output,method,repeat,launched_ns):
    setup_torch(cfg);assert_frozen(run);groups=load_ring(run,method);b=cfg['budget']
    order=[(k,i) for k in PATHS for i in range(b['latency_requests'] if k=='normal' else b['fallback_requests'])]
    journal=Journal(output/'requests.jsonl',identity(run,cfg,method,repeat));require(len(journal.rows)<=len(order),'Too many requests')
    for i,r in enumerate(journal.rows):require((r['path'],r['path_request'])==order[i],'Resume order mismatch')
    sessions=read_json(output/'sessions.json') if (output/'sessions.json').exists() else []
    engine=Engine(root,run,method);session={'pid':os.getpid(),'start_request':len(journal.rows),'before':system_counters(),'status':'RUNNING'}
    sessions.append(session);write_json(output/'sessions.json',sessions);stop=StopRequest()
    try:
        for key,group in groups.items():
            for i in range(b['warmup'] if key=='normal' else b['fallback_warmup']):
                raw,item=group[i%len(group)];engine.reset();result=engine.decide(lambda:raw.snapshot(item['step']))
                require(path_of(result)==key,'Warmup path changed')
                if key=='normal' and i==0:
                    session['process_to_first_result_ms']=(time.perf_counter_ns()-launched_ns)/1e6
                    session['first_result_ms']=result['e2e_ms']
        torch.cuda.reset_peak_memory_stats()
        for index in range(len(journal.rows),len(order)):
            key,i=order[index];group=groups[key]
            count=b['latency_requests'] if key=='normal' else b['fallback_requests']
            which=i%len(group) if count>=len(group) else round(i*(len(group)-1)/max(1,count-1))
            raw,item=group[which];engine.reset();started=time.perf_counter_ns()
            try:
                result=engine.decide(lambda:raw.snapshot(item['step']));require(path_of(result)==key,'Measured path changed');result['error']=None
            except (RuntimeError,ValueError) as exc:
                if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                result={'error':str(exc),'e2e_ms':(time.perf_counter_ns()-started)/1e6,'gpu_compute_ms':0.,
                        'fallback_reason':'inference_error','tokens':0,'thermal_age_s':1.}
            journal.append({'request':repeat*len(order)+index,'path':key,'path_request':i,'repeat':repeat,'input_id':item['id'],
                            'origin':item['origin'],'session':len(sessions)-1,**result})
            require(not(len(journal.rows)>=3 and all(r.get('error') for r in journal.rows[-3:])),'Three consecutive inference failures')
            if (index+1)%100==0:print(f'L2 latency {method}/{repeat} {index+1}/{len(order)}',flush=True)
            if stop.signal:raise InterruptedError('Latency request committed; resume after warmup')
        session.update(after=system_counters(),status='COMPLETE',peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(output/'sessions.json',sessions)
        result={'method':method,'repeat':repeat,'sessions':sessions,
                'by_path':{k:summarize([r for r in journal.rows if r['path']==k]) for k in PATHS}}
        write_json(output/'summary.json',result);seal(output,identity(run,cfg,method,repeat));return result
    finally:stop.close();engine.close()


def load_trace(run,cfg,method):
    from itertools import chain
    groups=[]
    for repeat in range(cfg['budget']['latency_repeats']):
        folder=run/'18_latency/trials'/f'{method}_{repeat}';verify(folder)
        journal=Journal(folder/'requests.jsonl',identity(run,cfg,method,repeat))
        require(all(any(r['path']==key and not r.get('error') for r in journal.rows) for key in PATHS),'Incomplete path traces')
        groups.append(journal.rows)
    return list(chain.from_iterable(zip(*groups)))
