"""Train-only on-policy FV relabeling and validation-only closed-loop selection.

Synthetic delays are development interventions, never a substitute for measured G6.
"""
from pathlib import Path
import gc
import time
import numpy as np
import torch
from ..common import read_json,write_json,require,save_array,seal,verify,digest,save_training,restore_training
from ..physics import candidates,branch,costs,soft_targets
from ..policy import policy_loss
from .contracts import entries,modality
from .cache import Rows,SHAPES,prediction
from .data import Plant,view,packed
from .engine import Engine
from .control import simulate
from .models import export,restore
from .graph import GraphPolicy

METHODS=['shared_no','shared_physics','small_physics','shared_sensor','shared_thermal']
DELAYS={'zero':[0.], '13ms':[13.], '38ms':[38.], '50ms':[50.], 'jitter':[13.,50.,38.,20.,60.]}


def delay_trace(name):
    require(name in DELAYS,'Unknown development delay')
    return [{'fallback_reason':reason,'tokens':0,'request':i,'e2e_ms':v,'error':None}
            for reason in [None,'cold_start_or_missing_observation','no_predicted_feasible_candidate']
            for i,v in enumerate(DELAYS[name])]


def settings(cfg):
    return cfg['l2']['rollout_learning']['profiles'][cfg['profile']]


def cohort(run,split,limit=None):
    require(split in ['train','validation'],'Rollout development forbids calibration/test/OOD')
    result=entries(run,[split]);require(result and all(r['split']==split for r in result),'Invalid development split')
    return result if limit is None else result[:limit]


def prepare_model(model):
    model.eval()
    if hasattr(model,'backbone'):model.backbone.gradient_checkpointing_disable()
    return GraphPolicy(model).eval()


def evaluate_loops(root,run,cfg,folder,method,model=None,checkpoint=None,stop=None):
    folder.mkdir(parents=True,exist_ok=True)
    identity={'method':method,'checkpoint':digest(checkpoint/'parameters.pt') if checkpoint else None,
              'split':'validation','delays':settings(cfg)['delays'],'margin':0.,'duration_s':cfg['budget']['duration_s'],
              'source':'synthetic common delays; not measured latency or final test'}
    if (folder/'identity.json').exists():require(read_json(folder/'identity.json')==identity,'Validation candidate changed on resume')
    else:write_json(folder/'identity.json',identity)
    if (folder/'COMPLETE').exists():verify(folder);return read_json(folder/'metrics.json')
    engine=Engine(root,run,method,calibrated=False,policy_override=prepare_model(model) if model is not None else None)
    rows=[]
    try:
        for name in settings(cfg)['delays']:
            for entry in cohort(run,'validation'):
                path=folder/(name+'_'+entry['episode_id']+'.json')
                if path.exists():row=read_json(path)
                else:
                    plant=Plant(run/'02_generate/data',entry,cfg['budget']['duration_s'])
                    metric,_=simulate(plant,engine,'delayed',delay_trace(name),stop)
                    row={**metric,'episode':entry['episode_id'],'stratum':entry['stratum'],'delay':name}
                    write_json(path,row)
                rows.append(row)
                print(f'L2 rollout validation {method} {folder.name} {name} {entry["episode_id"]} cost={row["episode_cost"]:.7g}',flush=True)
        write_json(folder/'metrics.json',rows);seal(folder,identity);return rows
    finally:engine.close();gc.collect();torch.cuda.empty_cache()


def selection_key(rows,references):
    """Safety first, then worst cost ratio over all delay/normal/all cohorts."""
    actual={(r['delay'],r['episode']):r for r in rows}
    require(len(actual)==len(rows) and rows,'Duplicate or empty validation decisions')
    violations=0;ratios=[]
    for ref in references.values():
        expected={(r['delay'],r['episode']):r for r in ref}
        require(set(actual)==set(expected),'Unpaired closed-loop validation')
        for key,a in actual.items():
            b=expected[key]
            require(np.isfinite(a['episode_cost']) and a['completed'],'Incomplete or nonfinite validation episode')
            violations+=int(a['new_violation']>b['new_violation'] or a['violation_s']>b['violation_s']+1e-8 or a['inference_failures']>b['inference_failures'])
        for delay in sorted({r['delay'] for r in rows}):
            for stratum in ['all','normal']:
                keys=[k for k,r in actual.items() if r['delay']==delay and (stratum=='all' or r['stratum']==stratum)]
                if keys:
                    cost=np.mean([actual[k]['episode_cost'] for k in keys]);base=np.mean([expected[k]['episode_cost'] for k in keys])
                    ratios.append(float(cost/max(base,1e-8)))
    return [violations,max(ratios),float(np.mean(ratios))]


def label_observation(obs,engine,normalization,temperature,code,step):
    """Expert sees the same current observed field, never the plant's truth."""
    command,valid=candidates(obs.previous)
    row={k:np.zeros(shape,dtype=(bool if k in ['valid','eligible','usable'] else np.int64 if k=='index' else np.float32)) for k,shape in SHAPES.items()}
    row.update(packed(obs,normalization,engine.mode))
    row.update(commands=command,valid=valid,usable=obs.usable,index=np.array([code,step]),goal=obs.goal,
               previous=obs.previous,actuator=obs.actuator,theta=obs.passport,thermal_age=obs.thermal_age[-1])
    if obs.usable:
        field=engine.observer(obs);future=branch(field,obs.actuator,command,obs.passport)
        cost=costs(future,command,obs.previous,obs.goal);eligible=valid&(future.max((1,2,3))<=120)
        pred,features=prediction(engine.operator,field,obs,command)
        row.update(field=field,features=features.cpu().numpy(),cost=cost,eligible=eligible,
                   target=soft_targets(cost,eligible,temperature) if eligible.any() else np.zeros(9),
                   fine_peak=future.max(),predicted_peak=float(pred.max()))
    return row


class Collector:
    def __init__(self,engine,entry,steps,normalization,temperature):
        self.engine=engine;self.entry=entry;self.steps=set(steps);self.normalization=normalization;self.temperature=temperature;self.rows=[];self.tick=0
    def reset(self):self.engine.reset();self.tick=0
    def decide(self,make_observation):
        raw=make_observation();decision=self.engine.decide(lambda:raw)
        if self.tick in self.steps:
            self.rows.append(label_observation(view(raw,self.engine.mode),self.engine,self.normalization,self.temperature,self.entry['code'],self.tick))
        self.tick+=1
        return decision


def collect(root,run,cfg,folder,method,model,checkpoint,round_index,stop):
    folder.mkdir(parents=True,exist_ok=True)
    identity={'checkpoint':digest(checkpoint/'parameters.pt'),'round':round_index,'split':'train','method':method}
    if (folder/'identity.json').exists():require(read_json(folder/'identity.json')==identity,'Rollout collector checkpoint changed')
    else:write_json(folder/'identity.json',identity)
    if (folder/'COMPLETE').exists():verify(folder);return folder
    spec=settings(cfg);selected=cohort(run,'train',spec['train_episodes'])
    require(len(selected)==spec['train_episodes'],'Rollout train budget exceeds cohort')
    steps=np.linspace(1,round(cfg['budget']['duration_s']*10)-21,spec['anchors'],dtype=int)
    require(len(set(steps))==len(steps) and min(steps)>=0,'Repeated rollout anchors')
    norm=read_json(run/'03_index/normalization.json')
    temperature=read_json(run/'05_cache'/modality(method)/'train/metadata.json')['label_temperature']
    engine=Engine(root,run,method,calibrated=False,policy_override=prepare_model(model));parts=[];records=[]
    try:
        for i,entry in enumerate(selected):
            name=spec['delays'][(i+round_index-1)%len(spec['delays'])];path=folder/(entry['episode_id']+'.npz');record=path.with_suffix('.json')
            if not path.exists() or not record.exists():
                tap=Collector(engine,entry,steps,norm,temperature)
                metric,_=simulate(Plant(run/'02_generate/data',entry,cfg['budget']['duration_s']),tap,'delayed',delay_trace(name),stop)
                require(len(tap.rows)==len(steps),'Missing rollout training rows')
                # Commit the whole episode; interrupted episodes restart, finished ones persist.
                with path.with_suffix('.tmp').open('wb') as f:
                    np.savez(f,**{k:np.stack([np.asarray(r[k],dtype=(bool if k in ['valid','eligible','usable'] else np.int64 if k=='index' else np.float32)) for r in tap.rows]) for k in SHAPES})
                path.with_suffix('.tmp').replace(path)
                write_json(record,{'episode':entry['episode_id'],'split':'train','delay':name,'rows':len(tap.rows),'sha256':digest(path),'control':metric})
            meta=read_json(record);require(digest(path)==meta['sha256'],'Rollout episode changed')
            parts.append(path);records.append(meta)
            print(f'L2 rollout collect {method} round={round_index} {i+1}/{len(selected)}',flush=True)
        for key in SHAPES:
            chunks=[]
            for p in parts:
                with np.load(p) as z:chunks.append(z[key])
            save_array(folder/(key+'.npy'),np.concatenate(chunks))
        info={**identity,'episodes':records,'rows':len(selected)*len(steps),'label_temperature':temperature,
              'labels':'FV64 from current observed field; no future plant truth or PINO self-labels',
              'normalization':'original train-only normalization and feature RMS remain fixed'}
        write_json(folder/'metadata.json',info);seal(folder,info);return folder
    finally:engine.close();gc.collect();torch.cuda.empty_cache()


def mixed_batch(original,pools,rng,size,device='cuda'):
    """Half original, half accumulated on-policy data; no validation sampling."""
    sources=[original,*pools]
    require(pools and all(r.meta['split']=='train' for r in sources),'Only training sources may enter rehearsal batches')
    eligible=[np.flatnonzero(np.asarray(r.arrays['target']).sum(1)>0) for r in sources]
    require(all(len(v)>0 for v in eligible),'No feasible labels in a training source')
    # Sampling pooled examples uniformly avoids overweighting small rounds.
    lengths=np.array([len(v) for v in eligible[1:]],dtype=float);weights=lengths/lengths.sum()
    chosen=[]
    for i in range(size):
        source=0 if (i%2==0 and size>1) or (size==1 and rng.random()<.5) else 1+int(rng.choice(len(pools),p=weights))
        chosen.append((source,int(rng.choice(eligible[source]))))
    return {k:torch.as_tensor(np.stack([sources[s].arrays[k][i] for s,i in chosen]),device=device) for k in SHAPES}


def fit_round(model,original,pools,cfg,folder,stop):
    folder.mkdir(parents=True,exist_ok=True);spec=settings(cfg)
    round_seed=cfg['seed']+int(folder.parent.name)*1009
    # Cache hits must not alter dropout RNG at a new training round.
    torch.manual_seed(round_seed);torch.cuda.manual_seed_all(round_seed)
    for name,p in model.named_parameters():p.requires_grad_(not name.startswith('backbone.') or 'lora_' in name)
    if hasattr(model,'backbone'):model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    new=[p for n,p in named.items() if 'lora_' not in n];lora=[p for n,p in named.items() if 'lora_' in n]
    base=[(p,p._version) for n,p in model.named_parameters() if n.startswith('backbone.') and 'lora_' not in n]
    groups=[{'params':new,'lr':1e-4}]+([{'params':lora,'lr':3e-5}] if lora else [])
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False);rng=np.random.default_rng(round_seed)
    restart=folder/'restart.pt';start=0;history=[]
    if restart.exists():start,extra=restore_training(restart,model,optimizer,rng,'cuda');history=extra['history']
    size=min(cfg['budget']['accumulation'],cfg['l2']['shared_microbatch'])
    accumulation=cfg['budget']['accumulation']//size
    for step in range(start+1,spec['steps']+1):
        model.train();optimizer.zero_grad(set_to_none=True);total=0.;started=time.perf_counter()
        for _ in range(accumulation):
            batch=mixed_batch(original,pools,rng,size);order=rng.permutation(9)
            scores=model(batch,batch['commands'][:,order],batch['features'][:,order])[0]
            loss=torch.stack([policy_loss(scores[j],*[batch[k][j,order] for k in ['target','cost','valid','eligible']],1e-6) for j in range(size)]).mean()
            require(torch.isfinite(loss).item(),'Nonfinite rollout loss');(loss/accumulation).backward();total+=float(loss.detach())/accumulation
        gn=sum(float(p.grad.abs().sum()) for p in new if p.grad is not None);gl=sum(float(p.grad.abs().sum()) for p in lora if p.grad is not None)
        require(gn>0 and (not lora or gl>0),'Missing rollout gradients')
        torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True);optimizer.step()
        require(all(p.grad is None and p._version==v for p,v in base),'Frozen base changed during rollout fitting')
        history.append({'phase':'rollout','step':step,'loss':total,'new_gradient':gn,'lora_gradient':gl,'seconds':time.perf_counter()-started,'global_decisions':size*accumulation})
        if step%25==0 or step==spec['steps'] or stop.signal:
            save_training(restart,model,optimizer,step,rng,{'history':history})
        print(f'L2 rollout fit {folder.parent.name} {step}/{spec["steps"]} loss={total:.6g}',flush=True)
        if stop.signal:raise InterruptedError('Rollout optimizer boundary committed')
    write_json(folder/'history.json',history);return history


def refine(root,run,cfg,output,method,model,original,val,baseline,stop):
    from .training import evaluate
    target=run/'models'/method;folder=output/'rollout';folder.mkdir(exist_ok=True)
    spec=settings(cfg);offline=target/'offline_best'
    if not offline.exists():export(model,offline,{'method':method,'selection':'offline validation regret','training_seed':cfg['seed']})
    restore(model,offline)
    refs={m:evaluate_loops(root,run,cfg,run/'rollout_validation_reference'/m,m,stop=stop) for m in ['pi','pino_mpc']}
    candidates_list=[];pools=[];history=[]
    try:
        for index in range(spec['rounds']+1):
            checkpoint=offline if index==0 else target/f'rollout_{index:02d}'
            if index:
                previous=offline if index==1 else target/f'rollout_{index-1:02d}'
                restore(model,previous)
                stage=folder/f'{index:02d}';stage.mkdir(exist_ok=True)
                cache=collect(root,run,cfg,stage/'01_collect',method,model,previous,index,stop);pools.append(Rows(cache))
                if not checkpoint.exists():
                    fit_round(model,original,pools,cfg,stage/'02_fit',stop)
                    export(model,checkpoint,{'method':method,'selection':['rollout',index],'training_seed':cfg['seed']})
                else:restore(model,checkpoint)
                history.extend(read_json(stage/'02_fit/history.json'))
            offline_score=evaluate(model,val)
            rows=evaluate_loops(root,run,cfg,folder/'03_selection'/f'candidate_{index:02d}',method,model,checkpoint,stop)
            eligible=offline_score['selection_score']<=baseline['selection_score'] and offline_score['infeasible']==0 and offline_score['unexpected_fallback']==0
            candidates_list.append({'round':index,'checkpoint':str(checkpoint.relative_to(run)),
                                    'checkpoint_sha256':digest(checkpoint/'parameters.pt'),'offline_validation':offline_score,
                                    'eligible':eligible,'closed_loop_key':selection_key(rows,refs)})
        allowed=[c for c in candidates_list if c['eligible']];require(allowed,'No candidate satisfies unchanged offline readiness criteria')
        chosen=min(allowed,key=lambda c:(*c['closed_loop_key'],c['round']))
        restore(model,run/chosen['checkpoint'])
        export(model,target/'best',{'method':method,'selection':['closed_loop',chosen['round']],'training_seed':cfg['seed']})
        result={'selected_round':chosen['round'],'candidates':candidates_list,'history':history,
                'selection':'validation only: safety first, worst all/normal cost ratio across synthetic delays, mean ratio, then earlier candidate',
                'test_accessed':False,'train_episode_ids':[r['episode_id'] for r in cohort(run,'train',spec['train_episodes'])],
                'validation_episode_ids':[r['episode_id'] for r in cohort(run,'validation')],
                'protocol':'on-policy train-only FV relabeling, 50% original rehearsal, original normalizers fixed; no measured latency claim',
                'remaining_gates':'unchanged pre-calibration audit and final G5/G6 still required'}
        write_json(folder/'selection.json',result);return result
    finally:
        for rows in pools:rows.close()
