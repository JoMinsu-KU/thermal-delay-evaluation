"""Operator training, validation-only selection, and compact restart checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from .common import (require, read_json, write_json, save_array, save_training, restore_training,
                     StopRequest, seal, digest)
from .data import EpisodeCache
from .physics import branch, restrict, candidates, costs, order_accuracy
from .operator import ThermalOperator, field_loss, heat_sequence, residual, physics_features, FEATURE_UNITS


class Supervision:
    """Truth access is confined to this training/evaluation reader, with two open episodes."""
    def __init__(self,base,entries):
        self.base=Path(base); self.entries={e['code']:e for e in entries}; self.open=OrderedDict()

    def arrays(self,code):
        code=int(code)
        if code not in self.open:
            if len(self.open)>=2:
                _,arrays=self.open.popitem(last=False)
                for a in arrays: a._mmap.close()
            e=self.entries[code]; eid=e['episode_id']
            self.open[code]=[np.load(self.base/'truth'/eid/(n+'.npy'),mmap_mode='r') for n in ['T32','p_actual']]
            self.open[code].append(np.load(self.base/'raw'/eid/'commands.npy',mmap_mode='r'))
        self.open.move_to_end(code)
        return self.open[code]

    def one(self,row):
        code,step=map(int,row[:2]); t,p,u=self.arrays(code)
        meta=read_json(self.base/'raw'/self.entries[code]['episode_id']/'meta.json')
        return (np.array(t[step,0],copy=True),np.array(meta['passport'],np.float32),np.array(p[step],copy=True),
                np.array(u[step:step+20,2],copy=True),np.array(t[step+1:step+21,0],copy=True))

    def batch(self,rows,device):
        items=[self.one(row) for row in rows]
        return tuple(torch.as_tensor(np.stack(values),device=device,dtype=torch.float32) for values in zip(*items))

    def close(self):
        for arrays in self.open.values():
            for a in arrays: a._mmap.close()
        self.open.clear()


def prepare_operator_data(run,config):
    base=run/'01_generate/data'; entries=read_json(base/'manifests/split.json')['episodes']
    output=run/'04_operator_data'; rng=np.random.default_rng(config['seed']+71)
    indexes={}
    for split in ['train','validation']:
        pool=np.array([[e['code'],step,1] for e in entries if e['split']==split
                       for step in range(int(round(read_json(base/'raw'/e['episode_id']/'meta.json')['duration_s']*10))-19)],np.int64)
        limit=config['budget']['operator_windows'] if split=='train' else min(128,len(pool))
        ids=rng.choice(len(pool),min(limit,len(pool)),replace=False); indexes[split]=pool[ids]
        save_array(output/(split+'_windows.npy'),indexes[split])
    data=Supervision(base,entries); count=min(config['budget']['paired_states'],len(indexes['train']))
    pairs=output/'pairs'; pairs.mkdir(exist_ok=True)
    # One pair at a time; no K*H counterfactual fields are materialized for all anchors.
    shapes={'initial':(count,32,32),'passport':(count,9),'p0':(count,4),
            'commands':(count,2,20,4),'future':(count,2,20,32,32)}
    arrays={k:np.lib.format.open_memmap(pairs/(k+'.npy'),mode='w+',dtype=np.float32,shape=v) for k,v in shapes.items()}
    for index,row in enumerate(indexes['train'][:count]):
        initial,passport,p0,u,_=data.one(row)
        e=data.entries[int(row[0])]; previous=np.load(base/'raw'/e['episode_id']/'commands.npy',mmap_mode='r')
        prev=np.array(previous[max(0,int(row[1])-1),2]) if row[1]>0 else np.zeros(4); previous._mmap.close()
        library,valid=candidates(prev); other=int(rng.choice(np.flatnonzero(valid)[1:]))
        command=library[[0,other]]; future=restrict(branch(initial,p0,command,passport),32)
        for key,value in zip(shapes,[initial,passport,p0,command,future]): arrays[key][index]=value
        if (index+1)%8==0: print(f'paired FV data {index+1}/{count}',flush=True)
    for a in arrays.values(): a.flush(); a._mmap.close()
    data.close(); seal(pairs,{'source_split':'train','initial_state':'true T32 lifted to FV64, both branches re-solved'})
    return {'train_windows':len(indexes['train']),'validation_windows':len(indexes['validation']),'paired_states':count,
            'regular_supervision':'actual applied command sequence and contiguous true rollout, area-restricted FV64',
            'counterfactual_supervision':'same lifted true initial state and actuator, independent FV64 branches',
            'test_accessed':False}


def pair_batch(directory,ids,device):
    values=[]
    for name in ['initial','passport','p0','commands','future']:
        a=np.load(directory/(name+'.npy'),mmap_mode='r'); values.append(np.array(a[ids],copy=True)); a._mmap.close()
    initial,passport,p0,u,future=values
    return tuple(torch.as_tensor(v,device=device,dtype=torch.float32) for v in
                 [np.repeat(initial,2,axis=0),np.repeat(passport,2,axis=0),np.repeat(p0,2,axis=0),
                  u.reshape(-1,20,4),future.reshape(-1,20,32,32)])


def evaluate_operator(model,data,index,device):
    model.eval(); square=0.; count=0; first=0.; peak=0.; delta=0.
    with torch.no_grad():
        for start in range(0,len(index),16):
            initial,passport,p0,u,target=data.batch(index[start:start+16],device)
            predicted=model(initial,passport,p0,u); require(torch.isfinite(predicted).all().item(),'Nonfinite operator prediction')
            error=predicted-target
            square+=float(error[:,-1].square().sum()); count+=error[:,-1].numel()
            first+=float(error[:,0].square().sum()); delta+=float(error.square().mean())*len(initial)
            peak=max(peak,float((predicted.amax((-2,-1))-target.amax((-2,-1))).abs().max()))
    return {'horizon_rmse_k':(square/count)**.5,'one_step_rmse_k':(first/count)**.5,
            'all_horizons_rmse_k':(delta/len(index))**.5,'peak_abs_error_k':peak,'windows':len(index)}


def candidate_validation(model,labels,device,config):
    model.eval(); errors=[]; correct=total=0; regrets=[]; peak_errors=[]
    fields=np.load(labels/'validation_field.npy',mmap_mode='r'); ps=np.load(labels/'validation_p0.npy',mmap_mode='r')
    passports=np.load(labels/'validation_passport.npy',mmap_mode='r'); commands=np.load(labels/'validation_commands.npy',mmap_mode='r')
    expert=np.load(labels/'validation_cost.npy',mmap_mode='r'); valid=np.load(labels/'validation_eligible.npy',mmap_mode='r')
    library=np.load(labels/'validation_valid.npy',mmap_mode='r')
    goals=np.load(labels/'validation_goal.npy',mmap_mode='r'); previous=np.load(labels/'validation_previous.npy',mmap_mode='r')
    for i in range(len(fields)):
        initial=torch.as_tensor(np.repeat(fields[i:i+1],9,axis=0),device=device)
        theta=torch.as_tensor(np.repeat(passports[i:i+1],9,axis=0),device=device)
        p=torch.as_tensor(np.repeat(ps[i:i+1],9,axis=0),device=device)
        u=torch.tensor(np.array(commands[i]),device=device)
        with torch.no_grad(): predicted=model(initial,theta,p,u).cpu().numpy()
        reference=branch(fields[i],ps[i],commands[i],passports[i]); coarse=restrict(reference,32)
        active=np.array(library[i]); active[0]=True
        errors.append(float(np.mean(((predicted-predicted[:1])-(coarse-coarse[:1]))[active]**2)))
        peak_errors.append(float(np.max(abs(predicted.max(axis=(2,3))-reference.max(axis=(2,3))))))
        estimated=costs(predicted,commands[i],previous[i],goals[i])
        ok,n=order_accuracy(estimated,expert[i],valid[i],config['acceptance']['solver']['significant_cost_gap'])
        correct+=ok; total+=n
        if valid[i].any():
            choice=np.where(valid[i],estimated,np.inf).argmin(); regrets.append(float(expert[i,choice]-expert[i,valid[i]].min()))
    for a in [fields,ps,passports,commands,expert,valid,library,goals,previous]: a._mmap.close()
    return {'candidate_delta_rmse_k':float(np.sqrt(np.mean(errors))),
            'significant_order_accuracy':correct/total if total else None,'significant_pairs':total,
            'candidate_regret':float(np.mean(regrets)) if regrets else None,'candidate_peak_abs_error_k':max(peak_errors)}


def train_operator(run,config,variant,output,device='cuda'):
    from .common import setup_torch
    setup_torch(config,device)
    base=run/'01_generate/data'; entries=read_json(base/'manifests/split.json')['episodes']
    folder=run/'04_operator_data'; data=Supervision(base,entries)
    train=np.load(folder/'train_windows.npy',mmap_mode='r'); val=np.load(folder/'validation_windows.npy',mmap_mode='r')
    model=ThermalOperator(variant).to(device); rng=np.random.default_rng(config['seed']+83)
    settings=config['poc']['training']; cfg=settings['stages']['operator']; budget=config['budget']
    initial=evaluate_operator(model,data,val,device); metrics=[]; best=initial['horizon_rmse_k']; stop=StopRequest()
    if not (output/'initial.pt').exists(): torch.save(model.state_dict(),output/'initial.pt')
    if not (output/'best.pt').exists(): torch.save(model.state_dict(),output/'best.pt')
    selected_step=['initial',0]
    pair_index=np.load(folder/'pairs/initial.npy',mmap_mode='r'); pairs_count=len(pair_index); pair_index._mmap.close()
    try:
        for phase,total in [('supervised',budget['operator_steps']),('paired',budget['pair_steps'])]:
            phase_cfg=cfg if phase=='supervised' else settings['stages']['operator_pairs']
            micro=phase_cfg.get('microbatch',phase_cfg.get('microbatch_pairs')); accumulation=phase_cfg['accumulation']
            optimizer=torch.optim.AdamW(model.parameters(),lr=phase_cfg['learning_rate'],weight_decay=.01,foreach=False)
            restart=output/(phase+'_restart.pt'); step=0
            if restart.exists():
                step,extra=restore_training(restart,model,optimizer,rng,device)
                metrics=extra['metrics']; best=extra['best']; selected_step=extra['selected_step']
            for step in range(step+1,total+1):
                model.train(); optimizer.zero_grad(set_to_none=True); summed=np.zeros(3); started=time.monotonic()
                for _ in range(accumulation):
                    if phase=='supervised':
                        batch=data.batch(train[rng.integers(len(train),size=micro)],device)
                    else: batch=pair_batch(folder/'pairs',rng.integers(pairs_count,size=micro),device)
                    field,passport,p0,u,target=batch; prediction=model(field,passport,p0,u)
                    field_term=field_loss(prediction,target)
                    physics_term=residual(prediction,field,heat_sequence(p0,u,passport),passport).div(40000).square().mean()
                    delta_term=prediction.new_zeros(())
                    if phase=='paired':
                        difference=(prediction[1::2]-prediction[::2])-(target[1::2]-target[::2])
                        delta_term=torch.nn.functional.smooth_l1_loss(difference/.25,torch.zeros_like(difference))
                    ramp=min(1.,(step+(budget['operator_steps'] if phase=='paired' else 0))/cfg['pde_ramp_steps'])
                    loss=field_term if variant in ['fno_data','fno_balance'] else field_term+.1*ramp*physics_term+.2*delta_term
                    require(torch.isfinite(loss).item(),'Nonfinite operator loss')
                    (loss/accumulation).backward()
                    summed+=np.array([float(field_term.detach()),float(physics_term.detach()),float(delta_term.detach())])/accumulation
                gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                warmup=max(1,int(np.ceil(total*.05)))
                for group in optimizer.param_groups: group['lr']=phase_cfg['learning_rate']*min(1,step/warmup)
                optimizer.step()
                row={'phase':phase,'step':step,'field_loss':summed[0],'pde_loss':summed[1],'delta_loss':summed[2],
                     'paired_batch_fraction':float(phase=='paired'),'gradient_norm':gradient_norm,'seconds':time.monotonic()-started}
                metrics.append(row)
                check=step==total or step%settings['validation_every_steps']==0 or config['profile']=='smoke'
                if check:
                    evaluation=evaluate_operator(model,data,val,device); row['validation']=evaluation
                    if evaluation['horizon_rmse_k']<best:
                        best=evaluation['horizon_rmse_k']; selected_step=[phase,step]; torch.save(model.state_dict(),output/'best.pt')
                if step%settings['save_every_steps']==0 or step==total or stop.signal:
                    save_training(restart,model,optimizer,step,rng,{'metrics':metrics,'best':best,'selected_step':selected_step})
                print(f'{variant} {phase} {step}/{total} field={summed[0]:.6g} PDE={summed[1]:.6g}',flush=True)
                if stop.signal: raise InterruptedError('Saved operator checkpoint at optimizer boundary; rerun --resume')
        torch.save(model.state_dict(),output/'last.pt')
        last=evaluate_operator(model,data,val,device)
        initial_weights=torch.load(output/'initial.pt',weights_only=True,map_location=device)
        update_max=max(float((v-initial_weights[k]).abs().max()) for k,v in model.state_dict().items())
        model.load_state_dict(torch.load(output/'best.pt',weights_only=True,map_location=device))
        final=evaluate_operator(model,data,val,device)
        ranking=candidate_validation(model,run/'03_observer_labels',device,config)
        thresholds=config['acceptance']['operator']
        gates={'field':final['horizon_rmse_k']<=thresholds['horizon_rmse_k'],
               'delta':ranking['candidate_delta_rmse_k']<=thresholds['candidate_delta_rmse_k'],
               'order':ranking['significant_order_accuracy'] is not None and ranking['significant_order_accuracy']>=thresholds['significant_order_accuracy'],
               'peak':max(final['peak_abs_error_k'],ranking['candidate_peak_abs_error_k'])<=thresholds['peak_abs_error_k']}
        write_json(output/'training.json',{'rows':metrics,'initial_validation':initial,'last_validation':last,
                   'selected_validation':final,'selected_step':selected_step,'parameter_update_max':update_max,'ranking':ranking,'gates':gates})
        return {'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'validation':final,
                'candidate_validation':ranking,'gates':gates,'quality_pass':all(gates.values()),'checkpoint':'best.pt',
                'selected_step':selected_step,'parameter_update_max':update_max,
                'checkpoint_sha256':digest(output/'best.pt'),'optimizer_steps':budget['operator_steps']+budget['pair_steps'],
                'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0.,'test_accessed':False}
    finally:
        stop.close(); data.close(); train._mmap.close(); val._mmap.close()


def cache_physics(run,config,output,device='cuda'):
    labels=run/'03_observer_labels'; operator_folder=run/'07_pino_balance'
    result=read_json(operator_folder/'result.json')
    require(config['profile']=='smoke' or result['quality_pass'], 'G3 failed: inspect operator metrics before policy fitting')
    model=ThermalOperator('pino_balance').to(device)
    model.load_state_dict(torch.load(operator_folder/'best.pt',map_location=device,weights_only=True)); model.eval()
    for split in ['train','validation']:
        names=['field','p0','passport','commands','previous','goal','age']
        arrays={name:np.load(labels/(split+'_'+name+'.npy'),mmap_mode='r') for name in names}
        count=len(arrays['field']); features=np.lib.format.open_memmap(output/(split+'_features.npy'),mode='w+',dtype=np.float32,shape=(count,9,16))
        for i in range(count):
            field=torch.tensor(np.repeat(arrays['field'][i:i+1],9,axis=0),device=device)
            passport=torch.tensor(np.repeat(arrays['passport'][i:i+1],9,axis=0),device=device)
            p0=torch.tensor(np.repeat(arrays['p0'][i:i+1],9,axis=0),device=device)
            u=torch.tensor(np.array(arrays['commands'][i]),device=device)
            previous=torch.tensor(np.repeat(arrays['previous'][i:i+1],9,axis=0),device=device)
            goal=torch.full((9,),float(arrays['goal'][i]),device=device); age=torch.full_like(goal,float(arrays['age'][i]))
            with torch.no_grad():
                fields=model(field,passport,p0,u)
                feature=physics_features(fields,field,p0,u,passport,previous,goal,age)
            require(torch.isfinite(feature).all().item(),'Nonfinite physics feature')
            features[i]=feature.cpu().numpy()
        features.flush(); features._mmap.close()
        for a in arrays.values(): a._mmap.close()
    metadata={'source':'predicted PINO-balance futures; no FV teacher cost input','units':FEATURE_UNITS,
              'operator_sha256':digest(operator_folder/'best.pt'),'labels_manifest_sha256':digest(labels/'manifest.json'),
              'dataset_split_sha256':digest(run/'01_generate/data/manifests/split.json'),
              'observer':'bilinear-probe-v1','library':'axis9_poc_v1','cost':'thermal_poc_v1',
              'dtype':'float32','timestamp_policy':'capture_receive_ready_le_decision_v1','margins_13_14':'fixed_zero',
              'operator_quality_pass':result['quality_pass'],'smoke_only_if_quality_failed':not result['quality_pass']}
    write_json(output/'cache_key.json',metadata)
    return metadata
