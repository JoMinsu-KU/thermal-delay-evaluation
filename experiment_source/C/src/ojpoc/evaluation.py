"""Preregistered pilot evaluation, paired episode statistics and resumable records."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import (require, read_json, write_json, object_hash, digest, seal, verify,
                     StopRequest, setup_torch)
from .data import generate_episode, RawEpisode, bilinear_observer
from .physics import candidates, branch, costs, restrict, order_accuracy
from .operator import ThermalOperator, BalanceOnly, heat_sequence, residual


class Journal:
    """Hash each committed row; discard only an incomplete final write after a crash."""
    def __init__(self, path, identity):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.identity=object_hash(identity); self.rows=[]
        if self.path.exists():
            with self.path.open('rb+') as stream:
                offset=0
                while line:=stream.readline():
                    if not line.endswith(b'\n'):
                        stream.truncate(offset); break
                    item=json.loads(line)
                    require(item['identity']==self.identity and item['index']==len(self.rows), 'Journal identity/order changed')
                    require(item['sha256']==object_hash(item['row']), 'Journal row corrupted')
                    self.rows.append(item['row']); offset=stream.tell()

    def append(self,row):
        item={'identity':self.identity,'index':len(self.rows),'row':row,'sha256':object_hash(row)}
        with self.path.open('ab') as stream:
            stream.write((json.dumps(item,allow_nan=False)+'\n').encode()); stream.flush(); os.fsync(stream.fileno())
        self.rows.append(row)


def mean(values):
    valid=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def quantiles(values):
    values=np.asarray(values,dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite metric samples')
    if not len(values): return {'count':0,'mean':None,'p50':None,'p95':None,'p99':None,'max':None}
    return {'count':len(values),'mean':float(values.mean()),'p50':float(np.quantile(values,.5)),
            'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99)),'max':float(values.max())}


def paired(rows, left, right, metric, config):
    """left - right, equal weight per episode; only common defined rows, coverage explicit."""
    groups={}; missing=0
    for row in rows:
        a=row['methods'][left].get(metric); b=row['methods'][right].get(metric)
        if a is None or b is None: missing+=1; continue
        require(np.isfinite(a) and np.isfinite(b),'Nonfinite paired values')
        groups.setdefault(str(row['episode']),[]).append(float(a)-float(b))
    values=np.array([np.mean(v) for v in groups.values()])
    out={'direction':left+' minus '+right,'metric':metric,'planned_rows':len(rows),'common_rows':len(rows)-missing,
         'excluded_undefined_rows':missing,'episodes':len(values),'per_episode':{k:float(np.mean(v)) for k,v in groups.items()},
         'mean_difference':None,'ci95':None,'resampling_unit':'episode','interpretation':'INCONCLUSIVE'}
    if not len(values): return out
    out['mean_difference']=float(values.mean())
    if len(values)>=2:
        rng=np.random.default_rng(config['seed']+991)
        samples=values[rng.integers(len(values),size=(config['validation']['bootstrap_resamples'],len(values)))].mean(1)
        lo,hi=map(float,np.quantile(samples,[.025,.975])); out['ci95']=[lo,hi]
        out['interpretation']='POSITIVE' if hi<0 else ('NEGATIVE' if lo>0 else 'INCONCLUSIVE')
    return out


def evaluation_plan(config):
    cfg=config['validation']; entries=[]
    for stratum,count in [('normal',cfg['normal_episodes']),('stress',cfg['stress_episodes'])]:
        for i in range(count):
            code=cfg.get('episode_code_start',5000)+len(entries)
            # Existing split uses codes below 416; reserve a disjoint block per run seed.
            entries.append({'code':code,'episode_id':f'pv_{code:06d}','group_id':f'pv_plant_recipe_{code:06d}',
                            'split':'pilot_eval','stratum':stratum,'sim_seed':config['seed']*10000+code,
                            'noise_seed':config['seed']*10000+code+10**8,
                            'render_seed':config['seed']*10000+code+2*10**8,'ood_group':None,
                            'strategy_override':'stress' if stratum=='stress' else ('pi' if i%2==0 else 'perturbation'),
                            'closed_loop':i<cfg['closed_loop_'+stratum]})
    steps=np.linspace(1,int(round(cfg['duration_s']*10))-20,cfg['anchors_per_episode'],dtype=int)
    require(len(np.unique(steps))==len(steps),'Duplicate evaluation anchors')
    return entries,steps


def prepare_evaluation(root,run,config,output,entries_override=None,replay_data=None):
    entries,steps=evaluation_plan(config)
    if entries_override is not None: entries=entries_override
    base=output/'data'; base.mkdir(exist_ok=True)
    original=read_json(run/'01_generate/data/manifests/split.json')['episodes']
    for key in ['code','group_id','sim_seed','noise_seed','render_seed']:
        require(not ({e[key] for e in entries}&{e[key] for e in original}),f'Evaluation overlap: {key}')
    freeze={str(p.relative_to(run)):digest(p) for folder in ['07_pino_balance','14_fno_balance']
            for p in (run/folder).glob('*.pt') if 'restart' not in p.name}
    for folder in ['09_native_no_physics','11_native_physics_text']:
        for checkpoint in ['best_model','adapt_model']:
            directory=run/folder/checkpoint; verify(directory)
            freeze[str((directory/'manifest.json').relative_to(run))]=digest(directory/'manifest.json')
    plan={'protocol':config['validation'],'entries':entries,'anchor_steps':steps.tolist(),
          'checkpoints_frozen_before_eval':freeze,'model_selection_split':'validation','evaluation_used_for_fitting':False}
    if (output/'plan.json').exists(): require(read_json(output/'plan.json')==plan,'Evaluation plan changed')
    else: write_json(output/'plan.json',plan)
    stop=StopRequest(); counts={'rows':0,'unusable':0,'expert_fallback':0}
    try:
        for entry in entries:
            directory=output/'labels'/entry['episode_id']
            if (directory/'COMPLETE').exists():
                verify(directory); result=read_json(directory/'counts.json')
                for k in counts: counts[k]+=result[k]
                continue
            if entry.get('cohort')=='replay':
                import shutil
                require(replay_data is not None,'Missing replay source')
                for role in ['raw','truth']:
                    source=Path(replay_data)/role/entry['episode_id']; target=base/role/entry['episode_id']
                    verify(source)
                    # Copy immutable inputs; never link writable outputs into the parent run.
                    if not (target/'COMPLETE').exists(): shutil.copytree(source,target,dirs_exist_ok=True)
                    require(verify(target)==verify(source),'Replay data differs from parent')
            else:
                generate_episode(base,entry,config,config['validation']['duration_s'])
            for role in ['raw','truth']: verify(base/role/entry['episode_id'])
            episode=RawEpisode(base/'raw'/entry['episode_id']); rows=[]; fields=[]; commands=[]; valid=[]; eligible=[]; expert=[]; refinement=None
            directory.mkdir(parents=True,exist_ok=True)
            try:
                for step in steps:
                    obs=episode.snapshot(int(step)); u,v=candidates(obs.previous)
                    row={'step':int(step),'usable':obs.usable,'reason':obs.reason}
                    t=bilinear_observer(obs) if obs.usable else np.zeros((32,32),np.float32)
                    if obs.usable:
                        future=branch(t,obs.actuator,u,obs.passport)
                        j=costs(future,u,obs.previous,obs.goal); ok=v&(future.max((1,2,3))<=120)
                        if refinement is None:
                            fine=branch(t,obs.actuator,u,obs.passport,grid=128,max_dt=.005)
                            coarse=restrict(fine,64)
                            delta_error=(future-future[:1])-(coarse-coarse[:1])
                            refinement={'step':int(step),'candidate_delta_mse':float((delta_error[v]**2).mean()),
                                        'field_rmse_k':float(np.sqrt(((future-coarse)**2).mean())),
                                        'peak_error_k':float(abs(future.max((2,3))-fine.max((2,3))).max())}
                    else: j=np.zeros(9); ok=np.zeros(9,bool)
                    rows.append(row); fields.append(t); commands.append(u); valid.append(v); eligible.append(ok); expert.append(j)
                    if stop.signal: raise InterruptedError('Evaluation generation interrupted; completed episodes retained')
                temporary=directory/'labels.tmp'
                with temporary.open('wb') as stream:
                    np.savez(stream,field=np.array(fields),commands=np.array(commands),valid=np.array(valid),
                             eligible=np.array(eligible),cost=np.array(expert)); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary,directory/'labels.npz'); write_json(directory/'rows.json',rows)
                write_json(directory/'refinement.json',refinement)
                result={'rows':len(rows),'unusable':sum(not r['usable'] for r in rows),
                        'expert_fallback':sum(r['usable'] and not ok.any() for r,ok in zip(rows,eligible))}
                write_json(directory/'counts.json',result); seal(directory,{'split':'pilot_eval'})
                for k in counts: counts[k]+=result[k]
                print(f'pilot_eval {entry["episode_id"]}: {result}',flush=True)
            finally: episode.close()
    finally: stop.close()
    return {**counts,'episodes':len(entries),'closed_loop_episodes':sum(e['closed_loop'] for e in entries),
            'frozen_checkpoints':freeze,'test_accessed':False,'evaluation_role':'held-out development evaluation, not final test'}


class EvaluationData:
    def __init__(self,run):
        self.folder=Path(run)/'15_eval_data'; self.plan=read_json(self.folder/'plan.json')
        for path,expected in self.plan['checkpoints_frozen_before_eval'].items():
            require(digest(Path(run)/path)==expected,'Frozen evaluation checkpoint changed')
        self.entries=self.plan['entries']; self.cached=None; self.raw=None; self.arrays={}

    def episode(self,entry):
        if self.cached!=entry['code']:
            self.close(); self.cached=entry['code']
            self.raw=RawEpisode(self.folder/'data/raw'/entry['episode_id'])
            with np.load(self.folder/'labels'/entry['episode_id']/'labels.npz',allow_pickle=False) as data:
                self.arrays={k:data[k] for k in data.files}
        return self.raw,self.arrays

    def rows(self):
        for entry in self.entries:
            raw,arrays=self.episode(entry)
            for i,step in enumerate(self.plan['anchor_steps']):
                yield entry,int(step),raw.snapshot(step),{k:v[i] for k,v in arrays.items()}

    def close(self):
        if self.raw is not None: self.raw.close()
        self.cached=None; self.raw=None; self.arrays={}


def tensor_inputs(field,obs,commands,device):
    n=len(commands)
    return tuple(torch.as_tensor(np.array(v,copy=True),device=device,dtype=torch.float32) for v in
                 [np.repeat(field[None],n,0),np.repeat(obs.passport[None],n,0),np.repeat(obs.actuator[None],n,0),commands])


def operator_metrics(predicted,raw,reference,initial,theta,p,u,obs,labels,gap):
    target=torch.as_tensor(restrict(reference,32),device=predicted.device,dtype=torch.float32)
    diff=predicted-target; spatial=diff-diff.mean((-2,-1),keepdim=True)
    temporal=torch.diff(torch.cat([initial[:,None],predicted],1),dim=1)-torch.diff(torch.cat([initial[:,None],target],1),dim=1)
    active=np.array(labels['valid']); active[0]=True
    delta=(predicted-predicted[:1])-(target-target[:1]); forecast=predicted.cpu().numpy()
    estimated=costs(forecast,labels['commands'],obs.previous,obs.goal)
    ok,total=order_accuracy(estimated,labels['cost'],labels['eligible'],gap)
    feasible=labels['valid']&(forecast.max((1,2,3))<=120)
    choice=int(np.where(feasible,estimated,np.inf).argmin()) if feasible.any() else -1
    reg=decision_metrics(choice,obs,labels,gap)
    heat=heat_sequence(p,u,theta); r=residual(predicted,initial,heat,theta); rr=residual(raw,initial,heat,theta)
    return {'horizon_mse':float(diff[:,-1].square().mean()),'field_mse_by_horizon':diff.square().mean((0,2,3)).cpu().tolist(),
            'temporal_delta_mse_by_horizon':temporal.square().mean((0,2,3)).cpu().tolist(),
            'spatial_deviation_mse':float(spatial.square().mean()),'mean_temperature_mse':float(diff.mean((-2,-1)).square().mean()),
            'candidate_delta_mse':float(delta[active].square().mean()),'cost_mae':float(abs(estimated[labels['valid']]-labels['cost'][labels['valid']]).mean()),
            'peak_error_k':float(abs(forecast.max((2,3))-reference.max((2,3))).max()),
            'order_correct':ok,'order_pairs':total,'local_pde_mae_w_m2':float(r.abs().mean()),
            'raw_global_balance_mae_w_m2':float(rr.mean((-2,-1)).abs().mean()),
            'projected_global_balance_mae_w_m2':float(r.mean((-2,-1)).abs().mean()),**reg}


def decision_metrics(choice,obs,labels,gap=1e-6,error=None):
    available=bool(obs.usable and labels['eligible'].any())
    valid_choice=choice is not None and 0<=choice<9
    infeasible=bool(valid_choice and not labels['eligible'][choice])
    reg=float(labels['cost'][choice]-labels['cost'][labels['eligible']].min()) if valid_choice and available else None
    return {'choice':int(choice) if valid_choice else -1,'regret':reg,'expert_feasible':available,
            'expert_equivalent':bool(valid_choice and not infeasible and reg is not None and reg<=gap),
            'infeasible':infeasible,'fallback':not valid_choice,'unexpected_fallback':available and not valid_choice,
            'unusable':not obs.usable,'error':error,'failure':bool(error or infeasible or (available and not valid_choice))}


def summarize_decisions(rows,method):
    values=[r['methods'][method] for r in rows]
    return {'rows':len(values),'regret':quantiles([r['regret'] for r in values if r.get('regret') is not None]),
            **{k:sum(bool(r.get(k)) for r in values) for k in ['expert_equivalent','infeasible','fallback','unexpected_fallback','unusable','failure']}}


def operator_ablation(root,run,config,output,device='cuda'):
    setup_torch(config,device); data=EvaluationData(run)
    models={'balance_only':BalanceOnly().to(device)}
    for variant,folder in [('fno_balance','14_fno_balance'),('pino_balance','07_pino_balance')]:
        for label,filename in [('initial','initial.pt'),('selected','best.pt'),('last','last.pt')]:
            model=ThermalOperator(variant).to(device)
            model.load_state_dict(torch.load(run/folder/filename,map_location=device,weights_only=True))
            models[variant+'_'+label]=model.eval()
    # The two trainable arms must start at identical neural parameters.
    for k,v in models['fno_balance_initial'].state_dict().items():
        require(torch.equal(v,models['pino_balance_initial'].state_dict()[k]),'Unmatched operator initialization')
    journal=Journal(output/'operator_rows.jsonl',{'plan':digest(data.folder/'plan.json'),'protocol':config['validation']})
    stop=StopRequest()
    try:
        for index,(entry,step,obs,labels) in enumerate(data.rows()):
            if index<len(journal.rows): continue
            row={'episode':entry['episode_id'],'stratum':entry['stratum'],'step':step,'usable':obs.usable,'methods':{}}
            if obs.usable:
                reference=branch(labels['field'],obs.actuator,labels['commands'],obs.passport)
                inputs=tensor_inputs(labels['field'],obs,labels['commands'],device)
                with torch.no_grad():
                    for name,model in models.items():
                        predicted,raw=model(*inputs,return_raw=True)
                        row['methods'][name]=operator_metrics(predicted,raw,reference,*inputs,obs,labels,
                                                              config['acceptance']['solver']['significant_cost_gap'])
            else: row['methods']={name:decision_metrics(-1,obs,labels) for name in models}
            journal.append(row)
            if (index+1)%32==0: print(f'PV1 {index+1} decision states',flush=True)
            if stop.signal: raise InterruptedError('PV1 row committed; resume supported')
    finally: data.close(); stop.close()
    refinements=[read_json(data.folder/'labels'/e['episode_id']/'refinement.json') for e in data.entries]
    numerical_floor=np.sqrt(mean([r['candidate_delta_mse'] for r in refinements if r is not None]) or 0.)
    summaries={}; comparisons={}
    for name in models:
        usable=[r['methods'][name] for r in journal.rows if r['usable']]
        summaries[name]={**summarize_decisions(journal.rows,name),
                         'horizon_rmse_k':np.sqrt(mean([r['horizon_mse'] for r in usable])) if usable else None,
                         'candidate_delta_rmse_k':np.sqrt(mean([r['candidate_delta_mse'] for r in usable])) if usable else None,
                         'peak_error_k':max([r['peak_error_k'] for r in usable],default=None),
                         'order_accuracy':sum(r['order_correct'] for r in usable)/sum(r['order_pairs'] for r in usable) if sum(r['order_pairs'] for r in usable) else None,
                         'spatial_deviation_mse':mean([r['spatial_deviation_mse'] for r in usable])}
        if name.endswith('_selected'):
            comparison={m:paired(journal.rows,name,'balance_only',m,config) for m in ['candidate_delta_mse','spatial_deviation_mse','regret']}
            criteria=(comparison['candidate_delta_mse']['interpretation']=='POSITIVE' and
                      comparison['spatial_deviation_mse']['interpretation']=='POSITIVE' and
                      comparison['regret']['mean_difference'] is not None and comparison['regret']['mean_difference']<=0 and
                      summaries[name]['peak_error_k']<=config['acceptance']['operator']['peak_abs_error_k'] and
                      summaries['balance_only']['candidate_delta_rmse_k']-summaries[name]['candidate_delta_rmse_k']>numerical_floor and
                      summaries[name]['failure']<=summaries['balance_only']['failure'])
            negative=comparison['candidate_delta_mse']['interpretation']=='NEGATIVE'
            comparisons[name]={'paired':comparison,'interpretation':'POSITIVE' if criteria else ('NEGATIVE' if negative else 'INCONCLUSIVE')}
    # Existing contiguous rollout validation starts from true T32, separately from observed-state branches.
    from .training import Supervision,evaluate_operator
    base=run/'01_generate/data'; truth=Supervision(base,read_json(base/'manifests/split.json')['episodes'])
    val=np.load(run/'04_operator_data/validation_windows.npy')
    try: truth_metrics={name:evaluate_operator(model,truth,val,device) for name,model in models.items()}
    finally: truth.close()
    result={'PV1':comparisons,'observed_state':summaries,'true_state_contiguous_rollout':truth_metrics,
            'trained_arms':{folder:read_json(run/folder/'training.json')['selected_step'] for folder in ['14_fno_balance','07_pino_balance']},
            'rows':len(journal.rows),'test_accessed':False,'pde_effect_is_bundled_with_delta_loss':True,
            'sampled_refinement_candidate_delta_rmse_k':float(numerical_floor),'refinement_sampling':'first usable anchor per episode; diagnostic, not a global error bound'}
    write_json(output/'operator_ablation.json',result); return result
