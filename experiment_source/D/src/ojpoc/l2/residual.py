"""Paired PINO cost correction: common observed-state FV labels, no test fitting."""
from copy import deepcopy
import gc
import hashlib
import time
import numpy as np
import torch
from ..common import (read_json, write_json, require, digest, seal, verify, save_array,
                      setup_torch, StopRequest, save_training, restore_training)
from .cache import Rows, SHAPES
from .contracts import entries


def pino_cost(commands, features):
    """The existing cost formula, entirely from deployable predicted inputs."""
    return features[..., 0] + .5*features[..., 1] + .002*commands.square().mean((-1,-2)) + .02*features[..., 8]


def corrected_scores(commands, features, raw, scale, bound, alpha):
    base=pino_cost(commands,features)
    # Center in FP64 to preserve tiny candidate differences and permutation parity.
    relative=((base.double()-base.double().mean(-1,keepdim=True))/scale).float()
    return -relative-alpha*bound*torch.tanh(raw.float())


def residual_loss(scores,batch,scale,temperature):
    mask=batch['valid'].bool() & batch['eligible'].bool()
    require(bool(mask.any(-1).all()),'No feasible residual targets')
    weight=mask.float();denom=weight.sum(-1,keepdim=True)
    def centered(x):return x-(x*weight).sum(-1,keepdim=True)/denom
    target=-centered(batch['cost'].double()).float()/scale
    regression=torch.nn.functional.smooth_l1_loss(centered(scores)[mask],target[mask])
    logits=(scores*scale/temperature).masked_fill(~batch['valid'].bool(),-torch.inf)
    logp=logits.log_softmax(-1);positive=batch['target']>0
    ce=-(batch['target'][positive]*logp[positive]).sum()/len(scores)
    return regression+.2*ce


def prepare_data(root,run,cfg,output):
    from .engine import Engine
    from .rollout import Collector,delay_trace
    from .control import simulate
    from .data import Plant
    setup_torch(cfg);spec=cfg['l2']['residual_learning']['profiles'][cfg['profile']]
    selected=entries(run,['train'])[:spec['train_episodes']]
    require(selected and all(e['split']=='train' for e in selected),'Collection is train only')
    steps=np.linspace(1,round(cfg['budget']['duration_s']*10)-21,spec['anchors'],dtype=int)
    require(len(set(steps))==len(steps),'Repeated collection anchors')
    norm=read_json(run/'03_index/normalization.json')
    temperature=read_json(run/'05_cache/both/train/metadata.json')['label_temperature']
    folder=output/'train';folder.mkdir(parents=True,exist_ok=True);parts=[];records=[];stop=StopRequest()
    try:
        for method in ['pi','pino_mpc']:
            engine=Engine(root,run,method,calibrated=False)
            try:
                for i,entry in enumerate(selected):
                    path=folder/(method+'_'+entry['episode_id']+'.npz');meta=path.with_suffix('.json')
                    delay=['zero','38ms'][i%2]
                    if not meta.exists():
                        tap=Collector(engine,entry,steps,norm,temperature)
                        metrics,_=simulate(Plant(run/'02_generate/data',entry,cfg['budget']['duration_s']),tap,'delayed',delay_trace(delay),stop)
                        require(len(tap.rows)==len(steps),'Incomplete common collection')
                        with path.with_suffix('.tmp').open('wb') as stream:
                            np.savez(stream,**{k:np.stack([np.asarray(r[k],dtype=(bool if k in ['valid','eligible','usable'] else np.int64 if k=='index' else np.float32)) for r in tap.rows]) for k in SHAPES})
                        path.with_suffix('.tmp').replace(path)
                        write_json(meta,{'method':method,'episode':entry['episode_id'],'split':'train','delay':delay,'sha256':digest(path),'metrics':metrics})
                    record=read_json(meta);require(digest(path)==record['sha256'],'Common collection changed')
                    records.append(record);parts.append(path)
                    print(f'Residual common train {method} {i+1}/{len(selected)}',flush=True)
            finally:engine.close()
        for k in SHAPES:
            chunks=[]
            for path in parts:
                with np.load(path,allow_pickle=False) as z:chunks.append(z[k])
            save_array(folder/(k+'.npy'),np.concatenate(chunks))
        write_json(folder/'metadata.json',{'split':'train','rows':len(parts)*len(steps),'label_temperature':temperature,
                   'records':records,'labels':'FV64 2-second held-action cost from the same observed field',
                   'identical_data_for_both_arms':True,'collector_policies':['pi','pino_mpc']})
        seal(folder)
        sources=[Rows(run/'05_cache/both/train'),Rows(folder)];errors=[];source_files={}
        try:
            for rows in sources:
                require(rows.meta['split']=='train','Residual scale is train-only')
                for i in range(len(rows)):
                    mask=rows.arrays['eligible'][i]&rows.arrays['valid'][i]
                    if not rows.arrays['usable'][i] or not mask.any():continue
                    b=rows.batch([i],device='cpu')
                    d=np.asarray(rows.arrays['cost'][i],np.float64)-pino_cost(b['commands'],b['features'])[0].numpy()
                    errors.extend((d[mask]-d[mask].mean()).tolist())
                for f in rows.folder.glob('*.npy'):source_files[str(f.relative_to(run))]=digest(f)
        finally:
            for rows in sources:rows.close()
        require(errors and np.isfinite(errors).all(),'Invalid residual training differences')
        result={'cost_scale':max(1e-6,float(np.sqrt(np.mean(np.square(errors))))),
                'scale_definition':'RMS of candidate-centered FV minus PINO costs on feasible train candidates only',
                'source_files':source_files,'candidate_count':len(errors),'shared_training_data':True,
                'test_accessed':False,'label_mode':'fv_hold_2s','original_fraction_per_batch':.5}
        write_json(output/'statistics.json',result);return result
    finally:stop.close()


def selection_config(cfg):
    result=deepcopy(cfg)
    result['l2']['rollout_learning']={'profiles':{cfg['profile']:{'delays':cfg['l2']['residual_learning']['selection_delays']}}}
    return result


def select_alpha(root,run,cfg,output,method,model,target):
    from .rollout import evaluate_loops,selection_key
    from .models import export
    selected_cfg=selection_config(cfg)
    refs={m:evaluate_loops(root,run,selected_cfg,run/'residual_validation_reference'/m,m)
          for m in ['pi','pino_mpc']}
    candidates=[]
    for alpha in cfg['l2']['residual_learning']['alphas']:
        model.residual['alpha']=alpha
        rows=evaluate_loops(root,run,selected_cfg,output/'selection'/f'alpha_{alpha:g}',method,model,target/'fit_best')
        key=selection_key(rows,refs)
        candidates.append({'alpha':alpha,'key':key,'rows':rows})
    selected=min(candidates,key=lambda c:(*c['key'],c['alpha']))
    model.residual['alpha']=selected['alpha']
    export(model,target/'best',{'method':method,'selection':'validation paired control alpha','training_seed':cfg['seed']})
    result={'selected_alpha':selected['alpha'],'candidates':candidates,'split':'validation',
            'criterion':'safety, worst all/normal reference cost ratio over zero/38ms, mean ratio, lower alpha',
            'synthetic_delay_is_not_E2E':True,'zero_alpha_is_not_evidence_of_JEV_value':True}
    write_json(output/'selection.json',result);return result


def train(root,run,cfg,output,method):
    from .models import load_policy,export,restore
    from .training import evaluate,scores_at
    from .rollout import mixed_batch
    setup_torch(cfg);model=load_policy(root,run,method,training=True)
    original=Rows(run/'05_cache/both/train');extra=Rows(run/'05_residual_data/train');val=Rows(run/'05_cache/both/validation')
    stats=read_json(run/'05_residual_data/statistics.json')
    require(all(digest(run/p)==h for p,h in stats['source_files'].items()),'Common train data changed')
    target=run/'models'/method;target.mkdir(parents=True,exist_ok=True)
    baseline=evaluate(model,val);best=baseline['selection_score'];history=[];selected_step=['initial',0]
    if not (target/'fit_best').exists():export(model,target/'fit_best',{'method':method,'selection':selected_step})
    rng=np.random.default_rng(cfg['seed']+211);stop=StopRequest();batch_records=[]
    base=[(p,p._version) for n,p in model.named_parameters() if n.startswith('backbone.') and 'lora_' not in n]
    phases=[('warmup',cfg['budget']['shared_warmup']),('adapt',cfg['budget']['shared_adapt'])]
    try:
        for phase,steps in phases:
            model.residual['alpha']=1.
            for n,p in model.named_parameters():p.requires_grad_(not n.startswith('backbone.') or ('lora_' in n and phase=='adapt'))
            if hasattr(model,'backbone'):model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            new=[p for n,p in named.items() if 'lora_' not in n];lora=[p for n,p in named.items() if 'lora_' in n]
            optimizer=torch.optim.AdamW([{'params':new,'lr':1e-4}]+([{'params':lora,'lr':3e-5}] if lora else []),weight_decay=.01,foreach=False)
            restart=output/(phase+'_restart.pt');start=0
            if restart.exists():
                start,state=restore_training(restart,model,optimizer,rng,'cuda')
                history=state['history'];best=state['best'];selected_step=state['selected_step'];batch_records=state['batch_records']
            before={n:p.detach().cpu().clone() for n,p in named.items()}
            for step in range(start+1,steps+1):
                model.train();optimizer.zero_grad(set_to_none=True);begin=time.perf_counter()
                batch=mixed_batch(original,[extra],rng,cfg['budget']['accumulation'])
                # Same RNG, batch size, source mix and permutation for both architectures.
                order=rng.permutation(9)
                h=hashlib.sha256()
                for k in ['index','commands','cost']:
                    h.update(batch[k].detach().cpu().contiguous().numpy().tobytes())
                h.update(order.tobytes());batch_records.append(h.hexdigest())
                scores=model(batch,batch['commands'][:,order],batch['features'][:,order])[0]
                labels={k:batch[k][:,order] for k in ['cost','valid','eligible','target']}
                loss=residual_loss(scores,labels,model.residual['scale'],original.meta['label_temperature'])
                require(bool(torch.isfinite(loss)),'Nonfinite residual loss');loss.backward()
                gn=sum(float(p.grad.abs().sum()) for p in new if p.grad is not None)
                gl=sum(float(p.grad.abs().sum()) for p in lora if p.grad is not None)
                require(gn>0 and np.isfinite(gn) and np.isfinite(gl),'Missing/nonfinite residual gradients')
                torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True);optimizer.step()
                require(all(p.grad is None and p._version==v for p,v in base),'Frozen base changed')
                row={'phase':phase,'step':step,'loss':float(loss.detach()),'new_gradient':gn,'lora_gradient':gl,
                     'seconds':time.perf_counter()-begin,'global_decisions':len(scores)}
                history.append(row)
                if step%cfg['budget']['validation_every']==0 or step==steps:
                    row['validation']=evaluate(model,val)
                    if row['validation']['selection_score']<best:
                        best=row['validation']['selection_score'];selected_step=[phase,step]
                        export(model,target/'fit_best',{'method':method,'selection':selected_step})
                if step%25==0 or step==steps or stop.signal:
                    save_training(restart,model,optimizer,step,rng,{'history':history,'best':best,'selected_step':selected_step,'batch_records':batch_records})
                print(f'Residual {method} {phase} {step}/{steps} loss={float(loss):.6g}',flush=True)
                if stop.signal:raise InterruptedError('Residual optimizer boundary saved')
            changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in named.items()}
            if start<steps:
                require(any(v>0 for n,v in changes.items() if 'lora_' not in n),'Residual head did not update')
                if lora:require(any(v>0 for n,v in changes.items() if 'lora_' in n),'Residual LoRA did not update')
            write_json(output/(phase+'_updates.json'),changes)
        model.residual['alpha']=1.;restore(model,target/'fit_best')
        selection=select_alpha(root,run,cfg,output,method,model,target)
        selected=evaluate(model,val)
        with torch.no_grad():scores=scores_at(model,val,0).float().cpu().tolist()
        write_json(output/'reload_reference.json',{'scores':scores,'row':0,'method':method})
        result={'method':method,'baseline':baseline,'selected':selected,'selected_step':selected_step,
                'history':history,'optimizer_steps':len(history),'frozen_base_unchanged':True,'test_accessed':False,
                'quality_pass':selected['mean_regret'] is not None and selected['mean_regret']<baseline['mean_regret'] and
                               selected['infeasible']==0 and selected['unexpected_fallback']==0,
                'selection':selection,'batch_sequence':batch_records,'training_data_hash':digest(run/'05_residual_data/statistics.json'),
                'comparison':'identical examples, order, teacher, global batch and optimizer steps; different architecture',
                'scope':'FV hold-2s residual supervision, not long-horizon RL or full-loop differentiation'}
        write_json(output/'training.json',result);return result
    finally:
        original.close();extra.close();val.close();stop.close()


def parity_check(root,run,cfg,output,method):
    from .models import load_policy
    from .graph import GraphPolicy
    setup_torch(cfg);model=load_policy(root,run,method,inference_backend='eager')
    rows=Rows(run/'05_cache/both/validation');checks=[]
    try:
        selected_alpha=model.residual['alpha'];model.residual['alpha']=0.
        for i in range(min(len(rows),8)):
            batch=rows.batch([i]);base=pino_cost(batch['commands'],batch['features'])
            with torch.no_grad():score,_=model(batch,batch['commands'],batch['features'])
            allowed=batch['valid'].bool() & (batch['features'][...,2]<=120)
            require(torch.equal((-base).masked_fill(~allowed,-torch.inf).argmax(-1),score.masked_fill(~allowed,-torch.inf).argmax(-1)),
                    'Zero residual changed PINO choice')
        # Check graph refreshing on multiple states with a nonzero correction.
        model.residual['alpha']=1.;model.eval()
        probe_perturbed=not bool(model.scorer[-1].weight.count_nonzero())
        if probe_perturbed:
            with torch.no_grad():model.scorer[-1].weight.fill_(.01)
        graph=GraphPolicy(model).eval()
        for i in range(min(len(rows),8)):
            b=rows.batch([i])
            with torch.no_grad():
                expected=model(b,b['commands'],b['features'])[0]
                actual=graph(b,b['commands'],b['features'])[0]
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
                checks.append(float((actual-expected).abs().max()))
        result={'passed':True,'method':method,'zero_alpha_matches_PINO':True,'graph_max_error':max(checks),
                'checked_rows':len(checks),'selected_alpha':selected_alpha,'selected_checkpoint_modified':False,
                'in_memory_nonzero_probe':probe_perturbed}
        write_json(output/('residual_parity_'+method+'.json'),result);return result
    finally:rows.close()
