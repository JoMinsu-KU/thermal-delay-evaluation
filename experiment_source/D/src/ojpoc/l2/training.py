"""Train-only fitting, validation-only selection, atomic optimizer-boundary resume."""
import time
import numpy as np
import torch
from ..common import read_json, write_json, require, save_training, restore_training, StopRequest, setup_torch
from ..data import EpisodeCache
from ..policy import serialize, tokenize, policy_loss
from .contracts import entries, modality, has_physics
from .cache import Rows
from .models import load_policy, export, restore


def token_cache(model,run,rows,output,split):
    output.mkdir(exist_ok=True,parents=True)
    ids=np.lib.format.open_memmap(output/(split+'_ids.npy'),mode='w+',dtype=np.int32,shape=(len(rows),9,384))
    mask=np.lib.format.open_memmap(output/(split+'_mask.npy'),mode='w+',dtype=bool,shape=(len(rows),9,384))
    ids[:]=model.tokenizer.pad_token_id;mask[:]=False
    raw=EpisodeCache(run/'02_generate/data',entries(run,['train','validation']))
    try:
        for i,(code,step) in enumerate(rows.arrays['index']):
            obs=raw.get(code).snapshot(step)
            record=serialize(obs,rows.arrays['field'][i],rows.arrays['commands'][i],
                             rows.arrays['features'][i] if has_physics(model.method) else None)
            encoded=tokenize(model,record);width=encoded['input_ids'].shape[1]
            ids[i,:,:width]=encoded['input_ids'].numpy();mask[i,:,:width]=encoded['attention_mask'].numpy().astype(bool)
        ids.flush();mask.flush()
    finally:raw.close();ids._mmap.close();mask._mmap.close()
    return tuple(np.load(output/(split+'_'+suffix+'.npy'),mmap_mode='r') for suffix in ['ids','mask'])


def scores_at(model,rows,i,tokens=None,order=None):
    order=np.arange(9) if order is None else order
    if tokens is not None:
        ids,mask=[np.array(x[i][order],copy=True) for x in tokens];length=int(mask.sum(-1).max())
        return model(torch.tensor(ids[:,:length],device='cuda',dtype=torch.long),
                     torch.tensor(mask[:,:length],device='cuda',dtype=torch.long))
    batch=rows.batch([i]);commands=batch['commands'][:,order];features=batch['features'][:,order]
    return model(batch,commands,features)[0][0]


def evaluate(model,rows,tokens=None):
    model.eval();regrets=[];wrong=0;missing=0;expected=0;feasible=0
    with torch.no_grad():
        for i in range(len(rows)):
            eligible=np.asarray(rows.arrays['eligible'][i]);expected+=int(eligible.any())
            if not rows.arrays['usable'][i]:missing+=int(eligible.any());continue
            scores=scores_at(model,rows,i,tokens).float().cpu().numpy()
            require(np.isfinite(scores).all(),'Nonfinite policy logits')
            allowed=rows.arrays['valid'][i]&(rows.arrays['features'][i,:,2]<=120)
            if not allowed.any():missing+=int(eligible.any());continue
            choice=int(np.where(allowed,scores,-np.inf).argmax())
            wrong+=int(not eligible[choice]);feasible+=int(eligible[choice])
            if eligible.any():regrets.append(float(rows.arrays['cost'][i,choice]-rows.arrays['cost'][i,eligible].min()))
    return {'rows':len(rows),'expected_feasible':expected,'regret_count':len(regrets),
            'mean_regret':float(np.mean(regrets)) if regrets else None,'infeasible':wrong,'unexpected_fallback':missing,
            'selection_score':(float(np.mean(regrets)) if regrets else 1.)+(wrong+missing)/max(1,len(rows)),
            'selection_scope':'validation only; penalty fixed before calibration/test'}


def train(root,run,cfg,output,method):
    setup_torch(cfg);model=load_policy(root,run,method,training=True);mode=modality(method)
    train=Rows(run/'05_cache'/mode/'train');val=Rows(run/'05_cache'/mode/'validation')
    tokens={};history=[];coverage=set();rng=np.random.default_rng(cfg['seed']+101)
    if method.startswith('native'):
        for split,rows in [('train',train),('validation',val)]:tokens[split]=token_cache(model,run,rows,output,split)
    baseline=evaluate(model,val,tokens.get('validation'))
    target=run/'models'/method;target.mkdir(parents=True,exist_ok=True)
    best=baseline['selection_score'];selected_step=['initial',0]
    if not (target/'best').exists():export(model,target/'best',{'method':method,'selection':selected_step,'training_seed':cfg['seed']})
    eligible=np.flatnonzero(np.asarray(train.arrays['target']).sum(1)>0);require(len(eligible)>0,'No expert-feasible train examples')
    stop=StopRequest();budget=cfg['budget'];native=method.startswith('native');small=method=='small_physics'
    phases=[('supervised',budget['small_steps'])] if small else [('warmup',budget['native_warmup' if native else 'shared_warmup']),('adapt',budget['native_adapt' if native else 'shared_adapt'])]
    base=[(p,p._version) for n,p in model.named_parameters() if n.startswith('backbone.') and 'lora_' not in n]
    try:
        for phase,steps in phases:
            for name,p in model.named_parameters():p.requires_grad_(not name.startswith('backbone.') or ('lora_' in name and phase=='adapt'))
            if hasattr(model,'backbone'):model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            lora=[p for n,p in named.items() if 'lora_' in n];new=[p for n,p in named.items() if 'lora_' not in n]
            groups=[{'params':new,'lr':.0001 if native or phase=='adapt' else .0003}]
            if lora:groups.append({'params':lora,'lr':.00005 if native else .00003})
            optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
            rates=[g['lr'] for g in groups];start=0;restart=output/(phase+'_restart.pt')
            if restart.exists():
                start,extra=restore_training(restart,model,optimizer,rng,'cuda');history=extra['history'];best=extra['best']
                coverage=set(extra['coverage']);selected_step=extra['selected_step']
            changes={n:p.detach().cpu().clone() for n,p in named.items()}
            for step in range(start+1,steps+1):
                model.train();optimizer.zero_grad(set_to_none=True);started=time.perf_counter();total=0.
                micro=1 if native or small else min(budget['accumulation'],int(cfg['l2'].get('shared_microbatch',1)))
                require(micro>=1 and budget['accumulation']%micro==0,'Microbatch must divide global decision batch')
                accumulation=2 if small and cfg['profile']!='smoke' else budget['accumulation']//micro
                for _ in range(accumulation):
                    count=min(budget['small_batch'],len(eligible)) if small else micro
                    indices=rng.choice(eligible,size=count,replace=True);coverage.update(map(int,indices))
                    if small or (not native and micro>1):
                        batch=train.batch(indices);order=rng.permutation(9)
                        scores=model(batch,batch['commands'][:,order],batch['features'][:,order])[0]
                        losses=[policy_loss(scores[j],batch['target'][j,order],batch['cost'][j,order],batch['valid'][j,order],batch['eligible'][j,order],1e-6) for j in range(count)]
                        loss=torch.stack(losses).mean()
                    else:
                        i=int(indices[0]);order=rng.permutation(9);scores=scores_at(model,train,i,tokens.get('train'),order)
                        tensors=[torch.tensor(np.array(train.arrays[k][i,order]),device='cuda') for k in ['target','cost','valid','eligible']]
                        loss=policy_loss(scores,*tensors,1e-6)
                    require(torch.isfinite(loss).item(),'Nonfinite loss')
                    (loss/accumulation).backward();total+=float(loss.detach())/accumulation
                grad_new=sum(float(p.grad.float().abs().sum()) for p in new if p.grad is not None)
                grad_lora=sum(float(p.grad.float().abs().sum()) for p in lora if p.grad is not None)
                require(grad_new>0 and (not lora or grad_lora>0),'Missing new-module/LoRA gradients')
                torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True)
                for g,rate in zip(optimizer.param_groups,rates):g['lr']=rate*min(1.,step/max(1,int(np.ceil(steps*.05))))
                optimizer.step();require(all(p.grad is None and p._version==v for p,v in base),'Frozen base changed')
                row={'phase':phase,'step':step,'seconds':time.perf_counter()-started,'loss':total,'new_gradient':grad_new,'lora_gradient':grad_lora,
                     'microbatch':count,'accumulation':accumulation,'global_decisions':count*accumulation}
                history.append(row)
                if step%budget['validation_every']==0 or step==steps:
                    row['validation']=evaluate(model,val,tokens.get('validation'))
                    if row['validation']['selection_score']<best:
                        best=row['validation']['selection_score'];selected_step=[phase,step]
                        export(model,target/'best',{'method':method,'selection':selected_step,'training_seed':cfg['seed']})
                if step%250==0 or step==steps or stop.signal:
                    save_training(restart,model,optimizer,step,rng,{'history':history,'best':best,'coverage':sorted(coverage),'selected_step':selected_step})
                print(f'L2 {method} {phase} {step}/{steps} loss={total:.6g}',flush=True)
                if stop.signal:raise InterruptedError('Committed optimizer boundary; resume this run')
            changed={n:float((p.detach().cpu()-changes[n]).abs().max()) for n,p in named.items()}
            if start<steps:
                require(any(v>0 for n,v in changed.items() if 'lora_' not in n),'New modules did not update')
                if lora:require(any(v>0 for n,v in changed.items() if 'lora_' in n),'LoRA did not update')
            write_json(output/(phase+'_updates.json'),changed)
            export(model,target/'last',{'method':method,'phase':phase,'step':steps,'training_seed':cfg['seed']})
        restore(model,target/'best')
        rollout_result=None
        if cfg['l2'].get('rollout_learning',{}).get('enabled',False):
            from .rollout import METHODS,refine
            if method in METHODS:
                rollout_result=refine(root,run,cfg,output,method,model,train,val,baseline,stop)
                history+=rollout_result['history'];selected_step=['closed_loop',rollout_result['selected_round']]
        selected=evaluate(model,val,tokens.get('validation'))
        model.eval()
        with torch.no_grad():scores=scores_at(model,val,0,tokens.get('validation')).float().cpu().tolist()
        write_json(output/'reload_reference.json',{'scores':scores,'row':0,'method':method})
        result={'method':method,'baseline':baseline,'selected':selected,'selected_step':selected_step,'history':history,
                'unique_train_rows':len(coverage),'train_rows':len(train),'optimizer_steps':sum(n for _,n in phases)+(len(rollout_result['history']) if rollout_result else 0),
                'quality_pass':selected['mean_regret'] is not None and baseline['mean_regret'] is not None and selected['mean_regret']<baseline['mean_regret'] and selected['infeasible']==0 and selected['unexpected_fallback']==0,
                'frozen_base_unchanged':True,'test_accessed':False,'base_weights_included':False,
                'rollout_selection':rollout_result}
        write_json(output/'training.json',result);return result
    finally:
        train.close();val.close();stop.close()
        for ids,mask in tokens.values():ids._mmap.close();mask._mmap.close()


def reload_check(root,run,cfg,output,method,training_stage):
    setup_torch(cfg);model=load_policy(root,run,method);rows=Rows(run/'05_cache'/modality(method)/'validation')
    reference=read_json(run/training_stage/'reload_reference.json');tokens=None
    if method.startswith('native'):
        tokens=tuple(np.load(run/training_stage/('validation_'+k+'.npy'),mmap_mode='r') for k in ['ids','mask'])
    try:
        with torch.no_grad():scores=scores_at(model,rows,reference['row'],tokens).float().cpu()
        expected=torch.tensor(reference['scores']);torch.testing.assert_close(scores,expected,atol=.03,rtol=.01)
        actual=scores.softmax(-1);target=expected.softmax(-1)
        torch.testing.assert_close(actual,target,atol=.001,rtol=.01)
        require(int(actual.argmax())==int(target.argmax()),'Reload changed canonical choice')
        return {'method':method,'score_max_error':float((scores-expected).abs().max()),'probability_max_error':float((actual-target).abs().max()),'choice':int(actual.argmax()),'fresh_process':True}
    finally:
        rows.close()
        if tokens:
            for a in tokens:a._mmap.close()


def resume_probe(root,run,cfg,output):
    """Actual shared Qwen/LoRA optimizer continuation, using train inputs only."""
    from ..common import digest
    setup_torch(cfg);model=load_policy(root,run,'shared_physics',inference_backend='eager');rows=Rows(run/'05_cache/both/train')
    if getattr(model,'residual',None):model.residual['alpha']=1.  # Probe actual updates even when validation selects alpha=0.
    for n,p in model.named_parameters():p.requires_grad_(not n.startswith('backbone.') or 'lora_' in n)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(list(named.values()),lr=1e-5,foreach=False)
    rng=np.random.default_rng(cfg['seed']+9911);eligible=np.flatnonzero(np.asarray(rows.arrays['target']).sum(1)>0)
    micro=min(cfg['budget']['accumulation'],int(cfg['l2'].get('shared_microbatch',1)))
    accumulation=cfg['budget']['accumulation']//micro
    def step():
        model.train();optimizer.zero_grad(set_to_none=True);total=0.
        for _ in range(accumulation):
            indices=rng.choice(eligible,size=micro,replace=True);order=rng.permutation(9);batch=rows.batch(indices)
            scores=model(batch,batch['commands'][:,order],batch['features'][:,order])[0]
            if getattr(model,'residual',None):
                from .residual import residual_loss
                loss=residual_loss(scores,{k:batch[k][:,order] for k in ['cost','valid','eligible','target']},
                                   model.residual['scale'],rows.meta['label_temperature'])
            else:loss=torch.stack([policy_loss(scores[j],*[batch[k][j,order] for k in ['target','cost','valid','eligible']],1e-6) for j in range(micro)]).mean()
            (loss/accumulation).backward();total+=float(loss.detach())/accumulation
        optimizer.step();return total
    try:
        step();checkpoint=output/'resume_probe.pt';save_training(checkpoint,model,optimizer,1,rng,{'scope':'diagnostic_train_only'})
        expected_loss=step();expected={n:p.detach().clone() for n,p in named.items()}
        restore_training(checkpoint,model,optimizer,rng,'cuda');actual_loss=step();maximum=0.
        for name,p in named.items():
            torch.testing.assert_close(p,expected[name],atol=1e-6,rtol=1e-5)
            maximum=max(maximum,float((p-expected[name]).abs().max()))
        require(abs(actual_loss-expected_loss)<=1e-5,'Resume loss changed')
        result={'passed':True,'model':'shared_physics','optimizer_boundary':1,'continued_step':2,
                'loss_error':abs(actual_loss-expected_loss),'parameter_max_error':maximum,
                'checkpoint_sha256':digest(checkpoint),'train_only':True,'selected_checkpoint_modified':False,
                'microbatch':micro,'accumulation':accumulation,
                'scope':'actual Qwen/LoRA optimizer and sampling RNG continuation at configured batch; not an OS kill test'}
        write_json(output/'resume_probe.json',result);return result
    finally:rows.close()
