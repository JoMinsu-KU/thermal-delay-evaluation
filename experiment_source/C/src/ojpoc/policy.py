"""Native Open-Jev adaptation to independent FV labels; frozen PINO features."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .common import (require, read_json, write_json, save_array, seal, verify, digest,
                     StopRequest, save_training, restore_training, object_hash)
from .data import EpisodeCache
from .physics import CANDIDATES


def serialize(obs,field,commands,features=None,order=None):
    """Only causal observation summaries and predicted features enter these records."""
    from jev.api import compile_request
    def fmt(x,places=1): return ','.join(f'{float(v):.{places}f}' for v in np.asarray(x).ravel())
    current=obs.sensor[-1,:8]; mask=obs.sensor_valid[:,:8]
    means=(obs.sensor[:,:8]*mask).sum(0)/np.maximum(mask.sum(0),1)
    changes=[]
    for j in range(8):
        ids=np.flatnonzero(mask[:,j]); changes.append(float(current[j]-obs.sensor[ids[0],j]) if len(ids) else 0.)
    # Four zone means and pooled probe history retain compact, explicit units.
    zones=np.array([field[:16,:16].mean(),field[:16,16:].mean(),field[16:,:16].mean(),field[16:,16:].mean()])
    theta=obs.passport
    state=(f'Thermal simulation. C probe={fmt(current)}; zone={fmt(zones)}; goal={obs.goal:.1f}. '
           f'History mean={float(means.mean()):.1f} C, change={float(np.mean(changes)):.2f} K. '
           f'u={fmt(obs.previous,2)}; p={fmt(obs.actuator,2)}. '
           f'Passport SI C={theta[0]:.0f},k={theta[1]:.2f},h={theta[2]:.1f},Tamb={theta[3]:.1f},q={theta[4]:.0f},tau={fmt(theta[5:],2)}. '
           f'Age={float(obs.thermal_age[-1]):.2f}s; probe_valid={int(obs.sensor_valid[-1,:8].sum())}/8.')
    order=np.arange(9) if order is None else np.asarray(order)
    options={}
    for k in order:
        text=f'u={fmt(commands[k,0],2)} for 2s'
        if features is not None:
            f=features[k]
            text+=f'; PINO tracking={f[0]:.5f},terminal={f[1]:.5f},peak={f[2]:.1f}C,std={f[4]:.2f}K'
        options[CANDIDATES[k]]=text
    return compile_request(state,{'action':{'type':'choice','instructions':'Choose low tracking cost and effort; limit 120 C.',
                                          'criteria':options}})


def tokenize(model,records):
    from jev.api import candidate_prompts
    prompts=[model.tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,
             add_generation_prompt=True,enable_thinking=False) for p in candidate_prompts(records[0])]
    encoded=model.tokenizer(prompts,padding=True,truncation=False,return_tensors='pt')
    require(int(encoded['attention_mask'].sum(-1).max())<=384,'Native serializer exceeds 384 tokens; no truncation permitted')
    return encoded


def score_tokens(model,ids,mask):
    device=model.device_name; ids=ids.to(device); mask=mask.to(device)
    output=model.backbone(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[torch.arange(len(ids),device=device),mask.sum(-1)-1]
    return model.head(hidden.float()).squeeze(-1)


def policy_loss(scores,target,cost,valid,feasible,gap):
    require(bool(valid.any()),'No valid candidates')
    require(torch.isfinite(scores).all().item(),'Nonfinite logits')
    logp=scores.float().masked_fill(~valid,-torch.inf).log_softmax(-1)
    positive=target>0
    require(bool(positive.any()) and not bool((positive&~valid).any()),'Invalid soft targets')
    ce=-(target[positive]*logp[positive]).sum()
    i,j=torch.triu_indices(len(scores),len(scores),1,device=scores.device)
    take=valid[i]&valid[j]&feasible[i]&feasible[j]&((cost[i]-cost[j]).abs()>gap)
    rank=scores.new_zeros(())
    if take.any():
        difference=(scores[i[take]]-scores[j[take]])*torch.sign(cost[j[take]]-cost[i[take]])
        rank=torch.nn.functional.softplus(-difference).mean()
    return ce+.2*rank


class NativePolicy:
    """Deployment-shaped decision API: observation and predicted features only, never expert labels."""
    def __init__(self,model,physics_text=False):
        self.model=model.eval(); self.physics_text=physics_text

    def decide(self,observation,commands,valid_mask,predicted_features):
        from .data import bilinear_observer
        started=time.perf_counter_ns(); valid=np.asarray(valid_mask,dtype=bool)
        require(np.shape(commands)==(9,20,4) and valid.shape==(9,) and np.shape(predicted_features)==(9,16),'Invalid decision shape')
        require(np.isfinite(predicted_features).all(),'Nonfinite predicted feature')
        allowed=valid&(predicted_features[:,2]<=120.)
        reason=observation.reason if not observation.usable else ('no_predicted_feasible_candidate' if not allowed.any() else None)
        if reason:
            return {'candidate_id':None,'raw_scores':None,'probabilities':[0.]*9,'valid_mask':valid.tolist(),
                    'fallback_reason':reason,'command':[0.]*4,'safety_mode':'simulation_shutdown_slew_exception',
                    'timing_ms':(time.perf_counter_ns()-started)/1e6}
        records=serialize(observation,bilinear_observer(observation),commands,predicted_features if self.physics_text else None)
        with torch.no_grad(): scores=self.model(records)[0].float().cpu()
        require(torch.isfinite(scores).all().item(),'Nonfinite decision scores')
        probabilities=scores.masked_fill(~torch.tensor(allowed),-torch.inf).softmax(-1)
        choice=int(probabilities.argmax())
        return {'candidate_id':CANDIDATES[choice],'raw_scores':scores.tolist(),'probabilities':probabilities.tolist(),
                'valid_mask':valid.tolist(),'predicted_safe_mask':allowed.tolist(),'fallback_reason':None,
                'command':commands[choice,0].tolist(),'calibrated':False,
                'timing_ms':(time.perf_counter_ns()-started)/1e6}


def load_original(root):
    from jev.model import DecisionModel
    from peft import PeftModel
    checkpoint=root/'models/Open-Jev-2B/package/checkpoint'; cfg=read_json(checkpoint/'model.json')
    model=DecisionModel(cfg['model_id'],cfg['revision'],device='cuda:0',lora_rank=0,max_length=384)
    model.backbone=PeftModel.from_pretrained(model.backbone,checkpoint/'adapter',is_trainable=True)
    model.lora_rank=cfg['lora_rank']
    model.head.load_state_dict(torch.load(checkpoint/'head.pt',map_location='cuda:0',weights_only=True))
    model.backbone.config.use_cache=False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.backbone.enable_input_require_grads()
    sys.path.insert(0,str(root/'stages/L0'))
    from model_checks import verify_cuda_model
    verify_cuda_model(model)
    return model


def build_tokens(model,run,output,physics):
    base=run/'01_generate/data'; labels=run/'03_observer_labels'; feature_dir=run/'08_physics_cache'
    entries=read_json(base/'manifests/split.json')['episodes']; cache=EpisodeCache(base,entries)
    lengths=[]; reference=None
    for split in ['train','validation']:
        index=np.load(labels/(split+'_index.npy'),mmap_mode='r')
        fields=np.load(labels/(split+'_field.npy'),mmap_mode='r'); commands=np.load(labels/(split+'_commands.npy'),mmap_mode='r')
        features=np.load(feature_dir/(split+'_features.npy'),mmap_mode='r')
        token_ids=np.lib.format.open_memmap(output/(split+'_tokens.npy'),mode='w+',dtype=np.int32,shape=(len(index),9,384))
        token_ids[:]=model.tokenizer.pad_token_id
        masks=np.lib.format.open_memmap(output/(split+'_masks.npy'),mode='w+',dtype=bool,shape=(len(index),9,384)); masks[:]=False
        for i,(code,step,_) in enumerate(index):
            obs=cache.get(code).snapshot(step); records=serialize(obs,fields[i],commands[i],features[i] if physics else None)
            encoded=tokenize(model,records); width=encoded['input_ids'].shape[1]
            token_ids[i,:,:width]=encoded['input_ids'].numpy(); masks[i,:,:width]=encoded['attention_mask'].numpy().astype(bool)
            lengths.extend(encoded['attention_mask'].sum(-1).tolist())
            if split=='validation' and i==0: reference=records
        for a in [token_ids,masks]: a.flush(); a._mmap.close()
        for a in [index,fields,commands,features]: a._mmap.close()
    cache.close()
    metadata={'serializer':'causal-summary-v1','physics':physics,'max_tokens':max(lengths),'mean_tokens':float(np.mean(lengths)),
              'model_id':model.model_id,'revision':model.revision,'template_hash':object_hash(model.tokenizer.chat_template),
              'feature_cache_sha256':digest(feature_dir/'cache_key.json'),'observer':'bilinear-probe-v1',
              'candidate_library':'axis9_poc_v1','timestamp_policy':'capture_receive_ready_le_decision_v1'}
    write_json(output/'token_cache.json',metadata); write_json(output/'reference_records.json',{'records':reference})
    return metadata,reference


class PolicyRows:
    def __init__(self,run,tokens,split):
        labels=run/'03_observer_labels'
        self.ids=np.load(tokens/(split+'_tokens.npy'),mmap_mode='r'); self.mask=np.load(tokens/(split+'_masks.npy'),mmap_mode='r')
        self.data={name:np.load(labels/(split+'_'+name+'.npy'),mmap_mode='r') for name in ['target','cost','valid','eligible']}
        self.features=np.load(run/'08_physics_cache'/(split+'_features.npy'),mmap_mode='r')

    def encoded(self,index,order=None):
        ids=np.array(self.ids[index],copy=True); mask=np.array(self.mask[index],copy=True)
        if order is not None: ids=ids[order]; mask=mask[order]
        width=int(mask.sum(-1).max())
        return torch.from_numpy(ids[:,:width].astype(np.int64)),torch.from_numpy(mask[:,:width].astype(np.int64))

    def close(self):
        for a in [self.ids,self.mask,self.features,*self.data.values()]: a._mmap.close()


def evaluate_policy(model,rows):
    """Expert feasibility is evaluation-only. Runtime filtering uses PINO predictions."""
    model.eval(); regrets=[]; predicted_fallback=expert_fallback=violations=0; agreement=0; selected=[]; unexpected_fallback=0
    with torch.no_grad():
        for i in range(len(rows.ids)):
            scores=score_tokens(model,*rows.encoded(i)).float().cpu().numpy()
            require(np.isfinite(scores).all(),'Nonfinite native evaluation')
            valid=np.array(rows.data['valid'][i]); eligible=np.array(rows.data['eligible'][i]); cost=rows.data['cost'][i]
            runtime_valid=valid&(rows.features[i,:,2]<=120.)  # same independent gate for both M0 variants
            if not eligible.any(): expert_fallback+=1
            if not runtime_valid.any():
                predicted_fallback+=1; unexpected_fallback+=int(eligible.any()); selected.append(-1); continue
            choice=int(np.where(runtime_valid,scores,-np.inf).argmax()); selected.append(choice)
            violations+=int(not eligible[choice])
            if eligible.any():
                regrets.append(float(cost[choice]-cost[eligible].min()))
                agreement+=int(cost[choice]<=cost[eligible].min()+1e-6 and eligible[choice])
    return {'rows':len(rows.ids),'regret_count':len(regrets),'mean_regret':float(np.mean(regrets)) if regrets else None,
            'expert_equivalent_choices':agreement,'expert_infeasible_choices':violations,
            'predicted_fallback_count':predicted_fallback,'expert_fallback_count':expert_fallback,
            'unexpected_fallback_count':unexpected_fallback,'expected_feasible_rows':len(rows.ids)-expert_fallback,
            'failure_or_fallback_fraction':(violations+predicted_fallback)/len(rows.ids),'selected_canonical_ids':selected,
            'selection_mask':'library validity AND predicted PINO peak<=120; no FV truth mask',
            'calibration':'temperature=1 uncalibrated after adaptation','test_accessed':False}


def policy_quality(baseline,selected,minimum_improvement):
    # Do not obtain an apparent regret improvement by dropping difficult/fallback decisions.
    complete=all(r['regret_count']==r['expected_feasible_rows'] for r in [baseline,selected])
    return bool(complete and selected['mean_regret'] is not None and baseline['mean_regret'] is not None and
                baseline['mean_regret']-selected['mean_regret']>minimum_improvement and
                selected['expert_infeasible_choices']==0 and selected['unexpected_fallback_count']==0)


def export_model(model,folder,metadata):
    import os,uuid
    temporary=folder.with_name('.'+folder.name+'.'+uuid.uuid4().hex)
    temporary.mkdir()
    model.backbone.save_pretrained(temporary/'adapter',safe_serialization=True,save_embedding_layers=False)
    torch.save({n:p.detach().cpu().clone() for n,p in model.head.state_dict().items()},temporary/'head.pt')
    cfg={'model_id':model.model_id,'revision':model.revision,'max_length':model.max_length,'lora_rank':model.lora_rank,
         'method':'thermal_l1_native_fv_supervised','training_metadata':metadata}
    write_json(temporary/'model.json',cfg)
    write_json(temporary/'temperature.json',{'temperature':1.,'calibrated':False,'reason':'manufacturing calibration deferred'})
    seal(temporary,{'base_weights_included':False,'initialization':'original Open-Jev release, never L0 diagnostic'})
    if folder.exists():
        import shutil
        shutil.rmtree(folder)  # Only this stage's explicitly owned export; restart checkpoint is separate.
    os.replace(temporary,folder)


def train_native(root,run,config,output,physics=False):
    from .common import setup_torch
    setup_torch(config); model=load_original(root)
    token_meta,reference=build_tokens(model,run,output,physics)
    train=PolicyRows(run,output,'train'); val=PolicyRows(run,output,'validation')
    model.eval()
    with torch.no_grad():
        original=model(reference)[0].float(); cached=score_tokens(model,*val.encoded(0)).float()
        torch.testing.assert_close(original,cached,atol=.03,rtol=.01)
    baseline=evaluate_policy(model,val)
    rng=np.random.default_rng(config['seed']+101); metrics=[]; stop=StopRequest()
    base=[(p,p._version) for n,p in model.backbone.named_parameters() if 'lora_' not in n]
    require(all(not p.requires_grad for p,_ in base),'Base not frozen')
    gap=read_json(run/'03_observer_labels/label_temperature.json')['equivalence_gap']
    eligible_rows=np.flatnonzero(np.array(train.data['target']).sum(1)>0)
    require(len(eligible_rows)>0,'No expert-feasible training rows; inspect fallback labels')
    best=float('inf'); selected_step=None; coverage=set()
    try:
        for phase,steps in [('warmup',config['budget']['native_warmup_steps']),('adapt',config['budget']['native_adapt_steps'])]:
            for name,parameter in model.backbone.named_parameters(): parameter.requires_grad_('lora_' in name and phase=='adapt')
            model.head.requires_grad_(True)
            cfg=config['poc']['training']['stages']['native_'+phase]
            groups=[{'params':list(model.head.parameters()),'lr':cfg['learning_rate_head']}]
            if phase=='adapt': groups.insert(0,{'params':[p for p in model.backbone.parameters() if p.requires_grad],
                                               'lr':cfg['learning_rate_lora']})
            optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
            parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
            write_json(output/(phase+'_trainable.json'),{'names':list(parameters),'count':sum(p.numel() for p in parameters.values())})
            before={n:p.detach().cpu().clone() for n,p in parameters.items()}
            restart=output/(phase+'_restart.pt'); start=0
            if restart.exists():
                start,extra=restore_training(restart,model,optimizer,rng,'cuda:0'); metrics=extra['metrics']
                best=extra['best']; coverage=set(extra['coverage']); selected_step=extra['selected_step']
            for step in range(start+1,steps+1):
                model.train(); optimizer.zero_grad(set_to_none=True); value=0.; started=time.monotonic()
                for _ in range(cfg['accumulation']):
                    index=int(rng.choice(eligible_rows)); coverage.add(index); order=rng.permutation(9)
                    scores=score_tokens(model,*train.encoded(index,order)).float()
                    target,cost,valid,feasible=[torch.tensor(np.array(train.data[name][index][order]),device='cuda:0')
                                             for name in ['target','cost','valid','eligible']]
                    loss=policy_loss(scores,target,cost,valid,feasible,gap)
                    require(torch.isfinite(loss).item(),'Nonfinite policy loss')
                    (loss/cfg['accumulation']).backward(); value+=float(loss.detach())/cfg['accumulation']
                grad={group:sum(float(p.grad.detach().float().abs().sum()) for n,p in parameters.items()
                                if p.grad is not None and (('lora_' in n)==(group=='lora'))) for group in ['head','lora']}
                require(grad['head']>0 and (phase=='warmup' or grad['lora']>0),'Missing head/LoRA gradient')
                torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True)
                warmup=max(1,int(np.ceil(steps*.05)))
                for i,group in enumerate(optimizer.param_groups):
                    lr=cfg['learning_rate_lora'] if phase=='adapt' and i==0 else cfg['learning_rate_head']
                    group['lr']=lr*min(1,step/warmup)
                optimizer.step()
                require(all(p.grad is None and p._version==version for p,version in base),'Frozen base mutated')
                row={'phase':phase,'step':step,'loss':value,'gradients':grad,'seconds':time.monotonic()-started}
                metrics.append(row)
                if step==steps or step%100==0 or config['profile']=='smoke':
                    evaluation=evaluate_policy(model,val); row['validation']=evaluation
                    metric=evaluation['mean_regret']
                    if metric is not None and metric<best:
                        best=metric; selected_step=[phase,step]
                        export_model(model,output/'best_model',{'features':'physics_text' if physics else 'none','selected_step':selected_step})
                if step%250==0 or step==steps or stop.signal:
                    save_training(restart,model,optimizer,step,rng,{'metrics':metrics,'best':best,'coverage':sorted(coverage),'selected_step':selected_step})
                print(f'M0 {"physics" if physics else "none"} {phase} {step}/{steps} loss={value:.6f}',flush=True)
                if stop.signal: raise InterruptedError('Saved native checkpoint at optimizer boundary; rerun --resume')
            changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in parameters.items()}
            if start<steps:
                require(any(v>0 for n,v in changes.items() if 'head' in n),'Head did not update')
                if phase=='adapt': require(any(v>0 for n,v in changes.items() if 'lora_' in n),'LoRA did not update')
            write_json(output/(phase+'_updates.json'),changes)
            export_model(model,output/(phase+'_model'),{'phase':phase,'features':'physics_text' if physics else 'none'})
        final=evaluate_policy(model,val)
        # Reload final trainable tensors into this instance from the selected export without another full backbone.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        require((output/'best_model').is_dir(),'Validation produced no selectable policy checkpoint')
        set_peft_model_state_dict(model.backbone,load_file(str(output/'best_model/adapter/adapter_model.safetensors'),device='cuda:0'))
        model.head.load_state_dict(torch.load(output/'best_model/head.pt',map_location='cuda:0',weights_only=True))
        selected=evaluate_policy(model,val); model.eval()
        with torch.no_grad(): scores=model(reference)[0].float().cpu(); valid=torch.tensor(np.array(val.data['valid'][0]))
        probabilities=scores.masked_fill(~valid,-torch.inf).softmax(-1)
        write_json(output/'reload_reference.json',{'records':reference,'scores':scores.tolist(),'valid':valid.tolist(),
                   'probabilities':probabilities.tolist(),'choice':int(probabilities.argmax())})
        quality=policy_quality(baseline,selected,config['acceptance']['policy']['regret_improvement_min'])
        write_json(output/'training.json',{'metrics':metrics,'baseline':baseline,'last':final,'selected':selected})
        return {'variant':'M0-physics-text' if physics else 'M0-no-physics','token_cache':token_meta,
                'baseline_validation':baseline,'selected_validation':selected,'selected_step':selected_step,
                'quality_pass':quality,'unique_train_rows':len(coverage),'train_rows':len(train.ids),
                'train_fallback_rows':len(train.ids)-len(eligible_rows),'optimizer_steps':sum(config['budget'][k] for k in ['native_warmup_steps','native_adapt_steps']),
                'frozen_base_unchanged':True,'original_forward_token_cache_parity':True,
                'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'checkpoint_sha256':digest(output/'best_model/manifest.json'),'calibrated':False,'test_accessed':False}
    finally:
        train.close(); val.close(); stop.close()


def reload_native(root,run,config,training_stage):
    from .common import setup_torch
    setup_torch(config)
    from jev.model import DecisionModel
    folder=run/training_stage; verify(folder/'best_model')
    expected=read_json(folder/'reload_reference.json'); model=DecisionModel.load(folder/'best_model',device='cuda:0')
    with torch.no_grad(): scores=model(expected['records'])[0].float().cpu()
    valid=torch.tensor(expected['valid']); probability=scores.masked_fill(~valid,-torch.inf).softmax(-1)
    tolerance=config['acceptance']['reload']
    torch.testing.assert_close(scores,torch.tensor(expected['scores']),atol=tolerance['score_atol'],rtol=tolerance['score_rtol'])
    torch.testing.assert_close(probability,torch.tensor(expected['probabilities']),atol=tolerance['probability_atol'],rtol=tolerance['probability_rtol'])
    require(int(probability.argmax())==expected['choice'],'Reload changed canonical selection')
    return {'score_max_error':float((scores-torch.tensor(expected['scores'])).abs().max()),
            'probability_max_error':float((probability-torch.tensor(expected['probabilities'])).abs().max()),
            'canonical_choice':CANDIDATES[expected['choice']],'base_weights_saved':False,'fresh_process':True}
