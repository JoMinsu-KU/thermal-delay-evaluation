"""Shared state Qwen once per request; candidate MLP is permutation equivariant."""
import gc
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..common import read_json, write_json, require, seal, verify
from ..policy import load_original, serialize, tokenize, score_tokens
from .contracts import has_physics, modality, model_kind
from .data import packed, baseline_field


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__();self.pad=4*dilation
        self.conv=nn.Conv1d(128,128,5,dilation=dilation)
        self.norm=nn.GroupNorm(8,128)

    def forward(self,x):
        return x+torch.nn.functional.gelu(self.norm(self.conv(torch.nn.functional.pad(x,(self.pad,0)))))


class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor=nn.Sequential(nn.Conv1d(48,128,1),*[CausalBlock(d) for d in [1,2,4]],nn.AdaptiveAvgPool1d(8))
        self.thermal=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),
                                   nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((2,4)),nn.Conv2d(64,128,1))
        self.plc=nn.Sequential(nn.Linear(12,128),nn.GELU(),nn.Linear(128,256))
        self.passport=nn.Sequential(nn.Linear(14,128),nn.GELU(),nn.Linear(128,512))
        self.types=nn.Parameter(torch.randn(4,128)*.01)

    def forward(self,batch,mode='both'):
        b=batch['sensor'].shape[0]
        sensor=self.sensor(batch['sensor'].permute(0,1,3,2).reshape(b,48,64)).transpose(1,2)+self.types[0]
        thermal=self.thermal(batch['thermal']).flatten(2).transpose(1,2)+self.types[1]
        plc=self.plc(batch['plc']).reshape(b,2,128)+self.types[2]
        passport=self.passport(batch['passport']).reshape(b,4,128)+self.types[3]
        pieces=[]
        if mode!='thermal':pieces.append(sensor)
        if mode!='sensor':pieces.append(thermal)
        return torch.cat([*pieces,plc,passport],1)


class SharedPolicy(nn.Module):
    def __init__(self,root,method,device='cuda:0',native=None,feature_version='legacy',relative_scale=None):
        super().__init__();self.method=method;self.mode=modality(method);self.physics=has_physics(method)
        require(feature_version in ['legacy','energy_scaled_v2','candidate_relative_v3'],'Unknown physics feature normalization')
        self.feature_version=feature_version
        self.small=method=='small_physics';self.device_name=device
        self.encoder=ObservationEncoder()
        if not self.small:
            original=native if native is not None else load_original(root)
            self.backbone=original.backbone;self.tokenizer=original.tokenizer
            self.model_id=original.model_id;self.revision=original.revision
            self.projection=nn.Linear(128,self.backbone.config.hidden_size)
            self.summary=nn.Parameter(torch.randn(1,1,self.backbone.config.hidden_size)*.01)
            ids=self.tokenizer.encode('Choose the thermal control action.',add_special_tokens=False)
            require(len(ids)<=16,'Task text contract')
            self.register_buffer('task_ids',torch.tensor(ids,dtype=torch.long),persistent=False)
            self.context=nn.Sequential(nn.Linear(self.backbone.config.hidden_size,256),nn.GELU())
        else:
            self.summary=nn.Parameter(torch.randn(1,1,128)*.01)
            self.sequence=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True,norm_first=True),3,enable_nested_tensor=False)
            self.context=nn.Sequential(nn.Linear(128,256),nn.GELU())
        self.action=nn.Sequential(nn.Linear(80,64),nn.GELU(),nn.Linear(64,64))
        self.physics_head=nn.Sequential(nn.Linear(32 if feature_version=='candidate_relative_v3' else 16,64),nn.GELU(),nn.Linear(64,64))
        self.scorer=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,1))
        self.register_buffer('feature_scale',torch.tensor(feature_scales(feature_version)),persistent=False)
        self.register_buffer('relative_scale',torch.tensor(relative_scale if relative_scale is not None else [1.]*16,dtype=torch.float32),persistent=False)
        require(self.relative_scale.shape==(16,) and bool(torch.isfinite(self.relative_scale).all()) and bool((self.relative_scale>0).all()),'Invalid relative feature scale')
        self.to(device)

    def encode_state(self,batch):
        observation=self.encoder(batch,self.mode); b=len(observation)
        if self.small:
            sequence=torch.cat([observation,self.summary.expand(b,-1,-1)],1)
            hidden=self.sequence(sequence)[:,-1]
            return self.context(hidden),sequence.shape[1]
        embedded=self.projection(observation)
        task=self.backbone.get_input_embeddings()(self.task_ids).float().expand(b,-1,-1)
        sequence=torch.cat([task,embedded,self.summary.expand(b,-1,-1)],1)
        length=sequence.shape[1];require(length<=39,'Shared token contract exceeded')
        mask=torch.zeros((b,64),device=sequence.device,dtype=torch.long);mask[:,:length]=1
        sequence=torch.nn.functional.pad(sequence,(0,0,0,64-length)).to(torch.bfloat16)
        positions=torch.arange(64,device=sequence.device).expand(b,-1)
        output=self.backbone(inputs_embeds=sequence,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
        hidden=output.last_hidden_state[:,length-1].float()
        return self.context(hidden),length

    def forward(self,batch,commands,features):
        context,tokens=self.encode_state(batch)
        action=self.action(commands.flatten(-2))
        features=policy_features(features,self.feature_scale,self.relative_scale,self.feature_version)
        if not self.physics:features=torch.zeros_like(features)
        physics=self.physics_head(features)
        scores=self.scorer(torch.cat([context[:,None].expand(-1,commands.shape[1],-1),action,physics],-1)).squeeze(-1)
        return scores,tokens


def feature_scales(version='legacy'):
    require(version in ['legacy','energy_scaled_v2','candidate_relative_v3'],'Unknown feature scale version')
    # Energy over H=2s on the 0.25m2 plate at 40000W/m2: 20000J.
    return [1,1,100,100,100,100,1,1 if version=='legacy' else 20000,1,1,40000,40000,40000,1,1,1]


def normalize_features(x,scale=None):
    scale=x.new_tensor(feature_scales()) if scale is None else scale
    y=x/scale
    y=y.clone();y[...,13:15]=0
    return y


def policy_features(x,absolute_scale,relative_scale,version):
    absolute=normalize_features(x,absolute_scale)
    if version!='candidate_relative_v3':return absolute
    # Accumulate the mean in FP64: FP32 order-dependent cancellation can exceed
    # the small inter-candidate temperature/cost differences being exposed.
    precise=x.double()
    relative=((precise-precise.mean(-2,keepdim=True))/relative_scale.double()).to(x.dtype)
    relative=relative.clone();relative[...,13:15]=0
    return torch.cat([absolute,relative],-1)


def fit_relative_scale(folder):
    """Only usable training candidates determine the fixed contrast RMS."""
    folder=Path(folder);require(folder.name=='train','Relative normalization must fit train only')
    features=np.load(folder/'features.npy',mmap_mode='r')
    valid=np.load(folder/'valid.npy',mmap_mode='r');usable=np.load(folder/'usable.npy',mmap_mode='r')
    delta=np.asarray(features,dtype=np.float64)-features.mean(1,keepdims=True,dtype=np.float64)
    mask=valid&usable[:,None];require(mask.any(),'No training candidates for feature normalization')
    scale=np.maximum(np.sqrt(np.square(delta[mask]).mean(0)),1e-6);scale[13:15]=1
    return scale.astype(np.float32).tolist()


def tensor_batch(arrays,device):
    return {k:torch.as_tensor(np.array(v,copy=True),device=device,dtype=torch.float32) for k,v in arrays.items()}


def export(model,folder,metadata):
    folder=Path(folder);temporary=folder.with_name('.'+folder.name+'.export')
    temporary.mkdir(parents=True,exist_ok=True)
    # All new modules and all LoRA parameters, including frozen LoRA during warmup.
    selected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('backbone.') or 'lora_' in n}
    torch.save(selected,temporary/'parameters.pt')
    write_json(temporary/'model.json',{**metadata,'feature_version':getattr(model,'feature_version','legacy'),
                                     'relative_scale':model.relative_scale.detach().cpu().tolist() if hasattr(model,'relative_scale') else None})
    seal(temporary,{'base_weights_included':False,'parameter_count':sum(v.numel() for v in selected.values())})
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    os.replace(temporary,folder)


def restore(model,folder,device='cuda:0'):
    verify(folder)
    require(read_json(Path(folder)/'model.json').get('feature_version','legacy')==getattr(model,'feature_version','legacy'),
            'Checkpoint feature normalization differs from requested policy')
    if getattr(model,'feature_version','legacy')=='candidate_relative_v3':
        expected=torch.tensor(read_json(Path(folder)/'model.json')['relative_scale'])
        require(torch.equal(model.relative_scale.detach().cpu(),expected),'Checkpoint relative normalization changed')
    state=torch.load(folder/'parameters.pt',map_location=device,weights_only=True)
    named=dict(model.named_parameters())
    expected={n for n in named if not n.startswith('backbone.') or 'lora_' in n}
    require(set(state)==expected,'Checkpoint architecture changed')
    with torch.no_grad():
        for name,value in state.items():named[name].copy_(value)
    return model


class NativePolicy(nn.Module):
    def __init__(self,root,method):
        super().__init__();self.method=method
        model=load_original(root);self.backbone=model.backbone;self.head=model.head
        self.tokenizer=model.tokenizer;self.model_id=model.model_id;self.revision=model.revision;self.device_name='cuda:0'

    def forward(self,ids,mask):
        return score_tokens(self,ids,mask)


def load_policy(root,run,method,training=False,inference_backend=None):
    kind=model_kind(method)
    settings=read_json(run/'resolved_config.json')['l2']
    version=settings.get('physics_feature_version','legacy')
    relative=None
    if not kind.startswith('native') and version=='candidate_relative_v3':
        relative=fit_relative_scale(run/'05_cache'/modality(method)/'train') if training else read_json(run/'models'/kind/'best/model.json')['relative_scale']
    model=NativePolicy(root,kind) if kind.startswith('native') else SharedPolicy(root,kind,feature_version=version,relative_scale=relative)
    if not training and kind!='native_original':restore(model,run/'models'/kind/'best')
    if not training:
        model.eval();model.requires_grad_(False)
        if hasattr(model,'backbone'):model.backbone.gradient_checkpointing_disable()
        backend=settings.get('inference_backend','eager') if inference_backend is None else inference_backend
        require(backend in ['eager','cuda_graph'],'Unknown inference backend')
        if not kind.startswith('native') and backend=='cuda_graph':
            from .graph import GraphPolicy
            model=GraphPolicy(model)
    return model


def close_model(model):
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def parity(root,run,cfg,output):
    from ..common import setup_torch
    setup_torch(cfg)
    native=load_original(root);native.eval();native.backbone.gradient_checkpointing_disable()
    encoded=native.tokenizer(['Control heat.','Choose an action for this thermal process.'],padding=True,return_tensors='pt')
    ids=encoded['input_ids'].cuda();mask=encoded['attention_mask'].cuda()
    positions=torch.arange(ids.shape[1],device=ids.device).expand(len(ids),-1)
    with torch.no_grad():
        a=native.backbone(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state
        b=native.backbone(inputs_embeds=native.backbone.get_input_embeddings()(ids),attention_mask=mask,
                          position_ids=positions,use_cache=False).last_hidden_state
        torch.testing.assert_close(a,b,atol=.03,rtol=.01)
    error=float((a-b).abs().max())
    version=cfg['l2'].get('physics_feature_version','legacy')
    relative=fit_relative_scale(run/'05_cache/both/train') if version=='candidate_relative_v3' else None
    model=SharedPolicy(root,'shared_physics',native=native,feature_version=version,relative_scale=relative)
    model.backbone.requires_grad_(False)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    batch={'sensor':torch.randn(1,3,64,16,device='cuda'),'thermal':torch.randn(1,6,16,16,device='cuda'),
           'plc':torch.randn(1,12,device='cuda'),'passport':torch.randn(1,14,device='cuda')}
    commands=torch.rand(1,9,20,4,device='cuda');features=torch.rand(1,9,16,device='cuda')
    model.train();scores,tokens=model(batch,commands,features);scores.square().mean().backward()
    grads={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
           for name,module in [('encoder',model.encoder),('projection',model.projection),('scorer',model.scorer)]}
    require(all(v>0 and np.isfinite(v) for v in grads.values()),'Frozen Qwen cut off encoder gradients')
    require(all(p.grad is None for p in model.backbone.parameters()),'Frozen base received gradients')
    model.eval()
    with torch.no_grad():
        original,_=model(batch,commands,features);order=torch.tensor([8,2,4,1,3,7,0,6,5],device='cuda')
        permuted,_=model(batch,commands[:,order],features[:,order])
        torch.testing.assert_close(permuted,original[:,order],atol=1e-4,rtol=1e-4)
    return {'token_embedding_max_error':error,'encoder_gradients':grads,'permutation_equivariance':True,
            'valid_tokens':tokens,'padded_tokens':64,'frozen_backbone_graph_preserved':True}
