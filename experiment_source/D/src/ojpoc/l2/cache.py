"""Mmap numerical inputs; independent FV labels; frozen predicted physics features."""
from pathlib import Path
import numpy as np
import torch
from ..common import read_json, write_json, require, save_array, seal, verify, digest, setup_torch
from ..data import RawEpisode, EpisodeCache
from ..physics import candidates, branch, costs, soft_targets
from ..operator import ThermalOperator, physics_features
from ..evaluation import tensor_inputs
from .data import view, packed
from .observer import Observer
from .contracts import entries, anchors


def operator(run,variant='pino_balance',device='cuda'):
    folder={'pino_balance':'07_pino_balance','pino_soft':'06_pino_soft','fno_data':'05_fno_data'}[variant]
    model=ThermalOperator(variant).to(device).eval()
    model.load_state_dict(torch.load(run/'01_frozen_inputs'/folder/'best.pt',map_location=device,weights_only=True))
    model.requires_grad_(False);return model


def prediction(model,field,obs,commands,device='cuda'):
    values=tensor_inputs(field,obs,commands,device)
    with torch.no_grad():
        future=model(*values)
        previous=torch.tensor(np.repeat(obs.previous[None],9,0),device=device,dtype=torch.float32)
        goal=torch.full((9,),obs.goal,device=device);age=torch.full_like(goal,float(obs.thermal_age[-1]))
        features=physics_features(future,values[0],values[2],values[3],values[1],previous,goal,age)
    return future,features


SHAPES={'sensor':(3,64,16),'thermal':(6,16,16),'plc':(12,),'passport':(14,),
        'field':(32,32),'commands':(9,20,4),'features':(9,16),'cost':(9,),'target':(9,),
        'valid':(9,),'eligible':(9,),'usable':(),'fine_peak':(),'predicted_peak':(),
        'index':(2,),'goal':(),'previous':(4,),'actuator':(4,),'theta':(9,),'thermal_age':()}


def make_split(run,cfg,output,mode,split,op=None):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'COMPLETE').exists():verify(output);return read_json(output/'metadata.json')
    selected=entries(run,[split]);n=len(selected)*len(anchors(cfg)); require(n>0,'Empty split')
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=(bool if k in ['valid','eligible','usable'] else np.int64 if k=='index' else np.float32),shape=(n,*shape)) for k,shape in SHAPES.items()}
    for a in arrays.values():a[:]=0
    norm=read_json(run/'03_index/normalization.json');observer=Observer(run,mode)
    model=operator(run) if op is None else op; gaps=[];i=0; coverage=0
    base=run/'02_generate/data'
    for entry in selected:
        raw=RawEpisode(base/'raw'/entry['episode_id'])
        for step in anchors(cfg):
            obs=view(raw.snapshot(step),mode); commands,valid=candidates(obs.previous)
            for k,v in packed(obs,norm,mode).items():arrays[k][i]=v
            for k,v in {'commands':commands,'valid':valid,'usable':obs.usable,'index':[entry['code'],step],
                        'goal':obs.goal,'previous':obs.previous,'actuator':obs.actuator,'theta':obs.passport,'thermal_age':obs.thermal_age[-1]}.items():arrays[k][i]=v
            if obs.usable:
                field=observer(obs); future=branch(field,obs.actuator,commands,obs.passport)
                cost=costs(future,commands,obs.previous,obs.goal); eligible=valid&(future.max((1,2,3))<=120)
                pred,features=prediction(model,field,obs,commands)
                for k,v in {'field':field,'features':features.cpu().numpy(),'cost':cost,'eligible':eligible,
                            'fine_peak':future.max(),'predicted_peak':float(pred.max())}.items():arrays[k][i]=v
                coverage+=int(eligible.any())
                if split=='train':
                    active=cost[eligible]
                    gaps.extend(abs(active[a]-active[b]) for a in range(len(active)) for b in range(a+1,len(active)))
            i+=1
        raw.close();print(f'L2 cache {mode}/{split} {i}/{n}',flush=True)
    if split=='train':
        significant=[x for x in gaps if x>1e-6];temperature=max(1e-6,float(np.median(significant)) if significant else 1e-6)
    else:temperature=read_json(run/'05_cache'/mode/'train/metadata.json')['label_temperature']
    for i in range(n):arrays['target'][i]=soft_targets(arrays['cost'][i],arrays['eligible'][i],temperature) if arrays['eligible'][i].any() else np.zeros(9)
    for a in arrays.values():a.flush();a._mmap.close()
    info={'mode':mode,'split':split,'rows':n,'expert_feasible_rows':coverage,'label_temperature':temperature,
          'labels':'FV64 from modality-consistent observed field; no PINO self-labeling',
          'normalizer_hash':digest(run/'03_index/normalization.json'),
          'observer_hash':digest(run/'04_observer/selection.json'),
          'operator_hash':digest(run/'01_frozen_inputs/07_pino_balance/best.pt'),
          'timestamp_policy':'capture_receive_ready_le_decision','candidate_library':'axis9_poc_v1',
          'physics_feature_calibration_indices_13_14':'fixed_zero','uses_future_truth_as_input':False}
    write_json(output/'metadata.json',info);seal(output,info);return info


def prepare(run,cfg,output):
    setup_torch(cfg);op=operator(run);result={}
    for mode in ['both','sensor','thermal']:
        for split in ['train','validation']:result[mode+'/'+split]=make_split(run,cfg,output/mode/split,mode,split,op)
    return result


class Rows:
    def __init__(self,folder):
        self.folder=Path(folder);self.meta=read_json(self.folder/'metadata.json')
        self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in SHAPES}

    def __len__(self):return len(self.arrays['index'])

    def batch(self,indices,device='cuda'):
        return {k:torch.as_tensor(np.array(v[indices],copy=True),device=device) for k,v in self.arrays.items()}

    def close(self):
        for a in self.arrays.values():a._mmap.close()


def obs_for_row(run,cache,index):
    code,step=map(int,index);return cache.get(code).snapshot(step)
