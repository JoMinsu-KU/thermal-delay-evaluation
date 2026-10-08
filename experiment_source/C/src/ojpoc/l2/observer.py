from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..common import read_json, write_json, require, setup_torch, save_training, restore_training, StopRequest
from ..data import RawEpisode
from ..physics import resize, PROBES
from .data import baseline_field
from .contracts import entries, anchors


def inputs(obs):
    frames=[resize(np.where(obs.thermal_valid[i,0],(obs.thermal[i,0]-25)/100,0),32) for i in range(2)]
    masks=[resize(obs.thermal_valid[i,0].astype(float),32) for i in range(2)]
    raster=np.zeros((32,32),np.float32);mask=np.zeros_like(raster)
    for j,(x,y) in enumerate(PROBES):
        if obs.sensor_valid[-1,j]:
            a,b=min(31,int(y/.5*32)),min(31,int(x/.5*32))
            raster[a,b]=(obs.sensor[-1,j]-25)/100;mask[a,b]=1
    return np.stack([*frames,*masks,raster,mask,np.full_like(raster,obs.thermal_age[-1]),
                     np.full_like(raster,float(obs.sensor_age[-1,:8].min()))]).astype(np.float32)


class SmallObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.e2=nn.Sequential(nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU())
        self.e3=nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU())
        self.d2=nn.Sequential(nn.Conv2d(96,32,3,padding=1),nn.GELU())
        self.d1=nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.d1[-1].weight);nn.init.zeros_(self.d1[-1].bias)

    def forward(self,x,baseline):
        a=self.e1(x);b=self.e2(a);c=self.e3(b)
        up=lambda v:torch.nn.functional.interpolate(v,scale_factor=2,mode='bilinear',align_corners=False)
        y=self.d1(torch.cat([up(self.d2(torch.cat([up(c),b],1))),a],1))
        return baseline+y[:,0]


def dataset(run,cfg,split):
    base=run/'02_generate/data';xx=[];bb=[];yy=[]
    for entry in entries(run,[split]):
        raw=RawEpisode(base/'raw'/entry['episode_id'])
        truth=np.load(base/'truth'/entry['episode_id']/'T32.npy',mmap_mode='r')
        for step in anchors(cfg):
            obs=raw.snapshot(step)
            if obs.usable:
                xx.append(inputs(obs));bb.append((baseline_field(obs)-25)/100);yy.append((truth[step,0]-25)/100)
        raw.close();truth._mmap.close()
    require(bool(xx),'No usable observer training/validation examples')
    return tuple(torch.from_numpy(np.array(v,np.float32)) for v in [xx,bb,yy])


def train(run,cfg,output):
    setup_torch(cfg);model=SmallObserver().cuda();train=dataset(run,cfg,'train');val=dataset(run,cfg,'validation')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);rng=np.random.default_rng(cfg['seed'])
    stop=StopRequest();restart=output/'restart.pt';start=0;history=[];best=float('inf')
    if restart.exists():start,extra=restore_training(restart,model,optimizer,rng,'cuda');history=extra['history'];best=extra['best']
    def evaluate():
        model.eval();errors=[];peaks=[]
        with torch.no_grad():
            for i in range(0,len(val[0]),32):
                x,b,y=[v[i:i+32].cuda() for v in val];pred=model(x,b)
                errors.extend(((pred-y)*100).square().mean((1,2)).cpu().tolist())
                peaks.extend(((pred.amax((1,2))-y.amax((1,2))).abs()*100).cpu().tolist())
        return {'rmse_k':float(np.sqrt(np.mean(errors))),'peak_error_k':max(peaks)}
    baseline={'rmse_k':float(((val[1]-val[2])*100).square().mean().sqrt()),
              'peak_error_k':float(((val[1].amax((1,2))-val[2].amax((1,2))).abs()*100).max())}
    try:
        for step in range(start+1,cfg['budget']['observer_steps']+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            ids=rng.integers(0,len(train[0]),size=min(32,len(train[0])))
            x,b,y=[v[ids].cuda() for v in train];pred=model(x,b)
            loss=torch.nn.functional.smooth_l1_loss(pred,y,beta=.01)
            probe_mask=x[:,5];loss+=.1*((pred-x[:,4]).abs()*probe_mask).sum()/probe_mask.sum().clamp_min(1)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row={'step':step,'loss':float(loss.detach())};history.append(row)
            if step%cfg['budget']['validation_every']==0 or step==cfg['budget']['observer_steps']:
                row['validation']=evaluate()
                if row['validation']['rmse_k']<best:
                    best=row['validation']['rmse_k'];torch.save(model.state_dict(),output/'best.pt')
            if step%100==0 or step==cfg['budget']['observer_steps'] or stop.signal:
                save_training(restart,model,optimizer,step,rng,{'best':best,'history':history})
            if stop.signal:raise InterruptedError('Observer checkpoint committed')
        model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True));candidate=evaluate()
        selected='small' if candidate['rmse_k']<=baseline['rmse_k'] and candidate['peak_error_k']<=baseline['peak_error_k'] else 'bilinear'
        result={'selected':selected,'baseline':baseline,'small':candidate,'history':history,
                'parameters':sum(p.numel() for p in model.parameters()),'selection_split':'validation','test_accessed':False}
        write_json(output/'selection.json',result);return result
    finally:stop.close()


class Observer:
    def __init__(self,run,mode='both',device='cuda'):
        self.mode=mode;self.device=device;self.model=None
        selection=read_json(run/'04_observer/selection.json')
        if mode=='both' and selection['selected']=='small':
            self.model=SmallObserver().to(device).eval()
            self.model.load_state_dict(torch.load(run/'04_observer/best.pt',map_location=device,weights_only=True))
            self.model.requires_grad_(False)

    def __call__(self,obs):
        baseline=baseline_field(obs,self.mode)
        if self.model is None:return baseline
        with torch.inference_mode():
            x=torch.as_tensor(inputs(obs)[None],device=self.device)
            b=torch.as_tensor((baseline[None]-25)/100,device=self.device)
            return (self.model(x,b)[0]*100+25).cpu().numpy()
