import gc
import time
import numpy as np
import torch
from ..common import read_json, require
from ..physics import candidates, costs, branch, PIController
from ..policy import serialize, tokenize
from .contracts import modality, model_kind, has_physics, ABLATIONS
from .models import load_policy, tensor_batch
from .cache import operator, prediction
from .observer import Observer
from .data import view, packed


class Engine:
    """No teacher cost/true state in the inference interface."""
    def __init__(self,root,run,method,calibrated=True,policy_override=None):
        self.root=root;self.run=run;self.method=method;self.mode=modality(method)
        self.residual_protocol=read_json(run/'resolved_config.json')['l2'].get('residual_learning',{}).get('enabled',False)
        self.normalization=read_json(run/'03_index/normalization.json')
        self.observer=Observer(run,self.mode);self.operator=operator(run)
        self.alternative=None
        if method in ['fno_mpc','physics_fno','physics_soft']:
            self.alternative=operator(run,'pino_soft' if method=='physics_soft' else 'fno_data')
        self.policy=policy_override
        if self.policy is None and (method.startswith(('shared','small','native')) or method in ABLATIONS):
            self.policy=load_policy(root,run,method)
        if self.policy is not None:self.policy.eval()
        self.temperature=1.;self.margin=0.
        if calibrated:
            cal=read_json(run/'16_calibration/calibration.json')
            self.temperature=cal['temperatures'].get(model_kind(method),{}).get('temperature',1.)
            self.margin=cal['margins'][self.mode]['margin_k']
        self.pi=PIController()

    def reset(self):self.pi=PIController()

    def close(self):
        self.policy=None;self.operator=None;self.alternative=None;self.observer=None;gc.collect();torch.cuda.empty_cache()

    def decide(self,make_observation):
        torch.cuda.synchronize();start=time.perf_counter_ns()
        obs=view(make_observation(),self.mode);commands,valid=candidates(obs.previous)
        out={'method':self.method,'choice':-1,'command':[0.]*4,'fallback_reason':obs.reason,'tokens':0,
             'scores':None,'probabilities':None,'thermal_age_s':float(obs.thermal_age[-1]),'filter_intervention':False,
             'valid_mask':valid.tolist(),'predicted_safe_mask':[False]*9,'calibrated_temperature':self.temperature,'margin_k':self.margin}
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        if obs.usable:
            field=self.observer(obs);future,features=prediction(self.operator,field,obs,commands)
            f=features.cpu().numpy();require(np.isfinite(f).all(),'Nonfinite physics')
            require(np.all(f[:,13:15]==0),'Calibration leaked into learned features')
            allowed=valid&(f[:,2]+self.margin<=120)
            out['predicted_safe_mask']=allowed.tolist();out['predicted_peak_c']=f[:,2].tolist()
            if not allowed.any():out['fallback_reason']='no_predicted_feasible_candidate';out['filter_intervention']=True
            else:
                with torch.inference_mode():
                    if self.method=='pi':
                        zones=np.array([field[:16,:16].mean(),field[:16,16:].mean(),field[16:,:16].mean(),field[16:,16:].mean()])
                        requested=self.pi.command(zones,obs.goal,obs.previous);scores=-((commands[:,0]-requested)**2).mean(1)
                    elif self.method in ['fv_mpc','pino_mpc','fno_mpc']:
                        if self.method=='fv_mpc':pred=branch(field,obs.actuator,commands,obs.passport)
                        elif self.method=='fno_mpc':pred=prediction(self.alternative,field,obs,commands)[0].cpu().numpy()
                        else:pred=future.cpu().numpy()
                        scores=-costs(pred,commands,obs.previous,obs.goal)
                        if self.method=='pino_mpc' and self.residual_protocol:
                            from .residual import pino_cost
                            scores=-pino_cost(torch.as_tensor(commands[None],device='cuda',dtype=torch.float32),features[None])[0].cpu().numpy()
                    elif self.method.startswith('native'):
                        encoded=tokenize(self.policy,serialize(obs,field,commands,f if self.method=='native_physics' else None))
                        out['tokens']=int(encoded['attention_mask'].sum(-1).max())
                        scores=self.policy(encoded['input_ids'],encoded['attention_mask']).float().cpu().numpy()
                    else:
                        if self.method=='physics_zero':f=np.zeros_like(f)
                        elif self.method=='physics_shuffle':f=f[np.roll(np.arange(9),1)]
                        elif self.method in ['physics_fno','physics_soft']:f=prediction(self.alternative,field,obs,commands)[1].cpu().numpy()
                        arrays={k:v[None] for k,v in packed(obs,self.normalization,self.mode).items()}
                        scores,tokens=self.policy(tensor_batch(arrays,'cuda'),torch.tensor(commands[None],device='cuda',dtype=torch.float32),
                                                  torch.tensor(f[None],device='cuda',dtype=torch.float32))
                        scores=scores[0].float().cpu().numpy();out['tokens']=int(tokens)
                require(np.isfinite(scores).all(),'Nonfinite policy score')
                choice=int(np.where(allowed,scores,-np.inf).argmax());prob=np.zeros(9)
                active=scores[allowed]/self.temperature;active-=active.max();prob[allowed]=np.exp(active)/np.exp(active).sum()
                command=commands[choice,0]
                require(np.isfinite(command).all() and np.all(abs(command-obs.previous)<=.04000001),'Invalid slew/finite command')
                out.update(choice=choice,command=command.tolist(),scores=scores.tolist(),probabilities=prob.tolist(),fallback_reason=None,
                           filter_intervention=choice!=int(np.where(valid,scores,-np.inf).argmax()))
        b.record();torch.cuda.synchronize()
        out['e2e_ms']=(time.perf_counter_ns()-start)/1e6;out['gpu_compute_ms']=a.elapsed_time(b)
        out['scope']='raw snapshot -> modality view -> observer -> PINO -> state/candidate score -> calibrated gate -> CPU command'
        return out
