"""New split, causal-only modality views and independently perturbed OOD plants."""
from dataclasses import replace
import numpy as np
from scipy.ndimage import distance_transform_edt
from ..common import read_json, write_json, save_array, seal, verify, require
from ..data import RawEpisode, Observation, initial_field, bilinear_observer
from ..control import LivePlant
from ..physics import advance, restrict, resize, PROBES, PIController
from .contracts import plan, anchors


def view(obs, mode='both'):
    require(mode in ['both','sensor','thermal'],'Unknown modality')
    if mode=='both': return obs
    sensor=obs.sensor.copy(); valid=obs.sensor_valid.copy(); age=obs.sensor_age.copy()
    thermal=obs.thermal.copy(); mask=obs.thermal_valid.copy(); ta=obs.thermal_age.copy()
    feedback=bool(valid[-1,12:16].all())
    if mode=='sensor':
        thermal[:]=0; mask[:]=False; ta[:]=1
        usable=bool(valid[-1,:8].sum()>=4 and feedback)
    else:
        sensor[:,:8]=0; valid[:,:8]=False; age[:,:8]=1
        usable=bool(mask[-1].mean()>=.5 and feedback)
    return replace(obs,sensor=sensor,sensor_valid=valid,sensor_age=age,thermal=thermal,
                   thermal_valid=mask,thermal_age=ta,usable=usable,
                   reason=None if usable else 'cold_start_or_missing_observation')


def baseline_field(obs, mode='both'):
    require(obs.usable,'Fallback required')
    if mode=='both': return bilinear_observer(obs)
    if mode=='thermal':
        frame=obs.thermal[-1,0].copy(); mask=obs.thermal_valid[-1,0]
        if not mask.all():
            ids=distance_transform_edt(~mask,return_distances=False,return_indices=True)
            frame[~mask]=frame[tuple(ids[:,~mask])]
        return resize(frame.astype(float),32).astype(np.float32)
    xy=(np.arange(32)+.5)*.5/32; yy,xx=np.meshgrid(xy,xy,indexing='ij')
    weights=1/((xx[None]-PROBES[:,0,None,None])**2+(yy[None]-PROBES[:,1,None,None])**2+.02**2)
    weights*=obs.sensor_valid[-1,:8,None,None]
    return ((weights*obs.sensor[-1,:8,None,None]).sum(0)/np.maximum(weights.sum(0),1e-12)).astype(np.float32)


def packed(obs, normalization, mode='both'):
    obs=view(obs,mode)
    mean=np.asarray(normalization['sensor_mean']); std=np.maximum(normalization['sensor_std'],1e-4)
    sensor=np.stack([np.where(obs.sensor_valid,(obs.sensor-mean)/std,0),obs.sensor_valid,
                     np.where(obs.sensor_valid,np.clip(obs.sensor_age,0,1),1)],0).astype(np.float32)
    thermal=np.concatenate([np.where(obs.thermal_valid[:,0],(obs.thermal[:,0]-25)/100,0),
                            obs.thermal_valid[:,0],np.broadcast_to(obs.thermal_age[:,None,None],(2,16,16))],0).astype(np.float32)
    # PLC is explicit previous command/actuator feedback and availability, never true plant state.
    plc=np.r_[obs.previous,obs.actuator,float(obs.usable),float(obs.reason is not None),
              np.clip(obs.sensor_age[-1,12:16].max(),0,1),1.].astype(np.float32)
    scales=np.array([12000,.75,25,25,40000,.1,.1,.1,.1])
    passport=np.r_[obs.passport/scales,(obs.goal-25)/100,1.,.5,.002,.02].astype(np.float32)
    return {'sensor':sensor,'thermal':thermal,'plc':plc,'passport':passport}


class Plant(LivePlant):
    """Known passport vs hidden plant mismatch are deliberately separate."""
    def __init__(self,base,entry,duration_s):
        super().__init__(base,entry,duration_s)
        self.condition=entry.get('ood_condition'); self.nominal=self.passport.copy()
        self.mismatch_value=read_json(base/'truth'/entry['episode_id']/'disturbances.json')['truth_only_mismatch_value']
        rng=np.random.default_rng(entry['noise_seed']+17)
        if self.condition=='probe_noise': self.noise[:,:8]=rng.normal(0,1.,self.noise[:,:8].shape)
        if self.condition=='thermal_noise': self.thermal_noise=rng.normal(0,1.5,self.thermal_noise.shape)
        if self.condition=='sensor_burst':
            start=int(round(duration_s*.4*100)); self.buffers['sensor_valid'][start:start+20]=False
        if self.condition=='thermal_occlusion':
            for frame in self.buffers['thermal_valid']:
                frame[:]=True; frame.reshape(-1)[rng.choice(256,size=77,replace=False)]=False
        self.emit()

    def advance_to(self,end_ns):
        require(self.now<=end_ns<=self.end_ns,'Plant clock must be monotone')
        while self.now<end_ns:
            next_tick=(self.tick+1)*10_000_000
            change=int(self.end_ns//2)
            following=min(end_ns,next_tick,change if self.now<change else self.end_ns)
            dt=(following-self.now)/1e9; theta=self.nominal.copy()
            if self.now>=change:
                if self.condition=='heater_gain': theta[4]*=self.mismatch_value
                if self.condition=='surface_loss': theta[2]+=self.mismatch_value
            self.field,self.p=advance(self.field,self.p,self.u,theta,duration=dt)
            self.error_integral+=float(((self.field-self.goal)**2).mean())*dt
            self.effort_integral+=float((self.u**2).mean())*dt
            peak=float(self.field.max()); violation=peak>120
            self.violation_s+=dt*violation;self.fallback_s+=dt*self.fallback
            self.peak=max(self.peak,peak);self.overshoot=max(self.overshoot,max(0.,peak-self.goal))
            if violation and not self.previous_violation:self.new_violation=True
            if self.initial_violation and not violation and self.recovery_s is None:self.recovery_s=following/1e9
            self.previous_violation=violation;self.now=following
            if self.now==next_tick:self.tick+=1;self.emit()


def generate_one(base,entry,cfg):
    raw=base/'raw'/entry['episode_id']; truth=base/'truth'/entry['episode_id']
    if (raw/'COMPLETE').exists() and (truth/'COMPLETE').exists():
        verify(raw);verify(truth);return read_json(raw/'meta.json')
    raw.mkdir(parents=True,exist_ok=True);truth.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(entry['sim_seed']); duration=cfg['budget']['duration_s']
    passport=np.r_[rng.uniform(8000,16000),rng.uniform(.4,1.2),rng.uniform(10,40),rng.uniform(20,30),40000,rng.uniform(.05,.2,4)]
    condition=entry['ood_condition']
    if condition=='capacity_low':passport[0]=rng.uniform(6000,7500)
    if condition=='capacity_high':passport[0]=rng.uniform(17000,20000)
    if condition=='conductivity_low':passport[1]=rng.uniform(.25,.35)
    if condition=='conductivity_high':passport[1]=rng.uniform(1.3,1.6)
    stress=entry['stratum']=='stress'; goal=float(rng.uniform(70,95))
    field=initial_field(rng,64,116. if stress else rng.uniform(40,70),stress)
    meta={**entry,'schema_version':'thermal-l2-v1','duration_s':duration,'control_dt_s':.1,'passport':passport.tolist(),
          'goal_c':goal,'recipe':[goal,1.,.5,.002,.02],'actuation_enabled':False,'candidate_library':'axis9_poc_v1',
          'cost_version':'thermal_poc_v1','solver_version':'fv64-explicit-l2-ood-v1',
          'strategy':'stress' if stress else 'pi' if entry['code']%2==0 else 'perturbation',
          'reason':'near_limit_stress' if stress else ('ood_'+entry['ood_group'] if entry['ood_group'] else 'normal_id'),
          'mismatch_value':float(rng.uniform(.7,.9) if condition=='heater_gain' else rng.uniform(10,20)),
          'initial_peak_c':float(field.max()),'hidden_mismatch_in_policy_passport':False}
    mismatch=meta.pop('mismatch_value')
    write_json(truth/'disturbances.json',{'condition':condition,'switch_ns':int(duration*1e9)//2,
                                        'truth_only_mismatch_value':mismatch if entry['ood_group']=='mismatch' else None})
    write_json(raw/'meta.json',meta);save_array(truth/'initial_T64.npy',field)
    plant=Plant(base,entry,duration); controller=PIController(); n=round(duration*10)
    states=[]; actuators=[]; peaks=[]; plc=[]; requested=[]; fine_anchors=[];anchor_set=set(map(int,anchors(cfg)))
    for step in range(n+1):
        plant.advance_to(step*100_000_000)
        states.append(restrict(plant.field,32)[None]);actuators.append(plant.p.copy());peaks.append(plant.field.max())
        if step in anchor_set:fine_anchors.append(plant.field.copy())
        if step==n:break
        obs=plant.raw().snapshot_ns(plant.now)
        fallback=not obs.usable
        command=np.zeros(4)
        if obs.usable:
            f=baseline_field(obs);zones=[f[:16,:16].mean(),f[:16,16:].mean(),f[16:,:16].mean(),f[16:,16:].mean()]
            command=controller.command(np.array(zones),goal,plant.u)
            if meta['strategy']=='perturbation':command+=rng.normal(0,.04,4)
            command=np.clip(np.clip(command,plant.u-.04,plant.u+.04),0,1)
            fallback=bool(f.max()>120)
            if fallback:command=np.zeros(4)
        requested.append(command.copy());plc.append([float(stress),float(fallback),*plant.u,*obs.actuator,0.,float(obs.usable)])
        plant.apply(command,fallback)
    arrays={**plant.buffers,'commands':np.array(plant.command_events),'command_times':np.array(plant.command_times,dtype=np.int64),'plc':np.array(plc,np.float32)}
    for key,value in arrays.items():save_array(raw/(key+'.npy'),value)
    for key,value in {'T32':states,'p_actual':actuators,'fine_peak':peaks}.items():save_array(truth/(key+'.npy'),np.array(value,np.float32))
    save_array(truth/'anchor_T64.npy',np.asarray(fine_anchors,np.float32))
    meta.update(final_mean_c=float(np.mean(states[-1])),target_reached_within_2k=bool(np.any(np.abs(np.mean(states,axis=(1,2,3))-goal)<=2)),
                temperature_violation=bool(max(peaks)>120))
    write_json(raw/'meta.json',meta)
    seal(raw,{'role':'causal_input'});seal(truth,{'role':'labels_and_evaluation_only'})
    return meta


def generate(run,cfg,output):
    base=output/'data';(base/'manifests').mkdir(parents=True,exist_ok=True)
    rows=plan(cfg);write_json(base/'manifests/split.json',{'episodes':rows,'seed':cfg['seed'],'group_split_before_generation':True})
    for i,entry in enumerate(rows):
        generate_one(base,entry,cfg);print(f'L2 data {i+1}/{len(rows)} {entry["split"]}',flush=True)
    return {'episodes':len(rows),'counts':cfg['budget']['counts'],'duration_s':cfg['budget']['duration_s'],
            'ood_contract':'single preregistered perturbation per episode; nominal passport retained for mismatch'}


def causal_index(run,cfg,output):
    from ..data import validate_dataset
    base=run/'02_generate/data'; rows=read_json(base/'manifests/split.json')['episodes']
    sums=np.zeros(16);sq=np.zeros(16);counts=np.zeros(16);windows={s:[] for s in ['train','validation']}
    for entry in rows:
        if entry['split'] not in windows:continue
        raw=RawEpisode(base/'raw'/entry['episode_id'])
        try:
            if entry['split']=='train':
                x=raw.array('sensor').astype(float);mask=raw.array('sensor_valid')
                sums+=(x*mask).sum(0);sq+=(x*x*mask).sum(0);counts+=mask.sum(0)
            for step in anchors(cfg): windows[entry['split']].append([entry['code'],int(step)])
        finally:raw.close()
    mu=sums/np.maximum(counts,1);sigma=np.sqrt(np.maximum(sq/np.maximum(counts,1)-mu*mu,1e-8))
    write_json(output/'normalization.json',{'source_split':'train','sensor_mean':mu.tolist(),'sensor_std':sigma.tolist(),'counts':counts.tolist()})
    for split,value in windows.items():save_array(output/(split+'_index.npy'),np.array(value,np.int64))
    # Schema checks on raw/truth do not select models using held-out performance.
    result=validate_dataset(base,rows)
    result['schema_validation_only']=True;result['normalization_fit_split']='train'
    return result
