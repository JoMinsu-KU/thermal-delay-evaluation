"""Streaming synthetic episodes, causal observations, and explicit truth/label boundaries."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .common import read_json, write_json, save_array, require, object_hash
from .physics import PIController, PROBES, advance, restrict, resize, probe_values


def split_plan(config):
    counts=config['poc']['data']['profiles'][config['profile']]
    plan=[]
    for split,key in [('train','train_episodes'),('validation','validation_episodes'),
                      ('calibration','calibration_episodes'),('test','test_episodes'),('ood','ood_episodes')]:
        for index in range(counts[key]):
            code=len(plan)
            plan.append({'code':code,'episode_id':f'thermal_{code:06d}','group_id':f'plant_recipe_{code:06d}',
                         'split':split,'sim_seed':config['seed']*10000+code,
                         'noise_seed':config['seed']*10000+code+10**8,
                         'render_seed':config['seed']*10000+code+2*10**8,
                         'ood_group':['parameters','observation','delay','mismatch'][index%4] if split=='ood' else None})
    return plan


def initial_field(rng,n,mean,stress=False):
    x=(np.arange(n)+.5)/n; y=x[:,None]
    field=mean+1.5*np.cos(2*np.pi*x)[None,:]*np.cos(2*np.pi*y)
    field+=rng.uniform(-1,1)*np.cos(np.pi*x)[None,:]
    if stress:
        field+=5*np.exp(-((x[None,:]-.25)**2+(y-.25)**2)/.01)
    return field


def generate_episode(base,entry,config,duration=60.):
    """Production duration is 60 s; short duration is used only by isolated tests."""
    raw=base/'raw'/entry['episode_id']; truth=base/'truth'/entry['episode_id']
    if (raw/'COMPLETE').is_file() and (truth/'COMPLETE').is_file():
        return read_json(raw/'meta.json')
    raw.mkdir(parents=True,exist_ok=True); truth.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(entry['sim_seed']); noise=np.random.default_rng(entry['noise_seed'])
    render=np.random.default_rng(entry['render_seed'])
    group=entry['ood_group']
    passport=np.array([rng.uniform(8000,16000),rng.uniform(.4,1.2),rng.uniform(10,40),
                       rng.uniform(20,30),40000,*rng.uniform(.05,.2,4)])
    if group=='parameters':
        passport[0]=19000; passport[1]=1.5; passport[5:]=.3
    goal=rng.uniform(70,95); strategy=rng.choice(['pi','perturbation','stress'],p=[.6,.25,.15])
    if 'strategy_override' in entry:
        require(entry['split']=='pilot_eval' and entry['strategy_override'] in ['pi','perturbation','stress'], 'Override only for preregistered evaluation strata')
        strategy=entry['strategy_override']
    stress=strategy=='stress'; mean=116. if stress else rng.uniform(40,70)
    field=initial_field(rng,64,mean,stress); p=np.zeros(4); u=np.zeros(4)
    initial_fine=field.copy()
    controls=int(round(duration*10)); ticks=controls*10; frames=ticks//20+1
    sensor=np.zeros((ticks+1,16),np.float32); sv=np.zeros_like(sensor,dtype=bool)
    st=np.zeros((ticks+1,2),np.int64); sr=np.zeros(ticks+1,np.int64)
    thermal=np.zeros((frames,1,16,16),np.float32); tv=np.ones_like(thermal,dtype=bool)
    tt=np.zeros((frames,2),np.int64); tr=np.zeros(frames,np.int64)
    commands=np.zeros((controls,3,4),np.float32); ct=np.zeros((controls,3),np.int64)
    plc=np.zeros((controls,12),np.float32)
    states=np.zeros((controls+1,1,32,32),np.float32); actual=np.zeros((controls+1,4),np.float32)
    peaks=np.zeros(controls+1,np.float32); pi=PIController(); bias=render.normal(0,.3)
    disturbances=[]; last_probe=np.full(4,passport[3]); mismatch=False
    for tick in range(ticks+1):
        now=tick*10_000_000
        values=np.concatenate([probe_values(field),p,p])
        std=np.array([.2]*8+[.005]*8)*(4 if group=='observation' else 1)
        sensor[tick]=values+noise.normal(0,std)
        sv[tick]=noise.random(16)>(.1 if group=='observation' else .01)
        receive=now+int(noise.integers(0,8_000_001))+(25_000_000 if group=='delay' else 0)
        st[tick]=[now,receive]; sr[tick]=receive+1_000_000
        if tick%20==0:
            j=tick//20
            thermal[j,0]=restrict(field,16)+bias+render.normal(0,.4, (16,16))
            if render.random()<.25 or group=='observation':
                a,b=render.integers(0,14,2); tv[j,0,a:a+2,b:b+2]=False
            rec=now+int(render.integers(2_000_000,20_000_001))+(300_000_000 if group=='delay' else 0)
            tt[j]=[now,rec]; tr[j]=rec+3_000_000
        if tick%10==0:
            step=tick//10; states[step,0]=restrict(field,32); actual[step]=p; peaks[step]=field.max()
            if step<controls:
                available_probes=np.zeros(4,dtype=bool)
                for j in range(4):
                    usable=np.flatnonzero((sr[:tick+1]<=now)&sv[:tick+1,j]&(now-st[:tick+1,0]<=100_000_000))
                    if len(usable):
                        last_probe[j]=sensor[usable[-1],j]; available_probes[j]=True
                # Probe order is low-x/low-y, low-x/high-y, high-x/low-y, high-x/high-y.
                temperatures=last_probe[[0,2,1,3]]
                requested=pi.command(temperatures,goal,u) if available_probes.all() else np.zeros(4)
                if strategy=='perturbation' and available_probes.all(): requested+=rng.normal(0,.04,4)
                filtered=np.clip(np.clip(requested,u-.04,u+.04),0,1)
                fallback=bool(np.max(temperatures)>120 or not available_probes.all())
                if fallback: filtered=np.zeros(4)  # explicit shutdown/cold-start mode
                commands[step]=np.stack([requested,filtered,filtered]); ct[step]=now
                feedback=sensor[max(0,tick-1),12:16]
                plc[step]=[float(strategy=='stress'),float(fallback),*u,*feedback,0.,float(available_probes.all())]
                u=filtered
        if tick==ticks: break
        effective=passport.copy()
        if group=='mismatch' and tick>=ticks//2:
            effective[2]*=1.5
            if not mismatch:
                disturbances.append({'apply_ns':now,'type':'unmodeled_surface_loss','factor':1.5}); mismatch=True
        field,p=advance(field,p,u,effective,duration=.01)
    for name,array in {'sensor':sensor,'sensor_valid':sv,'sensor_times':st,'sensor_ready':sr,
                       'thermal':thermal,'thermal_valid':tv,'thermal_times':tt,'thermal_ready':tr,
                       'commands':commands,'command_times':ct,'plc':plc}.items(): save_array(raw/(name+'.npy'),array)
    for name,array in {'T32':states,'p_actual':actual,'fine_peak':peaks,'initial_T64':initial_fine}.items(): save_array(truth/(name+'.npy'),array)
    (truth/'disturbances.jsonl').write_text(''.join(__import__('json').dumps(x)+'\n' for x in disturbances))
    meta={**entry,'schema_version':'ojpoc.v1','duration_s':duration,'control_dt_s':.1,'passport':passport.tolist(),
          'goal_c':float(goal),'recipe':[float(goal),1.,.5,.002,.02],'actuation_enabled':False,
          'candidate_library':'axis9_poc_v1','cost_version':'thermal_poc_v1','solver_version':'fv64-explicit-v1',
          'strategy':strategy,'source_commit':(config.get('openjev_commit') or 'recorded-in-run-provenance'),
          'schema_hash':object_hash({'sensor':[ticks+1,16],'thermal':[frames,1,16,16],'T32':[controls+1,1,32,32]}),
          'reason':'near_limit_stress' if stress else ('ood_'+group if group else 'normal_id'),
          'initial_peak_c':float(peaks[0]),'final_mean_c':float(states[-1].mean()),
          'target_reached_within_2k':bool(np.any(np.abs(states.mean(axis=(1,2,3))-goal)<=2)),
          'temperature_violation':bool(np.any(peaks>120))}
    write_json(raw/'meta.json',meta)
    from .common import seal
    seal(raw,{'episode':entry['episode_id']}); seal(truth,{'role':'supervision_and_evaluation_only'})
    return meta


@dataclass
class Observation:
    sensor: np.ndarray
    sensor_valid: np.ndarray
    sensor_age: np.ndarray
    thermal: np.ndarray
    thermal_valid: np.ndarray
    thermal_age: np.ndarray
    previous: np.ndarray
    actuator: np.ndarray
    passport: np.ndarray
    goal: float
    decision_ns: int
    usable: bool
    reason: str | None


class RawEpisode:
    """This reader has no truth path and cannot expose supervision to a policy."""
    def __init__(self,path):
        self.path=Path(path); self.meta=read_json(self.path/'meta.json'); self.arrays={}

    def array(self,name):
        allowed={'sensor','sensor_valid','sensor_times','sensor_ready','thermal','thermal_valid','thermal_times',
                 'thermal_ready','commands','command_times','plc'}
        require(name in allowed,'Truth is not an observation')
        if name not in self.arrays:
            self.arrays[name]=np.load(self.path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        return self.arrays[name]

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def snapshot(self,step):
        return self.snapshot_ns(int(step)*100_000_000,command_limit=int(step))

    def snapshot_ns(self,decision,command_limit=None):
        decision=int(decision)
        times=self.array('sensor_times'); ready=self.array('sensor_ready')
        eligible=(times[:,1]<=decision)&(ready<=decision)&(times[:,0]<=decision)
        values=np.zeros((64,16),np.float32); valid=np.zeros((64,16),bool); age=np.full((64,16),1.,np.float32)
        grid=decision-np.arange(63,-1,-1,dtype=np.int64)*10_000_000
        for channel in range(16):
            ids=np.flatnonzero(eligible & self.array('sensor_valid')[:,channel])
            if len(ids)==0: continue
            captures=times[ids,0]; selected=np.searchsorted(captures,grid,side='right')-1
            mask=selected>=0; safe=np.maximum(selected,0); indices=ids[safe]
            ages=(decision-times[indices,0])/1e9
            # Historical samples stay in the window; latest validity obeys sensor_max_age.
            mask &= (grid>=0)&((grid-times[indices,0])<=100_000_000)
            values[mask,channel]=self.array('sensor')[indices[mask],channel]
            age[mask,channel]=ages[mask]; valid[mask,channel]=True
        tt=self.array('thermal_times'); tr=self.array('thermal_ready')
        ids=np.flatnonzero((tt[:,0]<=decision)&(tt[:,1]<=decision)&(tr<=decision)&(decision-tt[:,0]<=500_000_000))[-2:]
        frames=np.zeros((2,1,16,16),np.float32); fm=np.zeros_like(frames,bool); fa=np.full(2,1.,np.float32)
        for index,frame in enumerate(ids,start=2-len(ids)):
            frames[index]=self.array('thermal')[frame]; fm[index]=self.array('thermal_valid')[frame]
            fa[index]=(decision-tt[frame,0])/1e9
        ct=self.array('command_times')[:command_limit]
        applied=np.flatnonzero((ct[:,1]<=decision)&(ct[:,2]<=decision))
        previous=np.array(self.array('commands')[applied[-1],2],copy=True) if len(applied) else np.zeros(4)
        usable=bool(fm[-1].mean()>=.5 and valid[-1,:8].sum()>=4 and valid[-1,12:16].all())
        return Observation(values,valid,age,frames,fm,fa,previous,np.clip(values[-1,12:16],0,1),
                           np.array(self.meta['passport']),self.meta['goal_c'],decision,usable,
                           None if usable else 'cold_start_or_missing_observation')


class EpisodeCache:
    def __init__(self,base,entries):
        self.base=Path(base); self.entries={e['code']:e for e in entries}; self.open=OrderedDict()

    def get(self,code):
        code=int(code)
        if code not in self.open:
            if len(self.open)>=2:
                _,old=self.open.popitem(last=False); old.close()
            self.open[code]=RawEpisode(self.base/'raw'/self.entries[code]['episode_id'])
        self.open.move_to_end(code)
        return self.open[code]

    def close(self):
        for episode in self.open.values(): episode.close()
        self.open.clear()


def bilinear_observer(obs):
    require(obs.usable,'Observer requires usable observations; invoke fallback otherwise')
    frame=obs.thermal[-1,0].copy(); mask=obs.thermal_valid[-1,0]
    if not mask.all():
        ids=distance_transform_edt(~mask,return_distances=False,return_indices=True)
        frame[~mask]=frame[tuple(ids[:,~mask])]
    field=resize(frame.astype(np.float64),32)
    valid=obs.sensor_valid[-1,:8]; residual=obs.sensor[-1,:8]-probe_values(field)
    x=(np.arange(32)+.5)*.5/32; yy,xx=np.meshgrid(x,x,indexing='ij')
    weights=1/((xx[None]-PROBES[:,0,None,None])**2+(yy[None]-PROBES[:,1,None,None])**2+.02**2)
    weights*=valid[:,None,None]
    field+=(weights*residual[:,None,None]).sum(0)/np.maximum(weights.sum(0),1e-12)
    return field.astype(np.float32)


def build_index(base,entries,config):
    index=base/'index'; index.mkdir(exist_ok=True)
    anchors=config['budget']['anchors_per_episode']; rows={'train':[],'validation':[]}
    fallback=[]; squared=[]; bias=[]; peaks=[]; first_valid={}
    sums=np.zeros(16); squares=np.zeros(16); counts=np.zeros(16)
    for entry in entries:
        if entry['split'] not in rows: continue  # calibration/test/OOD sealed for later evaluation
        episode=RawEpisode(base/'raw'/entry['episode_id'])
        if entry['split']=='train':
            x=episode.array('sensor'); m=episode.array('sensor_valid')
            sums+=(x*m).sum(0,dtype=np.float64); squares+=(x.astype(np.float64)**2*m).sum(0); counts+=m.sum(0)
        last=len(episode.array('commands'))-20
        require(last>=1,'Not enough future horizon')
        truth=np.load(base/'truth'/entry['episode_id']/'T32.npy',mmap_mode='r')
        for step in sorted(set(np.linspace(1,last,anchors,dtype=int))):
            obs=episode.snapshot(int(step))
            if not obs.usable:
                fallback.append({'code':entry['code'],'step':int(step),'reason':obs.reason}); continue
            first_valid.setdefault(entry['episode_id'],int(step)*100_000_000)
            rows[entry['split']].append([entry['code'],int(step),1])
            if entry['split']=='validation':
                estimated=bilinear_observer(obs); error=estimated-truth[step,0]
                squared.append(float((error**2).mean())); bias.append(float(error.mean()))
                peaks.append(float(abs(estimated.max()-truth[step,0].max())))
        truth._mmap.close(); episode.close()
    for split,array in rows.items():
        require(len(array)>0,f'No usable {split} anchors')
        save_array(index/(split+'_windows.npy'),np.array(array,np.int64))
    mean=sums/np.maximum(counts,1); std=np.sqrt(np.maximum(squares/np.maximum(counts,1)-mean**2,1e-8))
    write_json(base/'manifests/normalization.json',{'source_split':'train','sensor_mean':mean.tolist(),
                'sensor_std':std.tolist(),'count':counts.tolist(),'operator_temperature':'(T-25)/100',
                'native_text':'physical units; scaler not applied to text'})
    write_json(index/'fallback.json',{'rows':fallback,'first_valid_sampled_anchor_ns':first_valid})
    return {'counts':{k:len(v) for k,v in rows.items()},'fallback_anchors':len(fallback),
            'observer':'O-bilinear','validation_rmse_k':float(np.sqrt(np.mean(squared))),
            'validation_bias_k':float(np.mean(bias)),'validation_peak_error_k':float(np.max(peaks)),
            'g2':'baseline_selected_no_learned_observer_claim','test_accessed':False}


def validate_dataset(base,entries):
    from .common import verify
    groups=set(); checks=[]
    for entry in entries:
        require(entry['group_id'] not in groups,'Group leakage'); groups.add(entry['group_id'])
        raw=base/'raw'/entry['episode_id']; truth=base/'truth'/entry['episode_id']
        verify(raw); verify(truth)
        meta=read_json(raw/'meta.json'); require(meta['split']==entry['split'],'Split mismatch')
        steps=int(round(meta['duration_s']*10))
        expected={'sensor':(steps*10+1,16),'sensor_valid':(steps*10+1,16),'sensor_times':(steps*10+1,2),
                  'sensor_ready':(steps*10+1,),'thermal':(steps//2+1,1,16,16),'thermal_valid':(steps//2+1,1,16,16),
                  'thermal_times':(steps//2+1,2),'thermal_ready':(steps//2+1,),
                  'commands':(steps,3,4),'command_times':(steps,3),'plc':(steps,12)}
        for name,shape in expected.items():
            a=np.load(raw/(name+'.npy'),mmap_mode='r'); require(a.shape==shape,f'Shape mismatch {name}')
            require(np.isfinite(a).all(),f'Nonfinite {name}'); a._mmap.close()
        for name,shape in {'T32':(steps+1,1,32,32),'p_actual':(steps+1,4),'fine_peak':(steps+1,), 'initial_T64':(64,64)}.items():
            a=np.load(truth/(name+'.npy'),mmap_mode='r')
            require(a.shape==shape and np.isfinite(a).all(),f'Invalid truth schema: {name}')
            a._mmap.close()
        for name in ['sensor','thermal']:
            times=np.load(raw/(name+'_times.npy'),mmap_mode='r'); ready=np.load(raw/(name+'_ready.npy'),mmap_mode='r')
            require(times.dtype==np.int64 and ready.dtype==np.int64,'Times must be int64 ns')
            require(np.all(np.diff(times[:,0])>0) and np.all(times[:,1]>=times[:,0]) and np.all(ready>=times[:,1]),'Timestamp contract')
            times._mmap.close(); ready._mmap.close()
        commands=np.load(raw/'commands.npy',mmap_mode='r'); require(np.all((commands[:,2]>=0)&(commands[:,2]<=1)),'Command bounds')
        command_times=np.load(raw/'command_times.npy',mmap_mode='r')
        require(command_times.dtype==np.int64 and np.all(np.diff(command_times[:,2])>0) and
                np.all(command_times[:,0]<=command_times[:,1]) and np.all(command_times[:,1]<=command_times[:,2]),'Invalid command ACK/apply times')
        command_times._mmap.close()
        changes=np.abs(np.diff(np.vstack([np.zeros((1,4)),commands[:,2]]),axis=0))
        violations=(changes>.040001).any(axis=1)
        plc=np.load(raw/'plc.npy',mmap_mode='r'); require(np.all(~violations | (plc[:,1]>0)), 'Undeclared slew exception')
        commands._mmap.close(); plc._mmap.close()
        checks.append({'episode':entry['episode_id'],'reason':meta['reason'],'temperature_violation':meta['temperature_violation'],
                       'target_reached_within_2k':meta['target_reached_within_2k']})
    return {'episodes':len(checks),'group_overlap':False,'checks':checks,'test_performance_evaluated':False}
