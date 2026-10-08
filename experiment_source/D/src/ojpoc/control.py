"""Observation-only inference and event-driven FV plant; no real equipment interface."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import time
import numpy as np
import torch

from .common import require, read_json, verify
from .data import RawEpisode, bilinear_observer
from .physics import candidates, branch, costs, PIController, advance, probe_values, restrict
from .operator import ThermalOperator, physics_features
from .evaluation import tensor_inputs

METHODS=['pi','fv_mpc','pino_mpc','no_physics','physics']


class MemoryEpisode(RawEpisode):
    """Raw buffers only. The controller cannot retrieve the plant's truth state."""
    def __init__(self,meta,arrays):
        self.meta=meta; self.arrays=arrays

    def array(self,name):
        require(name in self.arrays,'Unknown raw observation field')
        return self.arrays[name]

    def close(self):
        pass


def freeze_request(raw,step):
    """Bounded raw-input ring entry; snapshot preparation remains inside timed inference."""
    arrays={}; tick=step*10; frame=tick//20
    for key in ['sensor','sensor_valid','sensor_times','sensor_ready']:
        arrays[key]=np.array(raw.array(key)[max(0,tick-80):tick+1],copy=True)
    for key in ['thermal','thermal_valid','thermal_times','thermal_ready']:
        arrays[key]=np.array(raw.array(key)[max(0,frame-4):frame+1],copy=True)
    for key in ['commands','command_times']:
        arrays[key]=np.array(raw.array(key)[max(0,step-2):step],copy=True)
    return MemoryEpisode(dict(raw.meta),arrays)


class DecisionEngine:
    def __init__(self,root,run,method,checkpoint='selected',device='cuda',load_native=True):
        require(method in METHODS,'Unknown controller'); self.method=method; self.device=device
        self.operator=ThermalOperator('pino_balance').to(device).eval()
        self.operator.load_state_dict(torch.load(run/'07_pino_balance/best.pt',map_location=device,weights_only=True))
        self.operator.requires_grad_(False); self.native=None; self.pi=PIController()
        if method in ['no_physics','physics'] and load_native:
            if checkpoint=='baseline':
                from .policy import load_original
                self.native=load_original(root)
            else:
                from jev.model import DecisionModel
                stage='09_native_no_physics' if method=='no_physics' else '11_native_physics_text'
                folder=run/stage/('best_model' if checkpoint=='selected' else 'adapt_model'); verify(folder)
                self.native=DecisionModel.load(folder,device=device)
            self.native.eval(); self.native.requires_grad_(False)
            self.native.backbone.gradient_checkpointing_disable()
        self.events=[]

    @contextmanager
    def gpu_interval(self):
        if self.device.startswith('cuda'):
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); yield; end.record(); self.events.append((start,end))
        else: yield

    def reset(self):
        self.pi=PIController()

    def close(self):
        self.native=None; self.operator=None; gc.collect()
        if self.device.startswith('cuda'): torch.cuda.empty_cache()

    def decide(self,make_observation):
        # Caller supplies a raw snapshot factory, not cached tokens, fields or PINO features.
        if self.device.startswith('cuda'): torch.cuda.synchronize()
        started=time.perf_counter_ns(); self.events=[]
        obs=make_observation(); commands,valid=candidates(obs.previous)
        result={'choice':-1,'command':[0.]*4,'fallback_reason':obs.reason,'tokens':0,'scores':None,
                'probabilities':None,'thermal_age_s':float(obs.thermal_age[-1]),'method':self.method,'filter_intervention':False}
        if obs.usable:
            field=bilinear_observer(obs); inputs=tensor_inputs(field,obs,commands,self.device)
            with torch.inference_mode():
                previous=torch.tensor(np.repeat(obs.previous[None],9,0),device=self.device,dtype=torch.float32)
                goal=torch.full((9,),obs.goal,device=self.device); age=torch.full_like(goal,float(obs.thermal_age[-1]))
                with self.gpu_interval():
                    future=self.operator(*inputs)
                    features=physics_features(future,inputs[0],inputs[2],inputs[3],inputs[1],previous,goal,age)
                feature=features.cpu().numpy()
                require(np.isfinite(feature).all(),'Nonfinite online physics features')
                allowed=valid&(feature[:,2]<=120)
                result['valid_mask']=valid.tolist(); result['predicted_safe_mask']=allowed.tolist()
                if not allowed.any():
                    result['fallback_reason']='no_predicted_feasible_candidate'; result['filter_intervention']=True
                else:
                    if self.method in ['no_physics','physics']:
                        from .policy import serialize,tokenize,score_tokens
                        records=serialize(obs,field,commands,feature if self.method=='physics' else None)
                        encoded=tokenize(self.native,records)
                        result['tokens']=int(encoded['attention_mask'].sum(-1).max())
                        ids=encoded['input_ids'].to(self.device); mask=encoded['attention_mask'].to(self.device)
                        with self.gpu_interval(): scores=score_tokens(self.native,ids,mask).float()
                        scores=scores.cpu().numpy()
                    elif self.method=='pi':
                        zones=np.array([field[:16,:16].mean(),field[:16,16:].mean(),field[16:,:16].mean(),field[16:,16:].mean()])
                        requested=self.pi.command(zones,obs.goal,obs.previous)
                        scores=-((commands[:,0]-requested)**2).mean(1)
                    else:
                        prediction=branch(field,obs.actuator,commands,obs.passport) if self.method=='fv_mpc' else future.cpu().numpy()
                        scores=-costs(prediction,commands,obs.previous,obs.goal)
                    require(np.isfinite(scores).all(),'Nonfinite controller score')
                    choice=int(np.where(allowed,scores,-np.inf).argmax())
                    result['filter_intervention']=choice!=int(np.where(valid,scores,-np.inf).argmax())
                    probability=np.zeros(9); v=scores[allowed]-scores[allowed].max(); probability[allowed]=np.exp(v)/np.exp(v).sum()
                    command=commands[choice,0]
                    require(np.all(abs(command-obs.previous)<=.04000001),'Controller violated slew')
                    result.update(choice=choice,command=command.tolist(),fallback_reason=None,scores=scores.tolist(),probabilities=probability.tolist())
        if self.device.startswith('cuda'): torch.cuda.synchronize()
        result['e2e_ms']=(time.perf_counter_ns()-started)/1e6
        result['gpu_compute_ms']=sum(a.elapsed_time(b) for a,b in self.events)
        result['scope']='raw snapshot -> observer -> candidates -> online PINO -> tokenizer/scorer -> filter -> CPU command'
        return result


class LivePlant:
    """Fine truth is private to the plant/evaluator; raw view exposes only emitted observations."""
    def __init__(self,base,entry,duration_s):
        self.entry=entry; self.meta=read_json(base/'raw'/entry['episode_id']/'meta.json')
        self.field=np.array(np.load(base/'truth'/entry['episode_id']/'initial_T64.npy'),dtype=np.float64)
        self.passport=np.array(self.meta['passport']); self.goal=self.meta['goal_c']; self.p=np.zeros(4); self.u=np.zeros(4)
        self.now=0; self.duration_s=duration_s; self.end_ns=int(round(duration_s*1e9)); self.tick=0
        ticks=int(round(duration_s*100))+1; frames=(ticks-1)//20+1
        self.noise=np.empty((ticks,16)); self.thermal_noise=np.empty((frames,16,16))
        rng=np.random.default_rng(entry['noise_seed']); render=np.random.default_rng(entry['render_seed']); bias=render.normal(0,.3)
        self.buffers={'sensor':np.zeros((ticks,16),np.float32),'sensor_valid':np.zeros((ticks,16),bool),
                      'sensor_times':np.zeros((ticks,2),np.int64),'sensor_ready':np.zeros(ticks,np.int64),
                      'thermal':np.zeros((frames,1,16,16),np.float32),'thermal_valid':np.ones((frames,1,16,16),bool),
                      'thermal_times':np.zeros((frames,2),np.int64),'thermal_ready':np.zeros(frames,np.int64)}
        for tick in range(ticks):
            now=tick*10_000_000; self.noise[tick]=rng.normal(0,np.array([.2]*8+[.005]*8))
            self.buffers['sensor_valid'][tick]=rng.random(16)>.01
            receive=now+int(rng.integers(0,8_000_001)); self.buffers['sensor_times'][tick]=[now,receive]
            self.buffers['sensor_ready'][tick]=receive+1_000_000
            if tick%20==0:
                j=tick//20; self.thermal_noise[j]=bias+render.normal(0,.4,(16,16))
                if render.random()<.25:
                    a,b=render.integers(0,14,2); self.buffers['thermal_valid'][j,0,a:a+2,b:b+2]=False
                receive=now+int(render.integers(2_000_000,20_000_001)); self.buffers['thermal_times'][j]=[now,receive]
                self.buffers['thermal_ready'][j]=receive+3_000_000
        self.command_events=[]; self.command_times=[]; self.fallback=True
        self.error_integral=0.; self.effort_integral=0.; self.delta_sum=0.; self.violation_s=0.; self.fallback_s=0.
        self.peak=float(self.field.max()); self.overshoot=max(0.,self.peak-self.goal); self.initial_violation=self.peak>120
        self.recovery_s=None; self.new_violation=False; self.previous_violation=self.initial_violation
        self.emit()

    def emit(self):
        self.buffers['sensor'][self.tick]=np.concatenate([probe_values(self.field),self.p,self.p])+self.noise[self.tick]
        if self.tick%20==0:
            self.buffers['thermal'][self.tick//20,0]=restrict(self.field,16)+self.thermal_noise[self.tick//20]

    def raw(self):
        arrays={k:v[:self.tick+1 if k.startswith('sensor') else self.tick//20+1] for k,v in self.buffers.items()}
        arrays['commands']=np.array(self.command_events,dtype=np.float64).reshape(-1,3,4)
        arrays['command_times']=np.array(self.command_times,dtype=np.int64).reshape(-1,3)
        return MemoryEpisode(self.meta,arrays)

    def apply(self,command,fallback=False):
        command=np.asarray(command,dtype=np.float64)
        require(command.shape==(4,) and np.isfinite(command).all() and np.all((command>=0)&(command<=1)),'Invalid simulation command')
        require(fallback or np.all(abs(command-self.u)<=.04000001),'Unsafe delayed slew')
        self.delta_sum+=float(((command-self.u)**2).mean()); self.u=command.copy(); self.fallback=fallback
        self.command_events.append(np.stack([command]*3)); self.command_times.append([self.now]*3)

    def advance_to(self,end_ns):
        require(self.now<=end_ns<=self.end_ns,'Plant clock must be monotone')
        while self.now<end_ns:
            following=min(end_ns,(self.tick+1)*10_000_000)
            dt=(following-self.now)/1e9
            self.field,self.p=advance(self.field,self.p,self.u,self.passport,duration=dt)
            self.error_integral+=float(((self.field-self.goal)**2).mean())*dt
            self.effort_integral+=float((self.u**2).mean())*dt
            peak=float(self.field.max()); violation=peak>120
            self.violation_s+=dt*violation; self.fallback_s+=dt*self.fallback
            self.peak=max(self.peak,peak); self.overshoot=max(self.overshoot,max(0.,peak-self.goal))
            if violation and not self.previous_violation: self.new_violation=True
            if self.initial_violation and not violation and self.recovery_s is None: self.recovery_s=following/1e9
            self.previous_violation=violation; self.now=following
            if self.now==(self.tick+1)*10_000_000:
                self.tick+=1; self.emit()

    def metrics(self):
        require(self.now==self.end_ns,'Incomplete episodes cannot enter cost comparison')
        n=int(round(self.duration_s*10)); tracking=self.error_integral/self.duration_s
        terminal=float((((self.field-self.goal)/100.)**2).mean())
        effort=self.effort_integral/self.duration_s; delta=self.delta_sum/n
        return {'episode_cost':tracking/10000+.5*terminal+.002*effort+.02*delta,
                'tracking_rmse_k':float(np.sqrt(tracking)),'overshoot_k':self.overshoot,'max_temperature_c':self.peak,
                'violation':bool(self.initial_violation or self.violation_s>0),'violation_s':self.violation_s,
                'initial_violation':self.initial_violation,'new_violation':self.new_violation,'recovery_s':self.recovery_s,
                'max_violation_k':max(0.,self.peak-120),'fallback_s':self.fallback_s,'action_effort':effort,
                'action_delta_mean_square':delta,'completed':True,'duration_s':self.duration_s,
                'integration':'right endpoint, <=10ms FV substeps; asynchronous action events included in slew cost'}


def choose_latency(trace,decision,request_id,strict=False):
    """Deterministic empirical trace replay matched on fallback and nearest token length."""
    require(bool(trace),'Missing measured latency trace')
    pool=[r for r in trace if (r['fallback_reason']==decision['fallback_reason'] if strict else
                             bool(r['fallback_reason'])==bool(decision['fallback_reason'])) and not r.get('error')]
    if strict: require(bool(pool),f'Missing measured latency path: {decision["fallback_reason"]}')
    elif not pool: pool=[r for r in trace if not r.get('error')]
    require(bool(pool),'No valid latency trace')
    distance=min(abs(r['tokens']-decision['tokens']) for r in pool)
    matched=[r for r in pool if abs(r['tokens']-decision['tokens'])==distance]
    row=matched[request_id%len(matched)]
    return row['e2e_ms'],row['request'],distance


def run_episode(plant,engine,mode='zero',latency_trace=None,stop=None,deadline_ms=100.,strict_latency=False):
    require(mode in ['zero','delayed'],'Unknown SIL mode')
    engine.reset(); trace=[]; pending=None; busy_until=0; requests=0; missed=0; skipped=0; failed=0; slew_rejections=0
    def events_until(end):
        nonlocal pending,slew_rejections
        if pending is not None and pending['event_ns']<=end:
            if pending['expired']:
                command=np.zeros(4); fallback=True
            else:
                command=np.array(pending['decision']['command']); fallback=pending['decision']['fallback_reason'] is not None
                # A stale result may not bypass a changed applied-command slew bound.
                if not fallback and np.any(abs(command-plant.u)>.04000001):
                    command=np.zeros(4); fallback=True; slew_rejections+=1
            unchanged=bool(np.array_equal(command,plant.u) and fallback==plant.fallback)
            # An identical command/mode has no plant event. Splitting an explicit FV step
            # here introduces latency-dependent integration/cost error even for all-zero
            # shutdown trajectories. Keep the canonical 10ms grid for these no-op events.
            if not unchanged:
                plant.advance_to(pending['event_ns']); plant.apply(command,fallback)
            trace.append({'event':'deadline_fallback' if pending['expired'] else 'apply',
                          'time_ns':pending['event_ns'],'request':pending['request'],
                          'fallback':fallback,'no_change':unchanged})
            pending=None
        plant.advance_to(end)
    for step in range(int(round(plant.duration_s*10))):
        now=step*100_000_000; events_until(now)
        if mode=='delayed' and now<busy_until:
            skipped+=1; trace.append({'event':'busy_skip','time_ns':now,'ready_ns':busy_until})
            continue
        try:
            decision=engine.decide(lambda:plant.raw().snapshot_ns(now))
        except (ValueError,RuntimeError) as exc:
            # Runtime failures remain in episode denominators and apply explicit shutdown.
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
            failed+=1; decision={'choice':-1,'command':[0.]*4,'fallback_reason':'inference_error','tokens':0,
                                'error':str(exc),'e2e_ms':0.,'gpu_compute_ms':0.}
            require(failed<3,'Repeated inference failures; episode cannot certify control quality')
        if mode=='zero': delay=0.; trace_id=None; token_gap=None
        else: delay,trace_id,token_gap=choose_latency(latency_trace,decision,requests,strict_latency)
        ready=now+int(np.ceil(delay*1e6)); deadline=now+int(round(deadline_ms*1e6)); expired=ready>deadline
        if expired: missed+=1
        event_ns=min(ready,deadline)
        pending={'event_ns':event_ns,'expired':expired,'decision':decision,'request':requests}
        busy_until=ready
        trace.append({'event':'request','request':requests,'snapshot_ns':now,'ready_ns':ready,'deadline_ns':deadline,
                      'expired':expired,'trace_request':trace_id,'trace_token_gap':token_gap,'delay_ms':delay,**decision})
        requests+=1
        events_until(min((step+1)*100_000_000,plant.end_ns))
        if step%100==0: print(f'SIL {engine.method} {mode} {step}/{int(plant.duration_s*10)}',flush=True)
        if stop and stop.signal: raise InterruptedError('SIL interrupted; current episode restarts, completed episodes retained')
    events_until(plant.end_ns)
    return {**plant.metrics(),'requests':requests,'missed_deadlines':missed,'busy_skipped_ticks':skipped,
            'inference_failures':failed,'mode':mode,'deadline_ms':deadline_ms,'delayed_slew_rejections':slew_rejections,
            'filter_interventions':sum(bool(r.get('filter_intervention')) for r in trace),
            'filter_intervention_fraction':sum(bool(r.get('filter_intervention')) for r in trace)/max(1,requests)},trace
