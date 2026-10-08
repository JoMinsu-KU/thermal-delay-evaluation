"""Independent FP64 cell-centred finite-volume teacher; no learned dynamics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, zoom

from .common import require

PROBES = np.array([[.125,.125],[.125,.375],[.375,.125],[.375,.375],
                   [.25,.125],[.25,.375],[.125,.25],[.375,.25]])
CANDIDATES = ['hold'] + [f'h{j}_{sign}' for j in range(4) for sign in ['up','down']]


def basis(n):
    require(n % 2 == 0, 'Even grid required')
    b = np.zeros((4,n,n), dtype=np.float64)
    b[0,:n//2,:n//2] = 1; b[1,:n//2,n//2:] = 1
    b[2,n//2:,:n//2] = 1; b[3,n//2:,n//2:] = 1
    return b  # array axes y,x; order low-y/low-x, low-y/high-x, high-y/low-x, high-y/high-x


def laplacian(t):
    dx = .5 / t.shape[-1]
    padded = np.pad(t, [(0,0)]*(t.ndim-2)+[(1,1),(1,1)], mode='edge')
    return (padded[...,2:,1:-1]+padded[...,:-2,1:-1]+padded[...,1:-1,2:]+
            padded[...,1:-1,:-2]-4*t)/dx**2


def restrict(t, n):
    old=t.shape[-1]; require(old % n == 0, 'Area restriction must have integer ratio')
    factor=old//n
    return t.reshape(*t.shape[:-2],n,factor,n,factor).mean(axis=(-3,-1))


def resize(t, n):
    return zoom(t, [1]*(t.ndim-2)+[n/t.shape[-2],n/t.shape[-1]], order=1, mode='nearest', grid_mode=True)


def probe_values(t):
    n=t.shape[-1]
    return map_coordinates(t, np.stack([PROBES[:,1]/.5*n-.5,PROBES[:,0]/.5*n-.5]), order=1,mode='nearest')


def actuator(p,u,dt,tau):
    z=-dt/np.asarray(tau)
    return u+(p-u)*np.exp(z), u+(p-u)*(-np.expm1(z))*np.asarray(tau)/dt


def advance(t,p,u,passport,duration=.1,max_dt=.01):
    """Does not mutate input state. Batch leading dimensions are supported."""
    t=np.array(t,dtype=np.float64,copy=True); p=np.array(p,dtype=np.float64,copy=True)
    u=np.asarray(u,dtype=np.float64); c,k,h,ambient,qmax=passport[:5]; tau=np.asarray(passport[5:])
    require(c>0 and k>=0 and h>=0 and np.all(tau>0) and duration>0, 'Invalid physical parameters')
    stable=.4/(4*(k/c)/(.5/t.shape[-1])**2+h/c) if k or h else max_dt
    steps=max(1,int(np.ceil(duration/min(max_dt,stable)))); dt=duration/steps
    b=basis(t.shape[-1])
    for _ in range(steps):
        pn,pa=actuator(p,u,dt,tau)
        q=qmax*np.einsum('...j,jyx->...yx',pa,b)
        t += dt/c*(k*laplacian(t)+q-h*(t-ambient)); p=pn
    require(np.isfinite(t).all(), 'Non-finite FV state')
    return t,p


def branch(t,p,commands,passport,grid=64,max_dt=.01):
    """Pure branch: same initial field and actuator for every command sequence."""
    commands=np.asarray(commands,dtype=np.float64)
    require(commands.ndim==3 and commands.shape[1:]==(20,4), 'Expected [K,20,4] commands')
    start=resize(t,grid) if t.shape[-1]!=grid else t
    state=np.repeat(start[None],len(commands),axis=0)
    actuator_state=np.repeat(np.asarray(p)[None],len(commands),axis=0)
    fields=[]
    for i in range(20):
        state,actuator_state=advance(state,actuator_state,commands[:,i],passport,max_dt=max_dt)
        fields.append(state.copy())
    return np.stack(fields,axis=1)


def candidates(previous):
    previous=np.asarray(previous,dtype=np.float64)
    require(previous.shape==(4,) and np.all((previous>=0)&(previous<=1)), 'Invalid applied command')
    values=np.repeat(previous[None],9,axis=0)
    for j in range(4):
        values[1+2*j,j]+= .04; values[2+2*j,j]-= .04
    values=np.clip(values,0,1)
    valid=np.ones(9,dtype=bool)
    for i in range(1,9):
        valid[i]=not any(np.allclose(values[i],values[j],atol=1e-10,rtol=0) for j in range(i) if valid[j])
    return np.repeat(values[:,None,:],20,axis=1),valid


def costs(fields,commands,previous,goal):
    e=((fields-goal)/100.)**2
    delta=np.diff(np.concatenate([np.broadcast_to(previous,(len(commands),1,4)),commands],axis=1),axis=1)
    return e.mean(axis=(1,2,3))+.5*e[:,-1].mean(axis=(1,2))+.002*(commands**2).mean(axis=(1,2))+.02*(delta**2).mean(axis=(1,2))


def order_accuracy(predicted,reference,valid,gap):
    i,j=np.triu_indices(len(reference),1)
    mask=valid[i]&valid[j]&(np.abs(reference[i]-reference[j])>gap)
    correct=((predicted[i]-predicted[j])*(reference[i]-reference[j])>0)&mask
    return int(correct.sum()),int(mask.sum())


def soft_targets(cost,eligible,temperature):
    q=np.zeros_like(cost,dtype=np.float64)
    if not eligible.any():
        return q
    values=-(cost[eligible]-cost[eligible].min())/temperature
    q[eligible]=np.exp(values-values.max()); q/=q.sum()
    return q


class PIController:
    def __init__(self):
        self.integral=np.zeros(4)

    def command(self,temperatures,goal,previous,dt=.1):
        error=goal-np.asarray(temperatures)
        desired=.03*error+.002*self.integral
        output=np.clip(np.clip(desired,np.asarray(previous)-.4*dt,np.asarray(previous)+.4*dt),0,1)
        # Conditional integration handles both saturation and rate limiting.
        integrate=((desired-output)*error<=0)
        self.integral += integrate*error*dt
        return output
