"""FP32 multi-horizon FNO/PINO with complex64 FFT and zero-flux FV residuals."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import require


def heater_basis(n,device,dtype=torch.float32):
    from .physics import basis
    return torch.as_tensor(basis(n),device=device,dtype=dtype)


def laplace(t):
    shape=t.shape
    flat=t.reshape(-1,1,*shape[-2:]); p=F.pad(flat,(1,1,1,1),mode='replicate')
    out=(p[...,2:,1:-1]+p[...,:-2,1:-1]+p[...,1:-1,2:]+p[...,1:-1,:-2]-4*flat)/(.5/shape[-1])**2
    return out.reshape(shape)


def heat_sequence(p0,u,passport,dt=.1):
    p=p0; sources=[]; b=heater_basis(32,u.device,u.dtype); tau=passport[:,5:]
    for i in range(20):
        pa=u[:,i]+(p-u[:,i])*(-torch.expm1(-dt/tau))*tau/dt
        p=u[:,i]+(p-u[:,i])*torch.exp(-dt/tau)
        sources.append(passport[:,4,None,None]*torch.einsum('bj,jyx->byx',pa,b))
    return torch.stack(sources,1)


def project(raw,initial,heat,passport,dt=.1):
    c=passport[:,0,None,None]; h=passport[:,2,None,None]; ambient=passport[:,3,None,None]
    # Subtract a nearby reference before reduction to avoid summing thousands of absolute °C values.
    def mean(value):
        reference=value[...,:1,:1]
        return reference+(value-reference).mean((-2,-1),keepdim=True)
    previous=mean(initial); out=[]
    for i in range(20):
        target=previous+dt*(heat[:,i].mean((-2,-1),keepdim=True)-h*(previous-ambient))/(c+dt*h/2)
        value=raw[:,i]+(target-mean(raw[:,i]))
        out.append(value); previous=mean(value)
    return torch.stack(out,1)


def residual(fields,initial,heat,passport,dt=.1):
    previous=torch.cat([initial[:,None],fields[:,:-1]],1)
    c=passport[:,0,None,None,None]; k=passport[:,1,None,None,None]
    h=passport[:,2,None,None,None]; ambient=passport[:,3,None,None,None]
    return c*(fields-previous)/dt-k*.5*(laplace(fields)+laplace(previous))-heat+h*((fields+previous)/2-ambient)


class SpectralConv(nn.Module):
    def __init__(self,width=32,modes=8):
        super().__init__(); self.modes=modes
        self.positive=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.complex64)/width)
        self.negative=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.complex64)/width)

    def forward(self,x):
        spectrum=torch.fft.rfft2(x.float(),norm='ortho'); m=self.modes
        out=torch.zeros_like(spectrum)
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',spectrum[:,:,:m,:m],self.positive)
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',spectrum[:,:,-m:,:m],self.negative)
        return torch.fft.irfft2(out,s=x.shape[-2:],norm='ortho')


class ThermalOperator(nn.Module):
    def __init__(self,variant='pino_balance'):
        super().__init__(); require(variant in ['fno_data','pino_soft','pino_balance','fno_balance'],'Unknown operator')
        self.variant=variant; self.lift=nn.Conv2d(14,32,1)
        self.spectral=nn.ModuleList([SpectralConv() for _ in range(3)])
        self.local=nn.ModuleList([nn.Conv2d(32,32,1) for _ in range(3)])
        self.condition=nn.Sequential(nn.Linear(90,64),nn.GELU(),nn.Linear(64,3*64))
        self.output=nn.Sequential(nn.Conv2d(32,64,1),nn.GELU(),nn.Conv2d(64,20,1))
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def inputs(self,field,passport,p0,u):
        b=len(field); n=32; require(field.shape==(b,n,n) and u.shape==(b,20,4),'Operator shape contract')
        basis=heater_basis(n,field.device)
        coords=(torch.arange(n,device=field.device,dtype=torch.float32)+.5)/n
        yy,xx=torch.meshgrid(coords,coords,indexing='ij')
        source=passport[:,4,None,None]*torch.einsum('bj,jyx->byx',p0,basis)/40000
        maps=[(field-25)/100]
        for i,scale in enumerate([12000.,.75,25.,25.]): maps.append((passport[:,i]/scale)[:,None,None].expand(-1,n,n))
        maps += [basis[j].expand(b,-1,-1) for j in range(4)]
        maps += [xx.expand(b,-1,-1),yy.expand(b,-1,-1),torch.ones_like(field),source,torch.full_like(field,.1)]
        condition=torch.cat([u.flatten(1),p0,passport[:,5:]/.1,passport[:,4:5]/40000,torch.ones((b,1),device=field.device)],1)
        return torch.stack(maps,1),condition

    def forward(self,field,passport,p0,u,return_raw=False):
        require(field.dtype==passport.dtype==p0.dtype==u.dtype==torch.float32,'PINO requires FP32')
        with torch.autocast(device_type=field.device.type,enabled=False):
            spatial,condition=self.inputs(field,passport,p0,u)
            x=F.pad(self.lift(spatial),(8,8,8,8),mode='reflect')
            film=self.condition(condition).reshape(-1,3,2,32,1,1)
            for i,(spectral,local) in enumerate(zip(self.spectral,self.local)):
                x=F.gelu((spectral(x)+local(x))*(1+film[:,i,0])+film[:,i,1])
            raw=field[:,None]+self.output(x)[...,8:-8,8:-8]*10.
            prediction=project(raw,field,heat_sequence(p0,u,passport),passport) if self.variant in ['pino_balance','fno_balance'] else raw
            return (prediction,raw) if return_raw else prediction


class BalanceOnly(nn.Module):
    """No trainable weights: preserve initial spatial deviations, evolve only the mean."""
    def forward(self,field,passport,p0,u,return_raw=False):
        raw=field[:,None].expand(-1,20,-1,-1)
        prediction=project(raw,field,heat_sequence(p0,u,passport),passport)
        return (prediction,raw) if return_raw else prediction


def field_loss(prediction,target):
    return F.smooth_l1_loss(prediction/10.,target/10.)


def physics_features(fields,initial,p0,u,passport,previous,goal,age):
    """16 features, physical units. Margins 13/14 stay zero; no teacher costs are inputs."""
    heat=heat_sequence(p0,u,passport); r=residual(fields,initial,heat,passport)
    error=((fields-goal[:,None,None,None])/100)**2
    delta=torch.diff(torch.cat([previous[:,None],u],1),dim=1)
    exceed=(fields-120).clamp_min(0)
    return torch.stack([error.mean((1,2,3)),error[:,-1].mean((1,2)),fields.amax((1,2,3)),
                        fields[:,-1].mean((1,2)),fields[:,-1].std((1,2),unbiased=False),
                        exceed.amax((1,2,3)),(exceed>0).float().mean((1,2,3)),
                        heat.mean((2,3)).sum(1)*.1*.25,(delta**2).mean((1,2)),
                        .4-delta.abs().amax((1,2))/.1,r.abs().mean((1,2,3)),r.abs().amax((1,2,3)),
                        r.mean((2,3)).abs().amax(1),torch.zeros_like(age),torch.zeros_like(age),age],1)


FEATURE_UNITS=['dimensionless','dimensionless','degC','degC','K','K','fraction','J','command_squared',
               'command_per_s','W/m2','W/m2','W/m2','K_fixed_zero','K_fixed_zero','s']
