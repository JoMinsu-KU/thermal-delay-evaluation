#!/usr/bin/env python3
"""Offline evidence audit and paired summaries; no research model is loaded."""
from __future__ import annotations
import argparse
import csv
import gzip
import math
from pathlib import Path
import numpy as np
from validation_io import completed_case, read_json, read_trace, require, sha256, write_json



def csv_file(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def truth(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        return [{k:int(v) if k=='time_ns' else float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def command_integrals(trace, duration):
    last=0.; u=np.zeros(4); fallback=True; effort=seconds=delta=0.; effective=0
    for row in trace:
        if row['event'] not in ('apply','deadline_fallback'):continue
        t=row['time_ns']/1e9
        require(last<=t<=duration,'Nonmonotone command trace')
        effort+=float(np.mean(u*u))*(t-last);seconds+=fallback*(t-last)
        v=np.asarray(row['command'],dtype=np.float64)
        require(v.shape==(4,) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all(),'Invalid command')
        same=bool(np.array_equal(u,v) and fallback==row['fallback'])
        require(same==row['no_change'],'No-op flag is inconsistent')
        require(row['fallback'] or np.max(np.abs(v-u))<=.04000001,'Applied nonfallback slew violation')
        if not same:
            delta+=float(np.mean((v-u)**2));effective+=1
        u=v;fallback=row['fallback'];last=t
    effort+=float(np.mean(u*u))*(duration-last);seconds+=fallback*(duration-last)
    return {'action_effort':effort/duration,'fallback_s':seconds,
            'action_delta_mean_square':delta/round(duration*10),'effective_applications':effective}


def audit_case(folder, metrics, trace, rows):
    duration=metrics['duration_s']; end=round(duration*1e9)
    require(metrics['completed'],'Incomplete metric marked sealed')
    require(rows and rows[0]['time_ns']==0 and rows[-1]['time_ns']==end,'Incomplete truth log')
    commands=command_integrals(trace,duration)
    with np.load(folder/'quadrature.npz',allow_pickle=False) as q:
        dt=q['dt_ns'].astype(np.int64); ends=q['end_ns']
        require(len(dt)>0 and (dt>0).all() and np.array_equal(np.cumsum(dt),ends) and ends[-1]==end,'Quadrature clock coverage changed')
        for key in ('mse_k2','command_mse','peak_c'):
            require(np.isfinite(q[key]).all(),f'Nonfinite {key}')
        require((q['mse_k2']>=0).all(),'Negative mean squared error')
        seconds=dt/1e9
        tracking=float(np.sum(q['mse_k2']*seconds)/duration)
        peak=max(rows[0]['max_temperature_c'],float(q['peak_c'].max()))
        qeffort=float(np.sum(q['command_mse']*seconds)/duration)
        qfallback=float(np.sum(q['fallback']*seconds))
        violation=float(np.sum((q['peak_c']>120)*seconds))
    field=np.load(folder/'final_T64.npy',allow_pickle=False)
    require(np.isfinite(field).all(),'Nonfinite final field')
    terminal=float(np.mean(((field-rows[0]['goal_c'])/100)**2))
    cost=tracking/10000+.5*terminal+.002*commands['action_effort']+.02*commands['action_delta_mean_square']
    expected={'episode_cost':cost,'tracking_rmse_k':math.sqrt(tracking),
              'max_temperature_c':peak,'violation_s':violation,
              **{k:v for k,v in commands.items() if k!='effective_applications'}}
    errors=[]
    for key,value in expected.items():
        if not math.isclose(value,metrics[key],abs_tol=1e-10,rel_tol=1e-8):
            errors.append({'metric':key,'reconstructed':value,'saved':metrics[key]})
    for key,a,b in (('quadrature_effort',qeffort,commands['action_effort']),('quadrature_fallback',qfallback,commands['fallback_s'])):
        if not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-8):errors.append({'metric':key,'quadrature':a,'trace':b})
    if 'requests' in metrics:
        requests=[r for r in trace if r['event']=='request']
        countchecks={'requests':len(requests),'missed_deadlines':sum(r['expired'] for r in requests),
                     'busy_skipped_ticks':sum(r['event']=='busy_skip' for r in trace),
                     'filter_interventions':sum(bool(r.get('filter_intervention')) for r in requests)}
        for key,value in countchecks.items():
            if metrics[key]!=value:errors.append({'metric':key,'reconstructed':value,'saved':metrics[key]})
        ticks=[r['snapshot_ns'] for r in requests]+[r['time_ns'] for r in trace if r['event']=='busy_skip']
        if sorted(ticks)!=list(range(0,end,100_000_000)):errors.append({'request_tick_coverage':'incomplete or duplicated'})
    return {'passed':not errors,'cost_abs_error':abs(cost-metrics['episode_cost']),'errors':errors}


