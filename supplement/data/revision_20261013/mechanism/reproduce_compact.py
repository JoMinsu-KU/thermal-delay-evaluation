#!/usr/bin/env python3
"""Recompute event/episode/response metrics from compact curves and commands.

Requires Python and NumPy only. No model, simulator or raw archive is needed.
python reproduce_compact.py --output reproduced
"""
from pathlib import Path
import argparse,csv,gzip,hashlib
import numpy as np
import analyze_saved_logs as a

def load_groups(path):
    groups={}
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):groups.setdefault(r.pop('key'),[]).append(r)
    return groups

def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,digest in a.read_json(source/'COMPACT_SHA256SUMS.json').items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name
    truth=load_groups(source/'truth_curves.csv.gz');commands=load_groups(source/'fv_application_records.csv.gz')
    metadata=a.read_csv(source/'case_metadata.csv');physical=[];episodes=[];events=[];checks=0
    assert len(truth)==len(metadata)==552 and len(commands)==336
    for r in metadata:
        key=r['key'];meta={k:r[k] for k in ['key','episode','method','mode','sequence','fallback_policy']}
        pm,t,y,goal=a.physical_metrics(truth[key])
        physical.append({**meta,**pm,'episode_cost':float(r['episode_cost']),'spatial_field_rmse_k':float(r['spatial_field_rmse_k'])})
        if key not in commands:continue
        apps=[]
        for c in commands[key]:
            apps.append({'event':c['event'],'time_ns':int(c['time_ns']),'request':int(c['request']),
              'fallback':bool(int(c['fallback'])),'no_change':bool(int(c['no_change'])),
              'command':[float(c[f'u{j}']) for j in range(4)]})
        ev,clocks,n=a.event_analysis(apps,t,y,goal);checks+=n
        assert len(apps)==int(r['requests'])
        for e in ev:
            assert np.allclose([e[f'fallback_u{j}'] for j in range(4)],0 if r['fallback_policy']=='zero' else [e[f'pre_u{j}'] for j in range(4)],atol=a.EPS,rtol=0)
            events.append({**meta,**e})
        nz=[e for e in ev if e['nonzero_precommand']];complete=[e for e in nz if e['restored']]
        episodes.append({**meta,'episode_cost':float(r['episode_cost']),'requests':int(r['requests']),
          'deadline_events':len(ev),'nonzero_precommand_events':len(nz),'restored_nonzero_events':len(complete),
          'censored_nonzero_events':len(nz)-len(complete),
          'mean_precommand':float(np.mean([e['pre_command_mean'] for e in ev])) if ev else '',
          'mean_precommand_max':float(np.mean([e['pre_command_max'] for e in ev])) if ev else '',
          'mean_error_at_expiration_k':float(np.mean([e['mean_error_at_expiration_k'] for e in ev])) if ev else '',
          'mean_coordinate_moves_lower_bound':float(np.mean([e['positive_coordinate_moves_from_zero_lower_bound'] for e in nz])) if nz else '',
          'conditional_mean_restoration_s':float(np.mean([e['restoration_time_s'] for e in complete])) if complete else '',
          'conditional_median_restoration_s':float(np.median([e['restoration_time_s'] for e in complete])) if complete else '',
          'first_deadline_s':ev[0]['time_s'] if ev else '',
          'mean_deadline_s':float(np.mean([e['time_s'] for e in ev])) if ev else '',
          'request_schedule_sha256':r['request_schedule_sha256']})
    a.write_csv(out/'physical_episode_metrics.csv',physical)
    a.write_csv(out/'deadline_episode_metrics.csv',episodes)
    a.write_csv(out/'deadline_events.csv',events)
    assert checks==196752 and len(events)==2256
    a.summarize(out)
    print('Reconstructed 552 curves, 336 command traces, and 2,256 expiration records.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'reproduced')
    args=p.parse_args();main(args.input,args.output)
