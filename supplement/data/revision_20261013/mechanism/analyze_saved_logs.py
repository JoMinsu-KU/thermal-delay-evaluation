#!/usr/bin/env python3
"""Read sealed saved logs or reproduce summaries from compact derived CSVs.

No model, simulator, training, or device interface is imported. Extraction:
python analyze_saved_logs.py --archive RETURNED_ROOT --validation-tools TOOLS
Summary reproduction: python analyze_saved_logs.py --summarize
See DEFINITIONS.md for definitions fixed before this summary computation.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, importlib.util, json
import numpy as np

HERE = Path(__file__).resolve().parent
EPS = 1e-8

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(p, rows):
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def read_csv(p):
    with Path(p).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def integration(x, t):
    return float(np.sum((x[:-1]+x[1:])*.5*np.diff(t)))

def physical_metrics(rows):
    t=np.array([int(r['time_ns']) for r in rows])/1e9
    y=np.array([float(r['mean_temperature_c']) for r in rows])
    goals=np.array([float(r['goal_c']) for r in rows])
    assert len(t)==601 and np.allclose(t,np.arange(601)*.1,atol=1e-12,rtol=0)
    assert np.unique(goals).size==1
    err=y-goals; inside=np.abs(err)<=1+EPS
    sustained=np.logical_and.accumulate(inside[::-1])[::-1]
    settled=bool(sustained.any())
    i50=t>=50; i10=t<=10
    total=integration(err*err,t)
    return {
      'goal_c':float(goals[0]), 'initial_mean_c':float(y[0]),
      'initial_absolute_error_k':float(abs(err[0])),
      'final_10s_mean_absolute_error_k':integration(abs(err[i50]),t[i50])/10,
      'sampled_mean_overshoot_k':float(max(0,err.max())),
      'settled_within_1k':int(settled),
      'settling_time_s':float(t[np.flatnonzero(sustained)[0]]) if settled else '',
      'settling_censor_time_s':'' if settled else 60,
      'first_10s_squared_mean_error_fraction':integration(err[i10]**2,t[i10])/total if total else 0,
    },t,y,goals[0]

def event_analysis(apps,t,y,goal):
    previous=np.zeros(4); events=[]; clocks=[]; nchecked=0
    for i,a in enumerate(apps):
        u=np.array(a['command'],dtype=float)
        if not a['fallback']:
            delta=np.abs(u-previous)
            assert np.count_nonzero(delta>EPS)<=1, 'Non-coordinate candidate move'
            assert delta.sum()<=.04000001, 'Candidate step above 0.04'
            nchecked+=1
        if a['event']=='deadline_fallback':
            ns=int(a['time_ns']); sec=ns/1e9
            next_event=next((v for v in apps[i+1:] if v['event']=='deadline_fallback'),None)
            stop_ns=next_event['time_ns'] if next_event else 60_000_000_000
            subsequent=[v for v in apps[i+1:] if v['time_ns']<stop_ns]
            restored=bool(np.all(u>=previous-EPS))
            recovery_ns=ns if restored else None
            move_count=0
            for v in ([] if restored else subsequent):
                if not v['fallback']:move_count+=1
                if recovery_ns is None and np.all(np.array(v['command'])>=previous-EPS):
                    recovery_ns=int(v['time_ns']); break
            first=next((v for v in subsequent if not v['fallback']),None)
            nonzero=bool(np.any(previous>EPS))
            bounds=int(np.maximum(0,np.ceil((previous-EPS)/.04)).sum())
            events.append({
              'event_index':len(events),'request':a['request'],'time_s':sec,
              **{f'pre_u{j}':float(previous[j]) for j in range(4)},
              **{f'fallback_u{j}':float(u[j]) for j in range(4)},
              'pre_command_mean':float(previous.mean()),'pre_command_max':float(previous.max()),
              'nonzero_precommand':int(nonzero),
              'positive_coordinate_moves_from_zero_lower_bound':bounds,
              'equivalent_nominal_slots_s':bounds*.1,
              'mean_temperature_at_expiration_c':float(np.interp(sec,t,y)),
              'mean_error_at_expiration_k':float(np.interp(sec,t,y)-goal),
              'followup_boundary_s':stop_ns/1e9,
              'followup_boundary':'next_deadline' if next_event else 'horizon',
              'restored':int(recovery_ns is not None),
              'restoration_time_s':(recovery_ns-ns)/1e9 if recovery_ns is not None else '',
              'restoration_censored':int(recovery_ns is None),
              'restoration_censor_time_s':(stop_ns-ns)/1e9 if recovery_ns is None else '',
              'ordinary_applications_to_restoration':move_count if recovery_ns is not None else '',
              'first_following_ordinary_application_delay_s':(first['time_ns']-ns)/1e9 if first else '',
            })
            clocks.append(ns)
        previous=u
    return events,clocks,nchecked

def extract(archive, validation_tools, out):
    spec=importlib.util.spec_from_file_location('revision_validation_io',validation_tools/'validation_io.py')
    vio=importlib.util.module_from_spec(spec);spec.loader.exec_module(vio)
    protocol=read_json(archive/'protocol.json'); proto_hash=vio.sha256(archive/'protocol.json')
    events=[]; physical=[]; episode=[]; receipts=[]; clock_groups={};transition_checks=0
    for job in protocol['design']['jobs']:
        keep=job['phase']=='baseline' or (job['method']=='fv_mpc' and job['mode']=='delayed')
        if not keep:continue
        for case in job['cases']:
            folders=[p for p in (archive/'cases'/case['key']).glob('attempt_*') if (p/'COMPLETE').is_file()]
            assert len(folders)==1
            p=folders[0];manifest=vio.verify_seal(p)
            assert manifest['metadata']['protocol_sha256']==proto_hash
            condition={**{k:v for k,v in job.items() if k!='cases'},**case}
            assert read_json(p/'condition.json')==condition
            with gzip.open(p/'truth.csv.gz','rt',encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            pm,t,y,goal=physical_metrics(rows)
            met=read_json(p/'metrics.json')
            meta={k:condition[k] for k in ['key','episode','method','mode','sequence','fallback_policy']}
            physical.append({**meta,**pm,'episode_cost':met['episode_cost'],'spatial_field_rmse_k':met['tracking_rmse_k']})
            receipts.append({'case':case['key'],'manifest_sha256':vio.sha256(p/'manifest.json'),'trace_sha256':manifest['files']['trace.jsonl.gz'],'truth_sha256':manifest['files']['truth.csv.gz'],'seal_verified':True})
            if job['method']=='fv_mpc' and job['mode']=='delayed':
                trace=vio.read_trace(p/'trace.jsonl.gz')
                apps=[v for v in trace if v['event'] in ('apply','deadline_fallback')]
                ev,clocks,checks=event_analysis(apps,t,y,goal);transition_checks+=checks
                requests=[v for v in trace if v['event']=='request']
                clocks_req=[{k:v[k] for k in ['request','snapshot_ns','ready_ns','compute_ready_ns','deadline_ns','expired']} for v in requests]
                schedule_hash=hashlib.sha256(json.dumps(clocks_req,sort_keys=True).encode()).hexdigest()
                group=clock_groups.setdefault(job['sequence'],{'deadline_schedule':clocks,'request_schedules':set(),'cases':0})
                assert group['deadline_schedule']==clocks
                group['request_schedules'].add(schedule_hash);group['cases']+=1
                for e in ev:
                    assert np.allclose([e[f'fallback_u{j}'] for j in range(4)],0 if job['fallback_policy']=='zero' else [e[f'pre_u{j}'] for j in range(4)],atol=EPS,rtol=0)
                    events.append({**meta,**e})
                nonzero=[e for e in ev if e['nonzero_precommand']]
                complete=[e for e in nonzero if e['restored']]
                episode.append({**meta,'episode_cost':met['episode_cost'],'requests':len(requests),
                 'deadline_events':len(ev),'nonzero_precommand_events':len(nonzero),
                 'restored_nonzero_events':len(complete),
                 'censored_nonzero_events':len(nonzero)-len(complete),
                 'mean_precommand':float(np.mean([e['pre_command_mean'] for e in ev])) if ev else '',
                 'mean_precommand_max':float(np.mean([e['pre_command_max'] for e in ev])) if ev else '',
                 'mean_error_at_expiration_k':float(np.mean([e['mean_error_at_expiration_k'] for e in ev])) if ev else '',
                 'mean_coordinate_moves_lower_bound':float(np.mean([e['positive_coordinate_moves_from_zero_lower_bound'] for e in nonzero])) if nonzero else '',
                 'conditional_mean_restoration_s':float(np.mean([e['restoration_time_s'] for e in complete])) if complete else '',
                 'conditional_median_restoration_s':float(np.median([e['restoration_time_s'] for e in complete])) if complete else '',
                 'first_deadline_s':ev[0]['time_s'] if ev else '',
                 'mean_deadline_s':float(np.mean([e['time_s'] for e in ev])) if ev else '',
                 'request_schedule_sha256':schedule_hash,
                })
    assert len(receipts)==552 and len(episode)==336
    assert all(v['cases']==48 and len(v['request_schedules'])==1 for v in clock_groups.values())
    write_csv(out/'deadline_events.csv',events)
    write_csv(out/'deadline_episode_metrics.csv',episode)
    write_csv(out/'physical_episode_metrics.csv',physical)
    dump(out/'seal_receipts.json',receipts)
    dump(out/'extraction_audit.json',{'verified_cases':len(receipts),'fv_cases':len(episode),'deadline_records':len(events),
      'ordinary_one_coordinate_transitions_checked':transition_checks,
      'all_48_FV_cases_share_identical_request_schedule_within_setting':True,
      'settings':{k:{'deadline_s':[t/1e9 for t in v['deadline_schedule']],'request_schedule_sha256':next(iter(v['request_schedules'])),'cases':v['cases']} for k,v in clock_groups.items()},
      'no_simulation_inference_training':True,'sampling_unit':'24 episodes, reused over seven fixed timing settings',
      'definitions_sha256':hashlib.sha256((HERE/'DEFINITIONS.md').read_bytes()).hexdigest()})

def f(r,k):return float(r[k])
def avg(rows,k):
    a=[f(r,k) for r in rows if r[k]!=''];return float(np.mean(a)) if a else ''

def summarize(out):
    physical=read_csv(out/'physical_episode_metrics.csv')
    episodes=read_csv(out/'deadline_episode_metrics.csv')
    events=read_csv(out/'deadline_events.csv')
    base={(r['method'],r['episode']):f(r,'episode_cost') for r in physical if r['mode']=='zero'}
    phys=[]
    keys=sorted({(r['method'],r['mode'],r['fallback_policy']) for r in physical if r['sequence']=='legacy'})
    for method,mode,policy in keys:
        rows=[r for r in physical if r['sequence']=='legacy' and (r['method'],r['mode'],r['fallback_policy'])==(method,mode,policy)]
        assert len(rows)==24
        phys.append({'method':method,'mode':mode,'fallback_policy':policy,'n_episodes':24,
          'mean_cost':avg(rows,'episode_cost'),'mean_spatial_field_rmse_k':avg(rows,'spatial_field_rmse_k'),
          'mean_final_10s_absolute_mean_temperature_error_k':avg(rows,'final_10s_mean_absolute_error_k'),
          'mean_sampled_mean_overshoot_k':avg(rows,'sampled_mean_overshoot_k'),
          'max_sampled_mean_overshoot_k':max(f(r,'sampled_mean_overshoot_k') for r in rows),
          'settled_episodes':sum(int(r['settled_within_1k']) for r in rows),
          'censored_episodes':sum(1-int(r['settled_within_1k']) for r in rows),
          'conditional_mean_settling_s':avg(rows,'settling_time_s'),
          'mean_first_10s_squared_mean_error_fraction':avg(rows,'first_10s_squared_mean_error_fraction'),
          'mean_initial_absolute_error_k':avg(rows,'initial_absolute_error_k')})
    write_csv(out/'physical_summary.csv',phys)
    sequence=[]
    for seq in sorted({r['sequence'] for r in episodes}):
        z=[r for r in episodes if r['sequence']==seq and r['fallback_policy']=='zero']
        h=[r for r in episodes if r['sequence']==seq and r['fallback_policy']=='hold']
        ev=[r for r in events if r['sequence']==seq and r['fallback_policy']=='zero']
        nz=[r for r in ev if int(r['nonzero_precommand'])]
        complete=[r for r in nz if int(r['restored'])]
        assert len(z)==len(h)==24
        b=float(np.mean([base['fv_mpc',r['episode']] for r in z]))
        sequence.append({'sequence':seq,'unique_episodes':24,'deadline_events_per_episode':int(z[0]['deadline_events']),
         'first_deadline_s':f(z[0],'first_deadline_s'),'mean_deadline_s':f(z[0],'mean_deadline_s'),
         'mean_zero_cost':avg(z,'episode_cost'),'mean_hold_cost':avg(h,'episode_cost'),
         'zero_increase_pct':100*(avg(z,'episode_cost')/b-1),'hold_increase_pct':100*(avg(h,'episode_cost')/b-1),
         'zero_minus_hold_cost':avg(z,'episode_cost')-avg(h,'episode_cost'),
         'mean_zero_precommand':avg(z,'mean_precommand'),'mean_hold_precommand':avg(h,'mean_precommand'),
         'mean_zero_temperature_error_at_expiration_k':avg(z,'mean_error_at_expiration_k'),
         'mean_hold_temperature_error_at_expiration_k':avg(h,'mean_error_at_expiration_k'),
         'nonzero_reset_events':len(nz),'restored_nonzero_events':len(complete),
         'censored_nonzero_events':len(nz)-len(complete),
         'mean_episode_conditional_restoration_s':avg(z,'conditional_mean_restoration_s'),
         'event_descriptive_min_completed_restoration_s':min(f(r,'restoration_time_s') for r in complete) if complete else '',
         'event_descriptive_median_completed_restoration_s':float(np.median([f(r,'restoration_time_s') for r in complete])) if complete else '',
         'event_descriptive_max_completed_restoration_s':max(f(r,'restoration_time_s') for r in complete) if complete else '',
         'event_descriptive_min_coordinate_moves':min(int(r['positive_coordinate_moves_from_zero_lower_bound']) for r in nz) if nz else '',
         'event_descriptive_max_coordinate_moves':max(int(r['positive_coordinate_moves_from_zero_lower_bound']) for r in nz) if nz else '',
        })
    write_csv(out/'setting_summary.csv',sequence)
    z=[r for r in events if r['fallback_policy']=='zero' and int(r['nonzero_precommand'])]
    h=[r for r in events if r['fallback_policy']=='hold']
    observed=[r for r in z if int(r['restored'])]
    dump(out/'summary.json',{'analysis':'Post hoc saved-log analysis; no new runs',
      'fixed_settings':7,'unique_episodes':24,'fv_zero_hold_cases':336,
      'nonzero_zero_reset_events':len(z),'completed_nonzero_restorations':len(observed),
      'censored_nonzero_restorations':len(z)-len(observed),
      'all_hold_events_restored_immediately':all(f(r,'restoration_time_s')==0 for r in h),
      'event_descriptive_completed_restoration_range_s':[min(f(r,'restoration_time_s') for r in observed),max(f(r,'restoration_time_s') for r in observed)],
      'event_descriptive_coordinate_moves_range':[min(int(r['positive_coordinate_moves_from_zero_lower_bound']) for r in z),max(int(r['positive_coordinate_moves_from_zero_lower_bound']) for r in z)],
      'interpretation':'Events are dependent and censored; summaries do not estimate a new-timing distribution or infer a causal effect of expiration time.',
      'physical_response_conditions':len(phys)})
    print(json.dumps(read_json(out/'summary.json'),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path);parser.add_argument('--validation-tools',type=Path)
    parser.add_argument('--output',type=Path,default=HERE);parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.summarize:
        parser.error('--archive and --validation-tools required for extraction') if args.archive is None or args.validation_tools is None else None
        extract(args.archive,args.validation_tools,args.output)
    summarize(args.output)
