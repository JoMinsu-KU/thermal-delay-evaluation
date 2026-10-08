"""Audit released logs and quadrature arrays; never execute experiment code (MIT)."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math
import numpy as np
from baseline_compare import baseline_compare
from reproduce import ROOT, METHODS, csvread, read, save, dump


def events(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-8)


def applied_terms(trace):
    u = np.zeros(4)
    fallback, last = True, 0
    energy, fallbacks, changes = [], [], []
    for e in trace:
        if e['event'] not in ('apply', 'deadline_fallback'):
            continue
        t = e['time_ns']
        v = np.asarray(e['command'])
        assert 0 <= last <= t <= 60_000_000_000 and v.shape == (4,)
        energy.append(float(u @ u / 4) * (t-last)/1e9)
        fallbacks.append(int(fallback) * (t-last)/1e9)
        noop = np.array_equal(u, v) and fallback == e['fallback']
        assert bool(noop) == e['no_change']
        if not noop:
            changes.append(float(np.mean((v-u)**2)))
        u, fallback, last = v, e['fallback'], t
    energy.append(float(u @ u / 4) * (60_000_000_000-last)/1e9)
    fallbacks.append(int(fallback) * (60_000_000_000-last)/1e9)
    return math.fsum(energy)/60, math.fsum(fallbacks), math.fsum(changes)/600


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', type=Path, required=True, help='Contains primary/ and numerical/ after extracting both release assets.')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/raw_audit')
    args = ap.parse_args()
    raw, out = args.raw_dir.resolve(), args.output.resolve()
    if out == raw or raw in out.parents or out == ROOT/'data' or (ROOT/'data') in out.parents:
        raise ValueError('Keep audit outputs separate from immutable input data.')
    out.mkdir(parents=True, exist_ok=True)
    index = csvread(ROOT/'data/numerical/case_index.csv')
    assert len(index) == 741
    costs, fixed, diagnostics = [], {}, {}
    baseline_count, initial_partial = 0, 0
    for i, row in enumerate(index):
        f = raw/row['path']
        m = read(f/'metrics.json')
        tr = events(f/'trace.jsonl.gz')
        truthname = 'truth.csv.gz' if row['batch'] == 'extended' else 'truth_100hz.csv.gz'
        with gzip.open(f/truthname, 'rt', encoding='utf-8', newline='') as z:
            truth = list(csv.DictReader(z))
        step = 100_000_000 if row['source'] == 'D' else 10_000_000
        assert [int(r['time_ns']) for r in truth] == list(range(0, 60_000_000_001, step))
        goal = float(truth[0]['goal_c'])
        effort, fallback, slew = applied_terms(tr)
        assert close(effort, m['action_effort']) and close(fallback, m['fallback_s'])
        assert close(slew, m['action_delta_mean_square'])
        final = np.load(f/'final_T64.npy', allow_pickle=False)
        assert final.shape == (64,64) and final.dtype == np.float64 and np.isfinite(final).all()
        if row['batch'] == 'extended':
            with np.load(f/'quadrature.npz', allow_pickle=False) as q:
                assert np.array_equal(np.cumsum(q['dt_ns'], dtype=np.int64), q['end_ns'])
                assert int(q['end_ns'][-1]) == 60_000_000_000
                dt = q['dt_ns']/1e9
                tracking = math.fsum((q['mse_k2']*dt).tolist())/60
                assert close(effort, math.fsum((q['command_mse']*dt).tolist())/60)
                assert close(fallback, math.fsum((q['fallback']*dt).tolist()))
                peak = max(float(truth[0]['max_temperature_c']), float(q['peak_c'].max()))
                assert close(peak, m['max_temperature_c'])
                assert close(float(dt[q['peak_c']>120].sum()), m['violation_s'])
            assert close(math.sqrt(tracking), m['tracking_rmse_k'])
        else:
            # This earlier batch did not save quadrature. Tracking is an input here.
            tracking = m['tracking_rmse_k']**2
            initial_partial += 1
        cost = tracking/10000 + .5*float(np.mean(((final-goal)/100)**2)) + .002*effort + .02*slew
        assert close(cost, m['episode_cost']), row['case']
        costs.append({'case': row['case'], 'batch': row['batch'], 'absolute_cost_error': abs(cost-m['episode_cost']),
                      'tracking_independently_reconstructed': row['batch']=='extended'})
        h = float(row['h_ms'])
        if row['source'] == 'D':
            req = [e for e in tr if e['event']=='request']
            skip = [e for e in tr if e['event']=='busy_skip']
            assert sorted([e['snapshot_ns'] for e in req]+[e['time_ns'] for e in skip]) == list(range(0,60_000_000_000,100_000_000))
            assert len(req)==m['requests'] and len(skip)==m['busy_skipped_ticks']
            assert sum(e['expired'] for e in req)==m['missed_deadlines']
            ident = (h, row['method'], row['mode'], row['episode'])
            diagnostics[ident] = (np.array([e['snapshot_ns'] for e in req]), np.array([e['command'] for e in req]),
                                 np.array([float(t['mean_temperature_c']) for t in truth]))
            if h == 10:
                orig = raw/'primary/D'/row['mode']/row['method']/row['episode']
                check = baseline_compare(m, tr, read(orig/'metrics.json'), events(orig/'control_trace.jsonl'))
                assert check['passed'], (row['case'], check['failures'][:1])
                baseline_count += 1
        elif row['mode'] == 'fixed':
            with np.load(f/'fields_1hz.npz', allow_pickle=False) as z:
                assert np.array_equal(z['time_ns'], np.arange(61,dtype=np.int64)*1_000_000_000)
                fixed[h, float(row['delay_ms'])] = (m['episode_cost'], z['temperature_c'].copy())
            if h == 10:
                old = read(raw/'primary/C'/f"shared_physics_{float(row['delay_ms']):.1f}.json")['metrics']
                for key in ('episode_cost','tracking_rmse_k','action_effort','action_delta_mean_square','fallback_s','max_temperature_c','violation_s'):
                    assert close(m[key], old[key])
                baseline_count += 1
        if (i+1)%100 == 0:
            print('Numerical cases audited:',i+1,flush=True)
    assert baseline_count==243 and initial_partial==9 and len(fixed)==12
    fixed_rows = []
    for old in csvread(ROOT/'data/numerical/fixed_replay.csv'):
        h1,h2,delay = map(float,[old['coarse_ms'],old['fine_ms'],old['delay_ms']])
        x,y = fixed[h1,delay],fixed[h2,delay]
        delta = y[1]-x[1]
        r={'delay_ms':delay,'coarse_ms':h1,'fine_ms':h2,'cost_relative_pct':(y[0]/x[0]-1)*100,
           'field_rmse_k_1hz':float(np.sqrt(np.mean(delta**2))),'field_max_abs_k_1hz':float(abs(delta).max()),
           'final_field_rmse_k':float(np.sqrt(np.mean(delta[-1]**2)))}
        assert all(close(v,float(old[k])) for k,v in r.items())
        fixed_rows.append(r)
    for old in csvread(ROOT/'data/numerical/episode_sensitivity.csv'):
        ident=(old['method'],old['mode'],old['episode'])
        x=diagnostics[(float(old['coarse_ms']),*ident)]; y=diagnostics[(float(old['fine_ms']),*ident)]
        assert np.array_equal(x[0],y[0])
        assert int(np.any(x[1]!=y[1],axis=1).sum())==int(old['command_disagreements'])
        assert close(float(np.sqrt(np.mean((x[2]-y[2])**2))),float(old['mean_temperature_rmse_k_10hz']))
    save(out,'cost_reconstruction.csv',costs)
    save(out,'fixed_field_comparisons.csv',fixed_rows)

    # Rebuild all 60 latency summaries and the consumed replay source mappings.
    timing_count, replay_count, quantiles = 0, 0, []
    expected = csvread(ROOT/'data/primary/latency_quantiles.csv')
    paths = ('normal','missing_observation','no_predicted_feasible_candidate')
    for method in METHODS:
        repeats = []
        for repeat in range(3):
            rows = []
            path = raw/'primary/latency'/f'{method}_{repeat}'/'requests.jsonl'
            with path.open(encoding='utf-8') as f:
                for i,line in enumerate(f):
                    journal = json.loads(line)
                    assert journal['index']==i
                    r = journal['row']
                    assert hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()==journal['sha256']
                    rows.append(r)
            assert len(rows)==12000
            timing_count+=len(rows)
            repeats.append(rows)
        pool=[r for triple in zip(*repeats) for r in triple]
        for tag,rows in [(str(i),r) for i,r in enumerate(repeats)]+[('pooled',pool)]:
            for p in paths:
                selected=[r for r in rows if r['path']==p]
                good=[r for r in selected if not r['error']]
                values=np.array([r['e2e_ms'] for r in good])
                r={'method':method,'path':p,'repeat':tag,'n':len(good),'mean_ms':float(values.mean()),
                   'median_ms':float(np.quantile(values,.5)),'p95_ms':float(np.quantile(values,.95)),
                   'p99_ms':float(np.quantile(values,.99)),'max_ms':float(values.max()),
                   'count_gt_100ms':int((values>100).sum()),'errors':len(selected)-len(good),
                   'distinct_inputs':len({x['input_id'] for x in good})}
                old=next(x for x in expected if x['method']==method and x['path']==p and x['repeat']==tag)
                assert all(close(v,float(old[k])) for k,v in r.items() if k not in ('method','path','repeat'))
                quantiles.append(r)
        cache={}
        normal=[r for r in read(ROOT/f'data/primary/delayed_{method}.json') if r['cohort']=='ID' and r['stratum']=='normal']
        for m in normal:
            tr=events(raw/'primary/D/delayed'/method/m['episode']/'control_trace.jsonl')
            apps={e['request']:e for e in tr if e['event'] in ('apply','deadline_fallback')}
            for e in [x for x in tr if x['event']=='request']:
                key=(e['fallback_reason'],e['tokens'])
                if key not in cache:
                    candidates=[r for r in pool if not r['error'] and r['fallback_reason']==key[0]]
                    distance=min(abs(r['tokens']-key[1]) for r in candidates)
                    cache[key]=[r for r in candidates if abs(r['tokens']-key[1])==distance]
                choices=cache[key]; original=choices[e['request']%len(choices)]
                assert original['request']==e['trace_request'] and close(original['e2e_ms'],e['delay_ms'])
                assert e['expired']==(e['ready_ns']>e['deadline_ns'])
                app=apps[e['request']]
                assert app['time_ns']==min(e['ready_ns'],e['deadline_ns'])
                if e['expired']:
                    assert app['event']=='deadline_fallback' and all(x==0 for x in app['command'])
                replay_count+=1
        print('Latency and replay mappings audited:',method,flush=True)
    assert timing_count==180000 and replay_count==71808 and len(quantiles)==60
    save(out,'latency_quantiles_recomputed.csv',quantiles)
    result={'passed':True,'extended_full_quadrature_reconstructions':732,
            'earlier_C_partial_cost_checks_using_reported_tracking':9,
            'baseline_reproductions_rechecked':baseline_count,
            'max_extended_cost_absolute_error':max(r['absolute_cost_error'] for r in costs if r['batch']=='extended'),
            'fixed_field_comparisons':len(fixed_rows),'D_episode_trace_comparisons':720,
            'latency_journal_rows':timing_count,'latency_summaries':len(quantiles),
            'normal_replay_request_mappings':replay_count,
            'research_simulation_inference_or_training_executed':False}
    dump(out/'audit_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
