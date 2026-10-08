"""Isolated-process, path-stratified latency for frozen L1 checkpoints."""
from __future__ import annotations

import argparse
from itertools import chain
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .common import (require, read_json, write_json, digest, seal, verify,
                     setup_torch, StopRequest)
from .control import DecisionEngine, MemoryEpisode, freeze_request, METHODS
from .evaluation import EvaluationData, Journal, quantiles
from .pilot import identity, system_counters, thermal_is_fresh

PATHS = ('normal', 'missing_observation', 'no_predicted_feasible_candidate')
REASONS = {'normal': None, 'missing_observation': 'cold_start_or_missing_observation',
           'no_predicted_feasible_candidate': 'no_predicted_feasible_candidate'}


def measured_path(result):
    for key, reason in REASONS.items():
        if result['fallback_reason'] == reason:
            return key
    raise ValueError(f'Unmeasured decision path: {result["fallback_reason"]}')


def make_inputs(root, run, config, output, device='cuda'):
    """Freeze raw buffers, not observations, tokens or predictions. Classify outside timing."""
    setup_torch(config, device)
    cfg = config['validation']; data = EvaluationData(run)
    engine = DecisionEngine(root, run, 'pino_mpc', device=device)
    groups = {key: [] for key in PATHS}; stop = StopRequest()
    try:
        points = [(e, int(s)) for e in data.entries for s in data.plan['anchor_steps']]
        take = np.linspace(0, len(points)-1, min(len(points), 2*cfg['normal_ring_size']), dtype=int)
        for i in take:
            entry, step = points[int(i)]; raw, _ = data.episode(entry)
            frozen = freeze_request(raw, step)
            result = engine.decide(lambda: frozen.snapshot_ns(step*100_000_000))
            key = measured_path(result)
            groups[key].append((frozen, step, entry, 'natural'))
            if stop.signal: raise InterruptedError('Latency input preparation interrupted')
        # Cold start has genuinely unavailable measurements, with unchanged receive/ready times.
        for entry in data.entries[:cfg['fallback_ring_size']]:
            raw, _ = data.episode(entry)
            frozen = freeze_request(raw, 0)
            require(measured_path(engine.decide(lambda: frozen.snapshot_ns(0))) == 'missing_observation',
                    'Cold-start path unexpectedly usable')
            groups['missing_observation'].append((frozen, 0, entry, 'natural_cold_start'))
        require(groups['normal'], 'No normal inference inputs')
        # Cover unavailable observations during operation as well as empty cold buffers.
        # These fixtures preserve values and timestamps; only probe validity is removed.
        missing=[]
        for raw,step,entry,_ in groups['normal'][:max(1,cfg['fallback_ring_size']//2)]:
            arrays={k:v.copy() for k,v in raw.arrays.items()}; arrays['sensor_valid'][:,:8]=False
            masked=MemoryEpisode(dict(raw.meta),arrays)
            require(measured_path(engine.decide(lambda: masked.snapshot_ns(step*100_000_000)))=='missing_observation',
                    'Masked probe fixture unexpectedly usable')
            missing.append((masked,step,entry,'synthetic_missing_probes'))
        cold=groups['missing_observation']
        groups['missing_observation']=cold[:max(1,cfg['fallback_ring_size']-len(missing))]+missing
        # Deterministic fault fixtures ensure fallback coverage even if a particular seed has no
        # natural unsafe anchor. They are benchmark inputs only, never control/quality episodes.
        if not groups['no_predicted_feasible_candidate']:
            raw, step, entry, _ = groups['normal'][0]
            arrays = {k: v.copy() for k, v in raw.arrays.items()}
            arrays['sensor'][:, :8] = 140.; arrays['thermal'][:] = 140.
            unsafe = MemoryEpisode(dict(raw.meta), arrays)
            require(measured_path(engine.decide(lambda: unsafe.snapshot_ns(step*100_000_000))) ==
                    'no_predicted_feasible_candidate', 'Unsafe fixture did not exercise PINO gate')
            groups['no_predicted_feasible_candidate'].append((unsafe, step, entry, 'synthetic_140C_fault'))
        entries = []
        for key, group in groups.items():
            count = cfg['normal_ring_size'] if key == 'normal' else cfg['fallback_ring_size']
            indices = np.linspace(0, len(group)-1, min(len(group), count), dtype=int)
            for i in indices:
                raw, step, entry, origin = group[int(i)]
                filename = f'input_{len(entries):04d}.npz'
                with (output/filename).open('wb') as stream:
                    np.savez(stream, **raw.arrays)
                entries.append({'id': len(entries), 'file': filename, 'sha256': digest(output/filename),
                                'meta': raw.meta, 'step': step, 'episode': entry['episode_id'],
                                'stratum': entry['stratum'], 'cohort': entry['cohort'],
                                'path': key, 'origin': origin})
        result = {'entries': entries, 'counts': {k: sum(e['path']==k for e in entries) for k in PATHS},
                  'selection': 'fixed evenly spaced inputs, then PINO path classification; no latency-based selection',
                  'fault_fixtures_are_quality_data': False, 'snapshot_preparation_inside_timer': True,
                  'plan_sha256': digest(run/'15_eval_data/plan.json')}
        write_json(output/'ring.json', result)
        return {'counts': result['counts'], 'inputs': len(entries), 'synthetic_faults': sum(e['origin'].startswith('synthetic') for e in entries)}
    finally:
        data.close(); engine.close(); stop.close()


def load_inputs(run):
    folder = run/'16_latency_inputs'; verify(folder)
    manifest = read_json(folder/'ring.json'); groups = {k: [] for k in PATHS}
    for item in manifest['entries']:
        require(digest(folder/item['file']) == item['sha256'], 'Latency input changed')
        with np.load(folder/item['file'], allow_pickle=False) as data:
            raw = MemoryEpisode(item['meta'], {k: data[k] for k in data.files})
        groups[item['path']].append((raw, item))
    require(all(groups.values()), 'Missing latency path input group')
    return groups


def trial_identity(run, config, method, repeat):
    return {**identity(run, config), 'ring': digest(run/'16_latency_inputs/ring.json'),
            'method': method, 'repeat': repeat, 'latency_schema': 'path-stratified-v1'}


def schedule(config):
    cfg = config['validation']
    return [(key, i) for key in PATHS for i in range(cfg['latency_requests'] if key=='normal' else cfg['fallback_requests_per_path'])]


def ring_input(group,index,requests):
    # Small development budgets must exercise both ends of a ring, including the
    # late-observation fault fixtures appended after cold-start inputs.
    position=index%len(group) if requests>=len(group) else round(index*(len(group)-1)/max(1,requests-1))
    return group[position]


def summarize_requests(rows):
    good = [r for r in rows if r['error'] is None]
    return {'requests': len(rows), 'failures': len(rows)-len(good),
            'local_e2e_ms': quantiles([r['e2e_ms'] for r in good]),
            'gpu_compute_ms': quantiles([r['gpu_compute_ms'] for r in good]),
            'deadline_miss_count': sum(bool(r['error']) or r['deadline_miss'] for r in rows),
            'deadline_miss_fraction': sum(bool(r['error']) or r['deadline_miss'] for r in rows)/len(rows),
            'native_scored_requests': sum(r['tokens']>0 for r in good),
            'thermal_fresh_e2e_ms': quantiles([r['e2e_ms'] for r in good if thermal_is_fresh(r['thermal_age_s'])]),
            'thermal_older_e2e_ms': quantiles([r['e2e_ms'] for r in good if not thermal_is_fresh(r['thermal_age_s'])]),
            'distinct_inputs': len({r['input_id'] for r in rows})}


def run_trial(root, run, config, method, repeat, launched_ns, device='cuda'):
    setup_torch(config, device); cfg = config['validation']
    output = run/'19_latency/trials'/f'{method}_{repeat}'; output.mkdir(parents=True, exist_ok=True)
    groups = load_inputs(run); order = schedule(config)
    journal = Journal(output/'requests.jsonl', trial_identity(run, config, method, repeat))
    require(len(journal.rows)<=len(order), 'Too many latency requests')
    for i, row in enumerate(journal.rows):
        require((row['path'], row['path_request'])==order[i], 'Latency path order changed')
    session_file = output/'sessions.json'; sessions = read_json(session_file) if session_file.exists() else []
    engine = None; stop = StopRequest()
    try:
        if len(journal.rows)<len(order):
            before = time.perf_counter(); counters = system_counters()
            engine = DecisionEngine(root, run, method, device=device)
            session = {'start_request': len(journal.rows), 'system_before': counters,
                       'model_load_s': time.perf_counter()-before, 'pid': __import__('os').getpid(),
                       'fresh_interpreter': True, 'status': 'RUNNING'}
            sessions.append(session); write_json(session_file, sessions)
            for key in PATHS:
                count = cfg['warmup_requests'] if key=='normal' else cfg['fallback_warmup_requests']
                for i in range(count):
                    raw, item = ring_input(groups[key],i,count); engine.reset()
                    result = engine.decide(lambda: raw.snapshot_ns(item['step']*100_000_000))
                    require(measured_path(result)==key, 'Latency warmup path mismatch')
                    if key=='normal' and i==0:
                        session['process_start_to_first_result_ms'] = (time.perf_counter_ns()-launched_ns)/1e6
                        session['first_result_e2e_ms'] = result['e2e_ms']
                    if stop.signal: raise InterruptedError('Warmup interrupted')
            if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            for index in range(len(journal.rows), len(order)):
                key, i = order[index]
                count=cfg['latency_requests'] if key=='normal' else cfg['fallback_requests_per_path']
                raw, item = ring_input(groups[key],i,count); engine.reset()
                started = time.perf_counter_ns()
                try:
                    result = engine.decide(lambda: raw.snapshot_ns(item['step']*100_000_000))
                    require(measured_path(result)==key, 'Measured path differs from frozen input class')
                    result['error'] = None
                except (RuntimeError, ValueError) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError): raise
                    result = {'error': str(exc), 'e2e_ms': (time.perf_counter_ns()-started)/1e6,
                              'gpu_compute_ms': None, 'tokens': 0, 'fallback_reason': 'inference_error'}
                journal.append({'request': repeat*len(order)+index, 'path_request': i, 'path': key,
                                'input_id': item['id'], 'episode': item['episode'], 'step': item['step'],
                                'cohort': item['cohort'], 'stratum': item['stratum'], 'origin': item['origin'],
                                'repeat': repeat, 'session': len(sessions)-1,
                                'deadline_miss': result['e2e_ms']>cfg['deadline_ms'], **result})
                if (index+1)%100==0: print(f'[L1 recheck latency] {method} repeat={repeat} {index+1}/{len(order)}', flush=True)
                require(not (len(journal.rows)>=3 and all(r['error'] for r in journal.rows[-3:])), 'Three consecutive inference errors')
                if stop.signal: raise InterruptedError('Latency request committed; resume will repeat warmup')
            session['system_after'] = system_counters(); session['status'] = 'COMPLETE'
            if device.startswith('cuda'):
                session['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
                session['peak_cuda_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
            write_json(session_file, sessions)
        result = {'method': method, 'repeat': repeat, 'sessions': sessions,
                  'by_path': {key: summarize_requests([r for r in journal.rows if r['path']==key]) for key in PATHS},
                  'mixed_diagnostic_only': summarize_requests(journal.rows),
                  'normal_is_primary_latency': True, 'mixture_is_not_production_frequency': True,
                  'quantiles': 'numpy linear, completed successes; all errors in failure/deadline denominator'}
        write_json(output/'summary.json', result); seal(output, trial_identity(run, config, method, repeat))
        return result
    finally:
        if engine is not None: engine.close()
        stop.close()


def benchmark(root, run, config, output):
    cfg = config['validation']; stop = StopRequest(); order = []
    try:
        for repeat in range(cfg['latency_repeats']):
            # Rotate method order to reduce fixed ordering effects across sessions.
            methods = METHODS[repeat%len(METHODS):]+METHODS[:repeat%len(METHODS)]
            for method in methods:
                folder = output/'trials'/f'{method}_{repeat}'; order.append([method, repeat])
                if (folder/'COMPLETE').exists(): verify(folder)
                else:
                    command = [sys.executable, '-u', '-m', 'ojpoc.recheck_latency', '--root', str(root),
                               '--run', str(run), '--method', method, '--repeat', str(repeat),
                               '--launched-ns', str(time.perf_counter_ns())]
                    subprocess.run(command, check=True)
                    verify(folder)
                if stop.signal: raise InterruptedError('Latency subprocess completed; stopping')
        reports = {m: [read_json(output/'trials'/f'{m}_{r}'/'summary.json') for r in range(cfg['latency_repeats'])] for m in METHODS}
        result = {'methods': reports, 'execution_order': order, 'deadline_ms': cfg['deadline_ms'],
                  'independent_process_repeats': cfg['latency_repeats'],
                  'normal_requests_per_repeat': cfg['latency_requests'],
                  'fallback_requests_per_path_per_repeat': cfg['fallback_requests_per_path'],
                  'warmup_normal': cfg['warmup_requests'], 'warmup_each_fallback': cfg['fallback_warmup_requests'],
                  'p99_run_variability_ms': {m: quantiles([r['by_path']['normal']['local_e2e_ms']['p99'] for r in reports[m]]) for m in METHODS},
                  'normal_is_primary_latency': True, 'mixture_is_not_production_frequency': True,
                  'request_batch': 1, 'candidate_batch': 9, 'online_features': True, 'token_cache': False,
                  'test_accessed': False, 'hard_realtime_guarantee': False, 'hard_request_timeout_ms': None,
                  'scope': 'isolated local E2E; excludes model loading, disk/network/logging; path-conditional traces',
                  'thermal_age_tolerance_s': cfg['thermal_age_tolerance_s']}
        write_json(output/'latency_summary.json', result); return result
    finally: stop.close()


def load_trace(run, config, method):
    groups = []
    for repeat in range(config['validation']['latency_repeats']):
        folder = run/'19_latency/trials'/f'{method}_{repeat}'; verify(folder)
        journal = Journal(folder/'requests.jsonl', trial_identity(run, config, method, repeat))
        require(len(journal.rows)==len(schedule(config)), 'Incomplete stratified latency trace')
        require(all(any(r['path']==key and r['error'] is None for r in journal.rows) for key in PATHS), 'Missing successful path trace')
        groups.append(journal.rows)
    # Interleave trials so nearest-token matching does not consume only the first repeat.
    return list(chain.from_iterable(zip(*groups)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--method', choices=METHODS, required=True); p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--launched-ns', type=int, required=True)
    args = p.parse_args()
    from .recheck import verify_run
    cfg = verify_run(args.root.resolve(), args.run.resolve())
    run_trial(args.root, args.run, cfg, args.method, args.repeat, args.launched_ns)


if __name__=='__main__': main()
