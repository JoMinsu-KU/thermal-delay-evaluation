from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with tmp.open('x') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def seal(directory, metadata=None):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name not in {'COMPLETE', 'manifest.json'} and not p.name.startswith('.')}
    write_json(directory / 'manifest.json', {'files': files, 'metadata': metadata or {}})
    atomic_text(directory / 'COMPLETE', digest(directory / 'manifest.json') + '\n')


def verify(directory):
    directory = Path(directory)
    require((directory / 'COMPLETE').is_file(), f'Incomplete artifact: {directory}')
    require((directory / 'COMPLETE').read_text().strip() == digest(directory / 'manifest.json'), 'Manifest changed')
    manifest = read_json(directory / 'manifest.json')
    for name, expected in manifest['files'].items():
        require((directory / name).resolve().is_relative_to(directory.resolve()), 'Escaping artifact path')
        require(digest(directory / name) == expected, f'Artifact changed: {directory / name}')
    return manifest


def source_hashes(root):
    paths = [root / 'run_l1.sh', root / 'configs/l1.yaml', root / 'configs/l1_validation.yaml', root / 'configs/acceptance.yaml',
             root / 'configs/poc.yaml', root / 'stages/L0/guard.py', root / 'stages/L0/common.py',
             root / 'scripts/env.sh', root / 'scripts/activate_jev.sh', root / 'locks/openjev.commit',
             root / 'locks/requirements.lock.txt']
    paths += sorted((root / 'src/ojpoc').rglob('*.py')) + sorted((root / 'stages/L1').rglob('*.py'))
    return {str(p.relative_to(root)): digest(p) for p in paths}


def configuration(root=ROOT, profile=None, seed=2026):
    poc = yaml.safe_load((root / 'configs/poc.yaml').read_text())
    cfg = yaml.safe_load((root / 'configs/l1.yaml').read_text())
    acceptance = yaml.safe_load((root / 'configs/acceptance.yaml').read_text())
    require(set(cfg) == {'schema_version', 'default_profile', 'memory_guard', 'observer', 'device', 'profiles'},
            'Unknown or missing l1.yaml key')
    require(cfg['schema_version'] == 'l1-v1' and cfg['observer'] == 'bilinear' and cfg['device'] == 'cuda',
            'L1 v1 uses the O-bilinear baseline and CUDA model training')
    require(type(cfg['memory_guard']) is bool, 'memory_guard must be boolean')
    require(set(cfg['profiles']) == {'smoke', 'pilot', 'core'}, 'Invalid profile names')
    keys = {'anchors_per_episode', 'operator_windows', 'paired_states', 'operator_steps', 'pair_steps',
            'native_warmup_steps', 'native_adapt_steps'}
    for values in cfg['profiles'].values():
        require(set(values) == keys and all(type(v) is int and v > 0 for v in values.values()), 'Invalid L1 budget')
    require(all(cfg['profiles']['smoke'][k] <= 10 for k in keys if k.endswith('steps')), 'Smoke step cap is 10')
    caps={'operator_steps':5000,'pair_steps':1000,'native_warmup_steps':250,'native_adapt_steps':1500}
    for selected,factor in [('pilot',.1),('core',1.)]:
        require(all(cfg['profiles'][selected][k]<=int(np.ceil(v*factor)) for k,v in caps.items()),'Profile exceeds specification step cap')
    # Original specification config is deliberately immutable; L1 budgets live in l1.yaml.
    original = dict((name.strip(), sha) for sha, name in
                    (line.split(maxsplit=1) for line in (root / 'SHA256SUMS.txt').read_text().splitlines()))
    require(digest(root / 'configs/poc.yaml') == original['configs/poc.yaml'], 'Original poc.yaml changed')
    require(poc['runtime']['actuation_enabled'] is False, 'Simulation only')
    require(set(acceptance) == {'schema_version', 'solver', 'operator', 'policy', 'reload'}, 'Invalid acceptance keys')
    require(acceptance['schema_version']=='thermal-l1-acceptance-v1','Unsupported acceptance protocol')
    expected = {'solver': {'cooling_abs_k', 'conservation_abs_k', 'refinement_rmse_k', 'refinement_peak_k',
                           'refinement_order_accuracy', 'significant_cost_gap'},
                'operator': {'horizon_rmse_k', 'candidate_delta_rmse_k', 'significant_order_accuracy', 'peak_abs_error_k'},
                'policy': {'regret_improvement_min'},
                'reload': {'score_atol', 'score_rtol', 'probability_atol', 'probability_rtol'}}
    for section, names in expected.items():
        require(set(acceptance[section]) == names, f'Invalid acceptance {section}')
        require(all(np.isfinite(v) and v >= 0 for v in acceptance[section].values()), 'Invalid gate threshold')
    profile = profile or cfg['default_profile']
    require(profile in cfg['profiles'], 'Unknown profile')
    require(type(seed) is int and 0 <= seed < 2**31, 'Invalid seed')
    poc['profile']=profile; poc['seed']=seed
    pv = yaml.safe_load((root / 'configs/l1_validation.yaml').read_text())
    require(pv['schema_version']=='l1-pv-v1', 'Unknown pilot validation protocol')
    require(set(pv)=={'schema_version','bootstrap_resamples','deadline_ms','latency_repeats','latency_ring_size','profiles'}, 'Unknown PV settings')
    require(set(pv['profiles'])=={'smoke','pilot','core'}, 'Invalid PV profiles')
    pv_keys={'normal_episodes','stress_episodes','anchors_per_episode','duration_s','closed_loop_normal',
             'closed_loop_stress','warmup_requests','latency_requests'}
    for name,b in pv['profiles'].items():
        require(set(b)==pv_keys and all(np.isfinite(v) and v>0 for v in b.values()), 'Invalid PV budget')
        require(all(type(v) is int for k,v in b.items() if k!='duration_s'), 'PV counts must be integers')
        require(b['duration_s']>2 and b['anchors_per_episode']<=int(round(b['duration_s']*10))-20, 'PV anchors exceed causal horizon')
        require(b['closed_loop_normal']<=b['normal_episodes'] and b['closed_loop_stress']<=b['stress_episodes'], 'Invalid closed loop subset')
        if name!='smoke':
            require(b['duration_s']==60 and b['warmup_requests']>=200 and b['latency_requests']>=10000, 'Full PV benchmark budget required')
    require(pv['deadline_ms']==100 and pv['bootstrap_resamples']>=2000 and pv['latency_repeats']>=1 and pv['latency_ring_size']>=1, 'Invalid PV protocol')
    return {'poc': poc, 'l1': cfg, 'acceptance': acceptance, 'profile': profile, 'seed': seed,
            'budget': cfg['profiles'][profile], 'validation':{**{k:v for k,v in pv.items() if k!='profiles'},**pv['profiles'][profile]},
            'schema': 'thermal-l1-resolved-v2'}


def setup_torch(config, device='cuda'):
    import torch
    torch.set_num_threads(config['poc']['resources']['torch_threads'])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if device == 'cuda':
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one CUDA GPU')
        torch.cuda.manual_seed_all(config['seed'])
    return torch


class StopRequest:
    """Only set a flag in signal handlers; save at completed optimizer boundaries."""
    def __init__(self):
        self.signal = None
        self.previous = {s: signal.signal(s, self.receive) for s in [signal.SIGTERM, signal.SIGINT]}

    def receive(self, number, frame):
        self.signal = number

    def close(self):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def save_training(path, model, optimizer, step, rng, extra):
    """Small trainable tensors only; original Qwen base is never materialized on CPU."""
    import torch
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    state = {'parameters': {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
             'optimizer': optimizer.state_dict(), 'step': step, 'numpy_rng': rng.bit_generator.state,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'extra': json.loads(json.dumps(extra, default=lambda x: x.item() if isinstance(x,np.generic) else list(x)))}
    torch.save(state, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    atomic_text(path.with_suffix('.sha256'), digest(path) + '\n')


def restore_training(path, model, optimizer, rng, device):
    import torch
    path = Path(path)
    require(path.with_suffix('.sha256').read_text().strip() == digest(path), 'Training checkpoint checksum mismatch')
    state = torch.load(path, map_location=device, weights_only=True)
    params = dict(model.named_parameters())
    require(set(state['parameters']) == {n for n,p in params.items() if p.requires_grad}, 'Trainable set changed')
    with torch.no_grad():
        for name, value in state['parameters'].items():
            params[name].copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    return state['step'], state['extra']
