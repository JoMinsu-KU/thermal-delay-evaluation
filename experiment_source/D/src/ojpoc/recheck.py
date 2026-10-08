"""L1.1: frozen-parent evaluation, replay and fresh episodes; no model training."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import yaml

from .common import (ROOT, require, read_json, write_json, atomic_text, digest,
                     source_hashes, verify)
from .cli import l0_runner, now
from .evaluation import evaluation_plan, prepare_evaluation

STAGES = [
    ('00_preflight', 'Environment and immutable parent verification'),
    ('01_frozen_inputs', 'Copy frozen checkpoints and record lineage; no training'),
    ('15_eval_data', 'Original episode replay plus fresh held-out development episodes'),
    ('16_latency_inputs', 'Freeze normal, missing-observation and unsafe raw input rings'),
    ('17_policy_regret', 'Selected policies only; separate replay/fresh regret'),
    ('18_closed_loop', 'Zero-compute-delay paired control, five controllers'),
    ('19_latency', 'Path-conditional latency, isolated process repeats'),
    ('20_delayed_loop', 'Strict matching of measured fallback/normal delays'),
    ('21_report', 'Separate execution, replication and latency findings'),
]


def hashes(root):
    result = source_hashes(root)
    for name in ['run_l1_recheck.sh', 'configs/l1_recheck.yaml']:
        result[name] = digest(root/name)
    for path in sorted((root/'tests/l1').glob('*.py')):
        result[str(path.relative_to(root))] = digest(path)
    return result


def parent_path(root, run_id):
    l0_runner(root).validate_run_id(run_id)
    path = (root/'runs/L1'/run_id).resolve()
    require(path.parent == (root/'runs/L1').resolve(), 'Parent must be a direct L1 run')
    return path


def parent_fingerprint(parent):
    paths = [parent/name for name in ['summary.json', 'provenance.json', 'resolved_config.json']]
    paths += sorted(parent.glob('*/manifest.json')) + sorted(parent.glob('*/COMPLETE'))
    return {str(p.relative_to(parent)): digest(p) for p in paths}


def validate_parent(root, parent):
    summary = read_json(parent/'summary.json')
    require(summary['status']=='L1_QUALITY_PASS' and not summary['check_only'], 'Parent must be a completed quality-pass L1 pilot/core')
    require(summary.get('extended_validation_completed'), 'Parent PV evaluation incomplete')
    cfg = read_json(parent/'resolved_config.json'); prov = read_json(parent/'provenance.json')
    require(digest(parent/'resolved_config.json')==prov['config_hash'], 'Parent config changed')
    require(cfg['poc']['runtime']['actuation_enabled'] is False, 'Simulation only')
    for item in summary['stages']:
        require(item['status']=='PASS', 'Incomplete parent stage')
        verify(parent/item['name'])
        for name, expected in read_json(parent/item['name']/'result.json').get('external_artifacts', {}).items():
            require(digest(parent/name)==expected, 'Parent external artifact changed')
    for name, expected in read_json(parent/'15_eval_data/plan.json')['checkpoints_frozen_before_eval'].items():
        require(digest(parent/name)==expected, 'Parent checkpoint changed')
    # The new implementation intentionally differs. Do not require old source hashes to equal new code.
    return {'run_id': parent.name, 'status': summary['status'], 'training_seed': cfg['seed'],
            'fingerprint': parent_fingerprint(parent), 'test_accessed': False}


def configuration(root, profile='pilot', parent_id=None):
    settings = yaml.safe_load((root/'configs/l1_recheck.yaml').read_text())
    require(settings['schema_version']=='l1-recheck-v1', 'Unsupported recheck schema')
    require(profile in settings['profiles'], 'Unknown recheck profile')
    require(settings['deadline_ms']==100., 'Keep the preregistered 100ms deadline')
    require(settings['bootstrap_resamples']>=2000, 'At least 2000 episode resamples')
    require(settings['thermal_age_tolerance_s']==1e-8 and settings['violation_time_tolerance_s']==1e-8,
            'Boundary tolerances must match the implemented numerical contract')
    require(type(settings['seed']) is int and 0<=settings['seed']<2**31, 'Invalid evaluation seed')
    require(type(settings['episode_code_start']) is int and settings['episode_code_start']>=6000, 'Reserved recheck episode code range')
    for key in ['normal_ring_size', 'fallback_ring_size']:
        require(type(settings[key]) is int and settings[key]>0, 'Positive input ring size required')
    budget = settings['profiles'][profile]
    for key, value in budget.items():
        require(isinstance(value, (int, float)) and value>0, 'Positive recheck budgets required')
        if key!='duration_s': require(type(value) is int, 'Counts must be integers')
    require(budget['duration_s']>2 and budget['anchors_per_episode']<=round(budget['duration_s']*10)-20,
            'Evaluation anchors must retain the full two-second prediction horizon')
    if profile=='pilot':
        require(budget['duration_s']==60 and budget['latency_requests']>=10000 and budget['warmup_requests']>=200
                and budget['latency_repeats']>=3 and budget['fallback_requests_per_path']>=1000,
                'Pilot requires 60s episodes and full repeated latency budgets')
    parent_id = parent_id or settings['parent_run']; parent = parent_path(root, parent_id)
    cfg = copy.deepcopy(read_json(parent/'resolved_config.json'))
    original = cfg['seed']; cfg['profile'] = profile; cfg['seed'] = settings['seed']
    require(cfg['seed']!=original, 'Fresh evaluation seed must differ from training seed')
    cfg['schema'] = 'l1-recheck-resolved-v1'
    cfg['validation'] = {**{k:v for k,v in settings.items() if k not in ['profiles','parent_run','seed']}, **budget,
                         'closed_loop_normal': budget['normal_episodes'], 'closed_loop_stress': budget['stress_episodes'],
                         'policy_checkpoints': ['selected'], 'strict_latency_matching': True,
                         'primary_cohort': 'fresh', 'latency_ring_size': settings['normal_ring_size']}
    cfg['recheck'] = {'parent_run': parent_id, 'training_seed': original, 'evaluation_seed': cfg['seed'],
                      'training_enabled': False, 'model_selection_enabled': False, 'primary_cohort': 'fresh',
                      'memory_guard_enabled': False}
    return cfg


def import_files(parent):
    files = [parent/'01_generate/data/manifests/split.json']
    for name in ['07_pino_balance', '14_fno_balance']:
        files += [parent/name/(checkpoint+'.pt') for checkpoint in ['initial','best','last']]
    for name in ['09_native_no_physics', '11_native_physics_text']:
        for checkpoint in ['best_model','adapt_model']:
            verify(parent/name/checkpoint)
            files += [p for p in sorted((parent/name/checkpoint).rglob('*')) if p.is_file()]
    return {str(p.relative_to(parent)): digest(p) for p in files}


def copy_frozen(parent, run):
    files = import_files(parent)
    for name, expected in files.items():
        target = run/name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): require(digest(target)==expected, 'Frozen copy changed or incomplete')
        else:
            temporary = target.with_name('.'+target.name+'.import')
            shutil.copyfile(parent/name, temporary)
            require(digest(temporary)==expected, 'Copy hash mismatch')
            os.replace(temporary, target)
    return {'external_artifacts': files, 'training_executed': False, 'checkpoint_selection': 'parent validation best, unchanged'}


def verify_run(root, run):
    require(run.resolve().parent==(root/'runs/L1_recheck').resolve(), 'Recheck run must be under runs/L1_recheck')
    prov = read_json(run/'provenance.json')
    require(prov['source_hashes']==hashes(root), 'Source/config changed; start a new recheck run')
    require(digest(run/'resolved_config.json')==prov['config_hash'], 'Resolved configuration changed')
    cfg = read_json(run/'resolved_config.json'); parent = parent_path(root, cfg['recheck']['parent_run'])
    require(parent_fingerprint(parent)==prov['parent_fingerprint'], 'Parent lineage changed')
    imported = run/'01_frozen_inputs/result.json'
    if imported.exists() and read_json(imported).get('status')=='PASS':
        for name, expected in read_json(imported)['external_artifacts'].items():
            require(digest(run/name)==expected, 'Frozen checkpoint/input changed')
    return cfg


def prepare_data(root, run, cfg, output):
    parent = parent_path(root, cfg['recheck']['parent_run'])
    old = read_json(parent/'15_eval_data/plan.json')['entries']
    original = read_json(parent/'01_generate/data/manifests/split.json')['episodes']
    fresh, _ = evaluation_plan(cfg)
    for key in ['code','group_id','sim_seed','noise_seed','render_seed']:
        require(not ({e[key] for e in fresh}&{e[key] for e in [*old,*original]}), f'Fresh evaluation overlaps parent: {key}')
    for entry in fresh: entry['cohort']='fresh'
    replay = []
    for stratum in ['normal','stress']:
        available = [e for e in old if e['closed_loop'] and e['stratum']==stratum]
        count = cfg['validation']['replay_'+stratum]
        require(len(available)>=count, 'Insufficient parent replay episodes')
        replay += [{**e, 'cohort': 'replay', 'closed_loop': True} for e in available[:count]]
    result = prepare_evaluation(root, run, cfg, output, [*replay,*fresh], parent/'15_eval_data/data')
    result.update(fresh_episodes=len(fresh), replay_episodes=len(replay), primary_cohort='fresh',
                  replay_scope='same raw inputs/initial states as parent; smoke truncates horizon to 3s',
                  parent_run=parent.name)
    return result


def final_report(root, run, cfg, output):
    from .pilot import summarize_closed_loop, METHODS
    parent = parent_path(root, cfg['recheck']['parent_run'])
    validate_parent(root, parent)
    loops = {}
    for mode, stage in [('zero','18_closed_loop'),('delayed','20_delayed_loop')]:
        rows = read_json(run/stage/f'closed_loop_{mode}.json')['episodes']
        loops[mode] = {cohort: summarize_closed_loop([r for r in rows if r['cohort']==cohort], cfg, mode)
                       for cohort in ['replay','fresh']}
    latency = read_json(run/'19_latency/latency_summary.json')
    policy = read_json(run/'17_policy_regret/regret_comparison.json')
    normal_deadline = all(r['by_path']['normal']['failures']==0 and
                          r['by_path']['normal']['local_e2e_ms']['p99'] is not None and
                          r['by_path']['normal']['local_e2e_ms']['p99']<=cfg['validation']['deadline_ms']
                          for r in latency['methods']['physics'])
    all_paths = all(v['failures']==0 and v['local_e2e_ms']['p99'] is not None and
                    v['local_e2e_ms']['p99']<=cfg['validation']['deadline_ms']
                    for r in latency['methods']['physics'] for v in r['by_path'].values())
    delayed_finding = loops['delayed']['fresh']['PV3']
    pv4 = ('NEGATIVE' if not (normal_deadline and all_paths) or delayed_finding=='NEGATIVE' else
           'POSITIVE' if delayed_finding=='POSITIVE' else 'INCONCLUSIVE')
    findings = {'fresh_zero_delay': loops['zero']['fresh']['PV3'],
                'replay_zero_delay': loops['zero']['replay']['PV3'],
                'fresh_delayed_control': delayed_finding, 'latency_plus_delayed_control': pv4}
    if cfg['profile']=='smoke': findings = {k:'SMOKE_ONLY' for k in findings}
    # Exact replay is reported separately, never pooled as additional independent episodes.
    parent_zero = read_json(parent/'18_closed_loop/closed_loop_zero.json')
    parent_delayed = read_json(parent/'20_delayed_loop/closed_loop_delayed.json')
    replay_changes = {}
    for mode, previous in [('zero',parent_zero),('delayed',parent_delayed)]:
        prior = {r['episode']:r for r in previous['episodes']}
        replay_changes[mode] = [{
            'episode': row['episode'], 'methods': {m:{
                'parent_cost': prior[row['episode']]['methods'][m]['episode_cost'],
                'recheck_cost': row['methods'][m]['episode_cost'],
                'difference': row['methods'][m]['episode_cost']-prior[row['episode']]['methods'][m]['episode_cost']
            } for m in METHODS}} for row in loops[mode]['replay']['episodes']]
    result = {'evaluation_completed': True, 'profile': cfg['profile'], 'parent_run': parent.name,
              'training_executed': False, 'models_reselected': False, 'primary_cohort': 'fresh',
              'findings': findings, 'loops_by_cohort': loops, 'parent_replay_changes': replay_changes,
              'replay_horizon_comparable': cfg['profile']=='pilot',
              'regret_by_cohort': {k:v for k,v in policy['stratified'].items() if k.startswith('cohort:')},
              'physics_normal_p99_deadline_met': normal_deadline, 'physics_all_paths_deadline_met': all_paths,
              'operator_learning_retested': False, 'original_PV1': read_json(parent/'21_report/validation_summary.json')['findings']['PV1'],
              'test_accessed': False, 'L2_complete': False,
              'limitations': ['one frozen training seed; fresh evaluation seeds are not independent retraining',
                             'measured input ring repeats are not independent plant states',
                             'empirical delay trace, not live hardware realtime',
                             'fallback benchmark frequency is diagnostic, not production prevalence',
                             'operator/observer/physics text serialization unchanged; prior peak-error limitation retained']}
    write_json(output/'validation_summary.json', result)
    lines = ['# L1.1 frozen-model recheck', '', f'Parent: `{parent.name}`; profile: `{cfg["profile"]}`',
             '재학습/모델 재선택 없음. 원래 episode 재평가와 새로운 episode 결과를 분리한다.',
             '실행 완료는 효과 입증을 뜻하지 않는다. smoke는 경로 검증만 수행한다.', '',
             '| Question | Finding |', '|---|---|']
    lines += [f'| {k} | {v} |' for k,v in findings.items()]
    for cohort in ['fresh','replay']:
        for mode in ['zero','delayed']:
            block = loops[mode][cohort]; comparison = block['paired_episode_cost']
            lines += ['', f'## {cohort} / {mode}', '',
                      f'Episodes={len(block["episodes"])}; Physics−No Physics cost difference={comparison["mean_difference"]}; episode bootstrap CI95={comparison["ci95"]}', '',
                      '| Method | Mean cost | Mean tracking K | Max overshoot K | Violations | Violation seconds | Fallback seconds |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for method, m in block['methods'].items():
                lines.append(f'| {method} | {m["mean_episode_cost"]:.7g} | {m["mean_tracking_rmse_k"]:.5g} | {m["max_overshoot_k"]:.5g} | {m["violation_episodes"]} | {m["violation_s"]:.7g} | {m["fallback_s"]:.7g} |')
            lines += ['', 'Physics의 5% 비열등성: '+', '.join(f'{m}={v["noninferior_5pct"]}' for m,v in block['reference_controller_comparisons'].items())]
    lines += ['', '## Latency by execution path', '',
              '정상 판단 10,000회와 fallback 표본은 서로 다른 분포다. 혼합 분포로 정상 추론의 deadline 달성을 판정하지 않는다.',
              '각 반복은 새 프로세스다. 모델별 실행 순서는 반복마다 순환한다. 실제 예산은 resolved_config.json을 따른다.', '',
              '| Method | Repeat | Path | Requests | p50 ms | p95 ms | p99 ms | Deadline miss | Failures |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for method, repeats in latency['methods'].items():
        for repeat in repeats:
            for path, m in repeat['by_path'].items():
                q=m['local_e2e_ms']
                lines.append(f'| {method} | {repeat["repeat"]} | {path} | {m["requests"]} | {q["p50"]} | {q["p95"]} | {q["p99"]} | {m["deadline_miss_fraction"]} | {m["failures"]} |')
    lines += ['', '## Selected-policy regret', '', 'regret 정의 불가와 fallback은 실패/coverage 분모에 보존한다. 해당 표는 정의된 공통 row 비교다.', '',
              '| Cohort | Common rows | Episodes | Physics−No Physics regret | CI95 |', '|---|---:|---:|---:|---|']
    for cohort, block in result['regret_by_cohort'].items():
        p=block['paired']; lines.append(f'| {cohort} | {p["common_rows"]} | {p["episodes"]} | {p["mean_difference"]} | {p["ci95"]} |')
        lines.append('')
        for m,v in block['methods'].items(): lines.append(f'{m}: rows={v["rows"]}, regret_count={v["regret"]["count"]}, failures={v["failure"]}, unexpected_fallback={v["unexpected_fallback"]}.')
    lines += ['', '원래 PV1 학습 효과 판정은 재평가하지 않았다. 신규 checkpoint 선택·test/OOD 평가·L2 구현은 이 실행에 포함하지 않는다.',
              '세부 원래/신규 episode 비교, token 길이 매칭 오차, 자원/CPU/OOM 기록은 각 단계의 JSON과 trace를 확인한다.',
              'fallback 경로 trace가 없으면 추정 대체하지 않고 실행을 실패 처리한다. 100ms 초과 명령은 기존과 같이 폐기 후 shutdown한다.']
    atomic_text(output/'report.md', '\n'.join(lines)+'\n')
    return result


def execute_stage(name, root, run, cfg, output):
    from . import pilot, recheck_latency
    parent = parent_path(root, cfg['recheck']['parent_run'])
    if name=='00_preflight':
        from .stages import preflight
        return {**preflight(root,run,cfg,output), 'parent': validate_parent(root,parent), 'training_executed': False}
    if name=='01_frozen_inputs': return copy_frozen(parent,run)
    if name=='15_eval_data': return prepare_data(root,run,cfg,output)
    if name=='16_latency_inputs': return recheck_latency.make_inputs(root,run,cfg,output)
    if name=='17_policy_regret': return pilot.policy_regret(root,run,cfg,output)
    if name=='18_closed_loop': return pilot.closed_loop(root,run,cfg,output,'zero')
    if name=='19_latency': return recheck_latency.benchmark(root,run,cfg,output)
    if name=='20_delayed_loop': return pilot.closed_loop(root,run,cfg,output,'delayed')
    if name=='21_report': return final_report(root,run,cfg,output)
    if name=='99_readiness':
        code = subprocess.call([sys.executable,'-m','pytest',str(root/'tests/l1'),'-m','not gpu','-q'], cwd=root)
        require(code==0,'Recheck implementation tests failed')
        return {'tests': 'tests/l1 excluding explicit GPU integration', 'experiment_executed': False}
    raise ValueError('Unknown recheck stage')


def stage_main(name):
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args(); root=args.root.resolve(); run=args.run_dir.resolve(); cfg=verify_run(root,run)
    output=run/name; output.mkdir(exist_ok=True); started=now()
    try:
        result=execute_stage(name,root,run,cfg,output)
        write_json(output/'result.json',{**result,'status':'PASS','started_at':started,'finished_at':now()})
        excluded={'console.log','command.json','resources.jsonl','guard.json','status.json','manifest.json','COMPLETE'}
        files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and
               p.name not in excluded and 'attempts' not in p.relative_to(output).parts and
               not any(s.startswith('.') for s in p.relative_to(output).parts)}
        write_json(output/'manifest.json',{'files':files,'metadata':{'stage':name,'schema':'l1-recheck-v1'}})
        atomic_text(output/'COMPLETE',digest(output/'manifest.json')+'\n')
        print(f'[L1 recheck] {name} PASS',flush=True)
    except BaseException as exc:
        write_json(output/'result.json',{'status':'FAIL','error':str(exc),'started_at':started,'traceback':traceback.format_exc()})
        raise


def update(root, run, state):
    write_json(run/'summary.json',state)
    lines=['# L1.1 execution','',f'Run: {run.name}',f'Status: {state["status"]}',
           f'Parent: {state["parent_run"]}; no training / no model reselection','',
           '| Stage | Status | Seconds |','|---|---|---:|']
    lines += [f'| {r["name"]} | {r["status"]} | {r.get("seconds",0):.2f} |' for r in state['stages']]
    if state.get('findings'): lines += ['',json.dumps(state['findings'],ensure_ascii=False)]
    if state.get('error'): lines += ['',state['error']]
    lines += ['','Completion is separate from scientific support. Final report: 21_report/report.md.']
    atomic_text(run/'summary.md','\n'.join(lines)+'\n')
    write_json(root/'logs/L1_recheck'/f'{run.name}.json',{'run_directory':str(run),'status':state['status']})


def execute(root, cfg, run_id, resume=False, check_only=False, memory_guard=False):
    helper=l0_runner(root); helper.validate_run_id(run_id)
    parent=parent_path(root,cfg['recheck']['parent_run'])
    with helper.exclusive_lock(root/'locks/l0.lock'):
        run=root/'runs/L1_recheck'/run_id
        if resume:
            state=read_json(run/'summary.json'); saved=verify_run(root,run)
            require(saved==cfg,'Resolved config differs; start a new run')
            require(state['check_only']==check_only and state['memory_guard_enabled']==memory_guard,'Resume mode changed')
            require(state['status'] not in ['RECHECK_COMPLETE','RECHECK_SMOKE_PASS','READINESS_PASS'],'Run already completed')
        else:
            # Validate before creating a scientific run, so an invalid parent cannot start evaluation.
            validate_parent(root,parent)
            run.mkdir(parents=True,exist_ok=False); write_json(run/'resolved_config.json',cfg)
            provenance={'source_hashes':hashes(root),'config_hash':digest(run/'resolved_config.json'),
                        'parent_fingerprint':parent_fingerprint(parent),'created_at':now(),'python':sys.executable,
                        'parent_run':parent.name,'argv':sys.argv,'training_executed':False}
            write_json(run/'provenance.json',provenance); write_json(root/'locks/L1_recheck'/f'{run_id}.json',provenance)
            selected=[STAGES[0],('99_readiness','CPU implementation checks')] if check_only else STAGES
            state={'run_id':run_id,'parent_run':parent.name,'profile':cfg['profile'],'started_at':now(),
                   'check_only':check_only,'memory_guard_enabled':memory_guard,'status':'RUNNING',
                   'stages':[{'name':name,'title':title,'status':'PENDING'} for name,title in selected]}
        state['status']='RUNNING'; state.pop('error',None); update(root,run,state)
        try:
            for item in state['stages']:
                name=item['name']; output=run/name
                if item['status']=='PASS':
                    verify(output)
                    for relative,expected in read_json(output/'result.json').get('external_artifacts',{}).items():
                        require(digest(run/relative)==expected,'Frozen input changed')
                    continue
                if output.exists():
                    archive=output/'attempts'/now().replace(':','-'); archive.mkdir(parents=True)
                    for old in ['console.log','command.json','resources.jsonl','guard.json','status.json','result.json','manifest.json','COMPLETE']:
                        if (output/old).exists(): shutil.move(output/old,archive/old)
                output.mkdir(exist_ok=True); item['status']='RUNNING'; update(root,run,state)
                worker=[sys.executable,'-u',str(root/'stages/L1/recheck'/name/'run.py'),'--root',str(root),'--run-dir',str(run)]
                command=helper.guard_command(root,output,worker,cfg['poc'],{'guard_interval_s':1.,'guard_grace_s':60.},memory_guard=memory_guard)
                write_json(output/'command.json',{'argv':command,'memory_guard_enabled':memory_guard})
                env=dict(os.environ); env['HF_HUB_OFFLINE']='1'; env['PYTHONPATH']=str(root/'src')
                started=time.monotonic(); print(f'\n[L1 recheck] {name}: {item["title"]}\nOutput: {output}',flush=True)
                code,_=helper.run_child(command,output/'console.log',env)
                item.update(seconds=time.monotonic()-started,exit_code=code)
                require(code==0,f'{name} exited {code}; inspect {output}/console.log')
                require(read_json(output/'guard.json')['status']=='PASS','Stage monitor failed')
                require(read_json(output/'result.json')['status']=='PASS','Stage failed')
                verify(output); item['status']='PASS'; write_json(output/'status.json',item); update(root,run,state)
            if check_only: state['status']='READINESS_PASS'
            else:
                result=read_json(run/'21_report/result.json'); require(result['evaluation_completed'],'Incomplete evaluation')
                state.update(findings=result['findings'],training_executed=False,
                             status='RECHECK_SMOKE_PASS' if cfg['profile']=='smoke' else 'RECHECK_COMPLETE')
            state['finished_at']=now(); update(root,run,state)
            print(f'\n[L1 recheck] {state["status"]}\nReport: {run/"21_report/report.md"}',flush=True)
            return 0
        except BaseException as exc:
            for item in state['stages']:
                if item['status']=='RUNNING': item['status']='FAIL'
                elif item['status']=='PENDING': item['status']='NOT_RUN'
            state.update(status='FAIL',error=str(exc),finished_at=now()); update(root,run,state)
            print(f'[L1 recheck] FAIL: {exc}\nResume: bash run_l1_recheck.sh --resume {run_id}',file=sys.stderr,flush=True)
            return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--profile',choices=['smoke','pilot'])
    p.add_argument('--parent-run'); p.add_argument('--run-id'); p.add_argument('--resume')
    p.add_argument('--check-only',action='store_true'); p.add_argument('--plan',action='store_true')
    guard=p.add_mutually_exclusive_group()
    guard.add_argument('--memory-guard',dest='memory_guard',action='store_true',default=None)
    guard.add_argument('--no-memory-guard',dest='memory_guard',action='store_false')
    args=p.parse_args(); root=args.root.resolve()
    if args.resume:
        require(args.run_id is None,'Use either --resume or --run-id')
        l0_runner(root).validate_run_id(args.resume)
        run=root/'runs/L1_recheck'/args.resume; cfg=verify_run(root,run); state=read_json(run/'summary.json')
        require(args.profile is None or args.profile==cfg['profile'],'Resume profile changed')
        require(args.parent_run is None or args.parent_run==cfg['recheck']['parent_run'],'Resume parent changed')
        memory_guard=state['memory_guard_enabled'] if args.memory_guard is None else args.memory_guard
        check_only=state['check_only'] or args.check_only
    else:
        cfg=configuration(root,args.profile or 'pilot',args.parent_run)
        memory_guard=False if args.memory_guard is None else args.memory_guard; check_only=args.check_only
    if args.plan:
        print(json.dumps({'profile':cfg['profile'],'recheck':cfg['recheck'],'validation':cfg['validation']},ensure_ascii=False,indent=2))
        for name,title in STAGES: print(f'{name}: {title}')
        return 0
    run_id=args.resume or args.run_id or f'l1_recheck_{cfg["profile"]}_s{cfg["seed"]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    return execute(root,cfg,run_id,bool(args.resume),check_only,memory_guard)


if __name__=='__main__': raise SystemExit(main())
