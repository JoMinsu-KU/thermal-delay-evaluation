"""One-command sequential L1 pipeline, reproducible provenance and explicit quality gates."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from .common import (ROOT, configuration, source_hashes, require, read_json, write_json,
                     atomic_text, seal, verify, digest)
from .stages import STAGES,execute_stage


def now(): return datetime.now(timezone.utc).isoformat()


def l0_runner(root):
    sys.path.insert(0,str(root/'stages/L0'))
    spec=importlib.util.spec_from_file_location('l0_orchestration',root/'stages/L0/run.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def update(run,state):
    write_json(run/'summary.json',state)
    rows=['# L1 execution','',f'Run: `{state["run_id"]}`',f'Status: **{state["status"]}**',
          f'Profile: `{state["profile"]}`; memory threshold termination: `{state["memory_guard_enabled"]}`','',
          '| Stage | Status | Seconds |','|---|---|---:|']
    rows.extend(f'| {s["name"]} | {s["status"]} | {s.get("seconds",0):.2f} |' for s in state['stages'])
    rows+=['','SMOKE_PASS means the reduced training/data workflow completed. Inspect `21_report/report.md` for PV1--PV4; `13_report/report.md` for original quality gates.',
           'READINESS_PASS covers implementation checks only. L1_QUALITY_PASS additionally requires the validation quality gates.',
           'Pilot evaluation includes native closed loops and measured local p99; it does not certify L2, calibration or final test/OOD quality.']
    if state.get('pv_findings'):
        rows+=['',f'PV findings: {state["pv_findings"]}',f'All hypotheses supported: {state["all_hypotheses_supported"]}']
    if state.get('error'): rows+=['',state['error']]
    atomic_text(run/'summary.md','\n'.join(rows)+'\n')
    write_json(run.parents[2]/'logs/L1'/f'{state["run_id"]}.json',{'run_directory':str(run),'status':state['status']})


def readiness(root,run,output):
    command=[sys.executable,'-m','pytest',str(root/'tests/l1'),'-m','not gpu','-q',
             '--junitxml='+str(output/'tests.xml')]
    code=subprocess.call(command,cwd=root)
    require(code==0,'L1 implementation tests failed')
    return {'tests':'tests/l1, excluding explicit gpu integration marker','experiment_executed':False}


def stage_main(name=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
    if name is None: p.add_argument('--stage',required=True)
    args=p.parse_args(); name=name or args.stage
    root=args.root.resolve(); run=args.run_dir.resolve(); output=run/name
    require(run.is_relative_to(root/'runs/L1'),'Run must be inside ROOT/runs/L1')
    cfg=read_json(run/'resolved_config.json'); provenance=read_json(run/'provenance.json')
    require(provenance['source_hashes']==source_hashes(root),'Implementation changed during run')
    require(provenance['config_hash']==digest(run/'resolved_config.json'),'Resolved configuration changed')
    output.mkdir(exist_ok=True); started=now()
    try:
        result=readiness(root,run,output) if name=='99_readiness' else execute_stage(name,root,run,cfg,output)
        write_json(output/'result.json',{**result,'status':'PASS','started_at':started,'finished_at':now()})
        # Guard writes its own live files; seal just scientific outputs.
        excluded={'console.log','command.json','resources.jsonl','guard.json','status.json','manifest.json','COMPLETE'}
        files={str(path.relative_to(output)):digest(path) for path in sorted(output.rglob('*'))
               if path.is_file() and path.name not in excluded and not any(p.startswith('.') for p in path.relative_to(output).parts)}
        write_json(output/'manifest.json',{'files':files,'metadata':{'stage':name}})
        atomic_text(output/'COMPLETE',digest(output/'manifest.json')+'\n')
        print(f'[L1] {name} PASS',flush=True)
    except BaseException as exc:
        write_json(output/'result.json',{'status':'FAIL','started_at':started,'finished_at':now(),
                   'error':str(exc),'traceback':traceback.format_exc()})
        raise


def execute(root,config,run_id,resume=False,check_only=False,memory_guard=False):
    helper=l0_runner(root); helper.validate_run_id(run_id)
    with helper.exclusive_lock(root/'locks/l0.lock'):  # All L0/L1 GPU jobs share the same exclusive lock.
        run=root/'runs/L1'/run_id
        if resume:
            require(run.is_dir(),'Resume run not found')
            state=read_json(run/'summary.json'); provenance=read_json(run/'provenance.json')
            require(provenance['source_hashes']==source_hashes(root),'Source/config changed; start a new run')
            require(read_json(run/'resolved_config.json')==config,'Resolved config differs from requested resume')
            require(provenance['config_hash']==digest(run/'resolved_config.json'),'Resolved config integrity error')
            require(state['memory_guard_enabled']==memory_guard and state['check_only']==check_only,'Resume execution mode changed')
            require(state['status'] not in ['SMOKE_PASS','L1_QUALITY_PASS','READINESS_PASS','QUALITY_FAIL'], 'Run already completed')
        else:
            run.mkdir(parents=True,exist_ok=False)
            write_json(run/'resolved_config.json',config)
            provenance={'created_at':now(),'python':sys.executable,'source_hashes':source_hashes(root),
                        'config_hash':digest(run/'resolved_config.json'),'memory_guard_enabled':memory_guard,
                        'argv':sys.argv,'openjev_commit':(root/'locks/openjev.commit').read_text().strip()}
            write_json(run/'provenance.json',provenance); write_json(root/'locks/L1'/f'{run_id}.json',provenance)
            selected=[STAGES[0],('99_readiness','Implementation tests; no L1 training')] if check_only else STAGES
            state={'run_id':run_id,'started_at':now(),'profile':config['profile'],'status':'RUNNING',
                   'memory_guard_enabled':memory_guard,'check_only':check_only,
                   'stages':[{'name':name,'title':title,'status':'PENDING'} for name,title in selected]}
        state['status']='RUNNING'; state.pop('error',None); update(run,state)
        try:
            for item in state['stages']:
                name=item['name']; output=run/name
                if item['status']=='PASS':
                    verify(output)
                    result=read_json(output/'result.json')
                    for relative,expected in result.get('external_artifacts',{}).items():
                        require(digest(run/relative)==expected,'External stage artifact changed')
                    continue
                if output.exists():
                    # Preserve failed attempt's logs. Training checkpoints stay for exact optimizer restart.
                    archive=output/'attempts'/now().replace(':','-'); archive.mkdir(parents=True)
                    for old in ['console.log','command.json','resources.jsonl','guard.json','status.json','result.json','manifest.json','COMPLETE']:
                        if (output/old).exists(): shutil.move(output/old,archive/old)
                output.mkdir(exist_ok=True)
                item.update({'status':'RUNNING'}); update(run,state)
                worker=[sys.executable,'-u',str(root/'stages/L1'/name/'run.py'),'--root',str(root),'--run-dir',str(run)]
                poc=config['poc']; timing={'guard_interval_s':1.,'guard_grace_s':60.}
                command=helper.guard_command(root,output,worker,poc,timing,memory_guard=memory_guard)
                write_json(output/'command.json',{'argv':command,'memory_guard_enabled':memory_guard})
                env=dict(os.environ); env['HF_HUB_OFFLINE']='1'; env['PYTHONPATH']=str(root/'src')
                start=time.monotonic(); print(f'\n[L1] {name}: {item["title"]}\nOutput: {output}',flush=True)
                code,interrupted=helper.run_child(command,output/'console.log',env)
                item.update({'seconds':time.monotonic()-start,'exit_code':code})
                require(code==0,f'{name} exited {code}; inspect {output}/console.log and guard.json')
                require(read_json(output/'guard.json')['status']=='PASS','Stage monitor failed')
                require(read_json(output/'result.json')['status']=='PASS','Stage did not produce PASS')
                verify(output); item['status']='PASS'; write_json(output/'status.json',item); update(run,state)
            if check_only: state['status']='READINESS_PASS'
            else:
                quality=read_json(run/'13_report/result.json')['quality_pass']; state['quality_gates_passed']=quality
                if any(name=='21_report' for name,_ in STAGES):
                    extended=read_json(run/'21_report/result.json')
                    require(extended['extended_validation_completed'],'PV1--PV4 incomplete')
                    state.update(extended_validation_completed=True,pv_findings=extended['findings'],
                                 all_hypotheses_supported=extended['all_hypotheses_supported'])
                state['status']='SMOKE_PASS' if config['profile']=='smoke' else ('L1_QUALITY_PASS' if quality else 'QUALITY_FAIL')
            state['finished_at']=now(); update(run,state)
            print(f'\n[L1] {state["status"]}\nSummary: {run/"summary.md"}',flush=True)
            return 2 if state['status']=='QUALITY_FAIL' else 0
        except BaseException as exc:
            for item in state['stages']:
                if item['status']=='RUNNING': item['status']='FAIL'
                elif item['status']=='PENDING': item['status']='NOT_RUN'
            state.update({'status':'FAIL','finished_at':now(),'error':str(exc)}); update(run,state)
            print(f'[L1] FAIL: {exc}\nResume after diagnosis: bash run_l1.sh --resume {run_id}',file=sys.stderr,flush=True)
            return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--config',type=Path)
    p.add_argument('--profile',choices=['smoke','pilot','core']); p.add_argument('--seed',type=int)
    p.add_argument('--run-id'); p.add_argument('--resume',metavar='RUN_ID')
    p.add_argument('--check-only',action='store_true'); p.add_argument('--plan',action='store_true')
    guard=p.add_mutually_exclusive_group()
    guard.add_argument('--memory-guard',dest='memory_guard',action='store_true',default=None)
    guard.add_argument('--no-memory-guard',dest='memory_guard',action='store_false')
    args=p.parse_args(); root=args.root.resolve()
    require(args.config is None or args.config.resolve()==root/'configs/poc.yaml',
            'Use the pinned poc.yaml; L1 budgets are configured in configs/l1.yaml')
    if args.resume:
        require(args.run_id is None,'Choose --resume or --run-id')
        old=read_json(root/'runs/L1'/args.resume/'resolved_config.json')
        profile=args.profile or old['profile']; seed=old['seed'] if args.seed is None else args.seed
    else: profile=args.profile; seed=2026 if args.seed is None else args.seed
    config=configuration(root,profile,seed)
    saved_state=read_json(root/'runs/L1'/args.resume/'summary.json') if args.resume else None
    memory_guard=(saved_state['memory_guard_enabled'] if saved_state else config['l1']['memory_guard']) if args.memory_guard is None else args.memory_guard
    if args.plan:
        print(f'Profile={config["profile"]}, seed={seed}, memory_guard={memory_guard}')
        print(json_budget(config['budget']))
        print('PV1--PV4: '+json_budget(config['validation']))
        for name,title in STAGES: print(f'{name}: {title}')
        return 0
    run_id=args.resume or args.run_id or f'l1_{config["profile"]}_s{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    return execute(root,config,run_id,bool(args.resume),args.check_only or bool(saved_state and saved_state['check_only']),memory_guard)


def json_budget(value):
    import json
    return json.dumps(value,ensure_ascii=False)


if __name__=='__main__':
    raise SystemExit(main())
