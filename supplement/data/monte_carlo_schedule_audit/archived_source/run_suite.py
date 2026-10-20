#!/usr/bin/env python3
"""Independent empirical latency Monte Carlo; launch manually on the original H200."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid
from validation_io import (bundle, completed_case, read_json, require, sha256, under, write_json)
from runtime import utc, environment_check, environment_identity, experiment_lock
from plan import design, DEFAULT_REPLICATES, DEFAULT_SEED

HERE = Path(__file__).resolve().parent
HARNESS_FILES = ('run_suite.py','worker.py','runtime.py','plan.py','scheduler.py','latency_replay.py','metric_audit.py','step_plant.py',
                 'validation_io.py','baseline_check.py','analyze_results.py','reference.json',
                 'run_validation.sh','README_KO.md','PROTOCOL.md','test_validation.py','legacy_scheduler_reference.py',
                 'mc_statistics.py','LICENSE','DATA_LICENSE')


def inventory(root, reference):
    failures=[]; count=total=0
    for name, expected in reference['files'].items():
        try:
            path=under(root,name)
            actual=sha256(path); count+=1; total+=path.stat().st_size
            if actual!=expected: failures.append({'file':name,'error':'SHA256 mismatch'})
        except (OSError,ValueError) as exc:
            failures.append({'file':name,'error':str(exc)})
    for key,source in reference['sources'].items():
        base=under(root,source['source_root_rel'])
        found={p.relative_to(base).as_posix() for p in (base/'src').rglob('*.py')}
        if found!=set(source['files']):
            failures.append({'source':key,'changed_file_set':sorted(found^set(source['files']))})
    return {'passed':not failures,'checked_files':count,'checked_bytes':total,'failures':failures,
            'time_utc':utc(),'scope':'read-only input hashes; no inference or simulation'}


def prepare_output(root, output, reference, resume, env, frozen_design):
    # Require a new leaf directory, never an ancestor/descendant of frozen inputs.
    protected=[HERE,root/'src',root/'models',root/'cache',root/'artifacts',
               root/reference['D_run'],root/'third_party',root/'stages']
    for p in protected:
        p=p.resolve()
        require(not output.is_relative_to(p) and not p.is_relative_to(output),f'Output overlaps protected path: {p}')
    protocol={'root':str(root),'design':frozen_design,'environment_identity':environment_identity(env),
              'harness_sha256':{name:sha256(HERE/name) for name in HARNESS_FILES}}
    if output.exists():
        require(resume,f'Output exists; use --resume --output {output}')
        require(read_json(output/'protocol.json')==protocol,'Resume protocol, environment or harness changed; use a new output directory')
        for name,digest in protocol['harness_sha256'].items():
            require(sha256(output/'harness'/name)==digest,f'Copied harness changed: {name}')
        for key,source in reference['sources'].items():
            for name,digest in source['files'].items():
                require(sha256(output/'frozen_source'/key/name)==digest,f'Copied source changed: {key}/{name}')
        return
    require(not resume,'--resume requires an existing output directory')
    output.mkdir(parents=True)
    (output/'harness').mkdir()
    for name in HARNESS_FILES: shutil.copyfile(HERE/name,output/'harness'/name)
    for key,source in reference['sources'].items():
        for name,digest in source['files'].items():
            target=output/'frozen_source'/key/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/source['source_root_rel']/name,target)
            require(sha256(target)==digest,'Source changed while copying')
    write_json(output/'protocol.json',protocol)


def evaluate_gate(output, phase):
    protocol=read_json(output/'protocol.json'); ph=sha256(output/'protocol.json'); checks={}
    for job in protocol['design']['jobs']:
        if job['phase']!=phase or job['h_ms']!=10.: continue
        for case in job['cases']:
            folder=completed_case(output,case['key'],ph)
            checks[case['key']]=(read_json(folder/'baseline_comparison.json') if folder else
                                 {'passed':False,'reason':'Missing completed baseline'})
    expected=240
    result={'passed':len(checks)==expected and all(c['passed'] for c in checks.values()),
            'expected_cases':expected,'checks':checks,'time_utc':utc()}
    write_json(output/'D_baseline_gate.json',result)
    return result


def run_worker(root, output, job, session):
    cmd=[sys.executable,'-B','-u',str(output/'harness/run_suite.py'),'--root',str(root),
         '--output',str(output),'--worker-job',job['id']]
    env=dict(os.environ,JEV_REPLAY_SUPERVISOR_PID=str(os.getpid()))
    log=session/(job['id']+'.log')
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                 encoding='utf-8',errors='replace',env=env)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            raise
    require(code==0,f'Worker failed ({code}); inspect {log}')


def supervise(args):
    reference=read_json(HERE/'reference.json')
    replicas=args.replicates if args.replicates is not None else DEFAULT_REPLICATES
    seed=args.seed if args.seed is not None else DEFAULT_SEED
    if args.resume and args.output and (args.output/'protocol.json').exists():
        old=read_json(args.output/'protocol.json')['design']
        replicas=old['replicates'] if args.replicates is None else replicas
        seed=old['seed'] if args.seed is None else seed
    frozen_design=design(reference, replicas, seed)
    if args.plan:
        print(json.dumps(frozen_design,indent=2));return 0
    require(args.root is not None,'--root is required')
    root=args.root.resolve()
    if args.inventory_only:
        result=inventory(root,reference);print(json.dumps(result,indent=2));return 0 if result['passed'] else 2
    require(not args.resume or args.output is not None,'--resume requires explicit --output')
    output=(args.output or root/'runs/L2_diagnostics'/('latency_monte_carlo_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    with experiment_lock(root):
        print(f'Checking {len(reference["files"])} pinned input files...',flush=True)
        before=inventory(root,reference)
        require(before['passed'],json.dumps(before,indent=2))
        env=environment_check(root)
        prepare_output(root,output,reference,args.resume,env,frozen_design)
        session=output/'sessions'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        write_json(session/'inventory_before.json',before);write_json(session/'environment.json',env)
        ph=sha256(output/'protocol.json')
        audit={'output':str(output),'session':str(session),'protocol_sha256':ph,'environment':env,
               'sources':reference['sources'],'actuation_enabled':False,'phase':args.phase}
        write_json(root/'locks/latency_monte_carlo'/f'{session.name}.json',audit)
        log_record=root/'logs/L2/latency_monte_carlo'/f'{session.name}.json'
        write_json(log_record,{'state':'PREFLIGHT_PASS','time_utc':utc(),**audit})
        print(f'Output: {output}',flush=True)
        if args.check_only:
            write_json(output/'STATUS.json',{'state':'PREFLIGHT_PASS','time_utc':utc()})
            print(f'No simulation run. Continue with --resume --output "{output}"');return 0
        error=None; state='PHASE_COMPLETE'
        jobs=frozen_design['jobs']
        try:
            for job in jobs:
                if args.phase!='all' and job['phase']!=args.phase: continue
                if job['phase']!='baseline':
                    require(evaluate_gate(output,'baseline')['passed'],'All 240 D 10-ms reproductions must pass before Monte Carlo comparisons')
                folders=[completed_case(output,c['key'],ph) for c in job['cases']]
                if all(folders):
                    for f in folders:
                        if (f/'baseline_comparison.json').exists():
                            require(read_json(f/'baseline_comparison.json')['passed'],f'Sealed baseline failure: {f}')
                    print(f"[{job['id']}] all completed cases verified; retained",flush=True)
                else:
                    run_worker(root,output,job,session)
            if args.phase in ('all','baseline'):
                require(evaluate_gate(output,'baseline')['passed'],'D baseline gate failed')
            complete=sum(completed_case(output,c['key'],ph) is not None for j in jobs for c in j['cases'])
            if complete==frozen_design['total_runs'] and evaluate_gate(output,'baseline')['passed']:
                state='EXECUTION_COMPLETE'
        except BaseException as exc:
            state='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'FAILED'
            error={'message':repr(exc),'traceback':traceback.format_exc()}
        try:
            after=inventory(root,reference)
            write_json(session/'inventory_after.json',after);write_json(output/'INPUT_INTEGRITY.json',after)
            require(after['passed'],'Archived inputs changed during execution')
        except Exception as exc:
            state='FAILED';error={'message':repr(exc),'prior_error':error}
        write_json(output/'STATUS.json',{'state':state,'error':error,'time_utc':utc(),'phase':args.phase})
        try:
            from analyze_results import analyze
            result=analyze(output)
            require(result['independent_metric_audit_passed'],'Independent metric reconstruction failed')
            if state=='EXECUTION_COMPLETE':
                require(result['state']=='COMPLETE_VERIFIED','Execution count and independent audit completeness differ')
        except Exception as exc:
            state='FAILED';error={'message':repr(exc),'prior_error':error,'traceback':traceback.format_exc()}
            write_json(output/'analysis_error.json',error)
            write_json(output/'STATUS.json',{'state':state,'error':error,'time_utc':utc()})
        archive=bundle(output)
        write_json(log_record,{'state':state,'time_utc':utc(),'archive':str(archive),'error':error,**audit})
        print(f'{state}\nResults: {output}\nReturn this ZIP: {archive}',flush=True)
        if error: print(error['message'],file=sys.stderr)
        return 0 if state in ('EXECUTION_COMPLETE','PHASE_COMPLETE') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--replicates',type=int,help='Freeze before execution; default 16, >=64 recommended for precision')
    parser.add_argument('--seed',type=int,help='Freeze before execution; default 20261013')
    parser.add_argument('--phase',choices=('all','baseline','monte_carlo'),default='all')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true',help='Print frozen design; no model imports')
    mode.add_argument('--check-only',action='store_true',help='Verify files/environment; no simulation')
    mode.add_argument('--inventory-only',action='store_true',help='Read-only file inventory; no GPU required')
    parser.add_argument('--worker-job',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_job:
        require(args.root is not None and args.output is not None,'Worker paths required')
        from worker import run
        run(args.root.resolve(),args.output.resolve(),args.worker_job);return 0
    return supervise(args)


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,RuntimeError) as exc:
        print(f'STOP: {exc}',file=sys.stderr);raise SystemExit(2)
