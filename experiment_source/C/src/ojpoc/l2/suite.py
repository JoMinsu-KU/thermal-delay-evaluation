"""Small training/evaluation first; promote to full core only after an explicit readiness PASS."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import uuid
from ..common import ROOT,read_json,write_json,atomic_text,require
from ..cli import l0_runner,now
from . import runner
from .contracts import configuration,source_hashes,TRAIN_METHODS


def promotion(run,require_quality=False):
    summary=read_json(run/'summary.json')
    if summary['status']!='L2_COMPLETE' or summary['profile']!='pilot':
        return {'pass':False,'checks':{'pilot_completed':False},'quality_required':require_quality}
    report=read_json(run/'22_report/validation_summary.json');checks={}
    checks.update({k:bool(report['gates'][k]) for k in ['G0_environment','G1_data','G2_observer']})
    checks['resources_within_quota']=report['resources']['observed_within_quota']
    checks['optimizer_resume']=report['resume_probe']['passed']
    checks['all_stages_complete']=len(summary['stages'])==len(runner.STAGES) and all(s['status']=='PASS' for s in summary['stages'])
    for stage,method in runner.TRAIN_STAGES.items():
        t=read_json(run/stage/'training.json')
        checks[method+'_finite_updates_and_frozen_base']=bool(t['frozen_base_unchanged'] and t['history'] and
                all(r['new_gradient']>0 and math.isfinite(r['loss']) and math.isfinite(r['new_gradient']) for r in t['history']) and
                (method=='small_physics' or all(r['lora_gradient']>0 for r in t['history'] if r['phase']=='adapt')))
        checks[method+'_selection_nonworsening']=t['selected']['selection_score']<=t['baseline']['selection_score']+1e-12
    checks['inference_errors_zero']=all(v['failures']==0 for repeats in report['latency'].values() for repeat in repeats for v in repeat['by_path'].values())
    quality={k:bool(v) for k,v in report['gates'].items() if not k.startswith('G7')}
    if require_quality:checks.update({'quality_'+k:v for k,v in quality.items()})
    return {'pass':all(checks.values()),'checks':checks,'quality_required':require_quality,'quality_gates':quality,
            'definition':'execution/training/reload/resume/resource readiness; effect and deadline quality are separate unless --require-quality',
            'not_a_claim_of_L2_scientific_success':True}


def update(suite,state):
    write_json(suite/'summary.json',state)
    lines=['# L2 automatic suite','',f'Suite: {suite.name}',f'Status: {state["status"]}',
           f'Pilot: {state["pilot_run"]}',f'Core: {state["core_run"]}',
           '', 'Pilot readiness PASS precedes core. Scientific quality is reported separately.']
    if state.get('promotion'):
        lines+=['','| Readiness check | Pass |','|---|---|',*[f'| {k} | {v} |' for k,v in state['promotion']['checks'].items()]]
    if state.get('error'):lines+=['',state['error']]
    atomic_text(suite/'summary.md','\n'.join(lines)+'\n')


def execute_suite(root,suite_id,resume=False,memory_guard=False,require_quality=False,config_file=None):
    helper=l0_runner(root);helper.validate_run_id(suite_id);suite=root/'runs/L2_suites'/suite_id
    with helper.exclusive_lock(root/'locks/l2_suite.lock'):
        if resume:
            state=read_json(suite/'summary.json')
            require(state['source_hashes']==source_hashes(root),'Source changed; start a new suite')
            require(state['memory_guard_enabled']==memory_guard and state['require_quality']==require_quality,'Resume promotion/guard policy changed')
            require(state['status']!='SUITE_COMPLETE','Suite already completed')
            require(config_file is None or str(config_file)==state.get('config_file','configs/l2.yaml'),'Resume config changed')
            config_file=state.get('config_file','configs/l2.yaml')
        else:
            suite.mkdir(parents=True,exist_ok=False)
            state={'suite_id':suite_id,'status':'RUNNING','started_at':now(),'source_hashes':source_hashes(root),
                   'pilot_run':suite_id+'_pilot','core_run':suite_id+'_core','memory_guard_enabled':memory_guard,'require_quality':require_quality,
                   'config_file':str(config_file or 'configs/l2.yaml')}
        state['status']='RUNNING';state.pop('error',None);update(suite,state)
        try:
            for profile in ['pilot','core']:
                require(state['source_hashes']==source_hashes(root),'Source/config changed during suite; start a new suite')
                if profile=='core':
                    pilot=root/'runs/L2'/state['pilot_run'];decision=promotion(pilot,require_quality)
                    state['promotion']=decision;update(suite,state)
                    if not decision['pass']:
                        state.update(status='PILOT_NOT_READY',finished_at=now());update(suite,state)
                        print('L2 pilot readiness FAIL; core was not started. Inspect '+str(suite/'summary.md'),flush=True);return 2
                    unmet=[k for k,v in decision.get('quality_gates',{}).items() if not v]
                    if unmet:print('Pilot scientific quality gates not met: '+', '.join(unmet)+'; promotion uses the preselected readiness policy.',flush=True)
                    print('L2 pilot readiness PASS: starting full core training and evaluation.',flush=True)
                run_id=state[profile+'_run'];run=root/'runs/L2'/run_id;cfg=configuration(root,profile,config_file)
                already=run.exists()
                if already:
                    require(runner.verify_run(root,run)==cfg,'Suite child configuration changed')
                    saved=read_json(run/'summary.json')
                    if saved['status']=='L2_COMPLETE':
                        for stage in saved['stages']:runner.verify_stage(run,stage['name'])
                        continue
                state['active_profile']=profile;update(suite,state)
                code=runner.execute(root,cfg,run_id,resume=already,memory_guard=memory_guard)
                require(code==0,f'{profile} execution failed; full core cannot proceed')
            state.update(status='SUITE_COMPLETE',finished_at=now());state.pop('active_profile',None);update(suite,state)
            print('L2 suite complete: '+str(root/'runs/L2'/state['core_run']/'22_report/report.md'),flush=True);return 0
        except BaseException as exc:
            state.update(status='SUITE_FAILED',error=str(exc),finished_at=now());update(suite,state)
            print(f'L2 suite stopped: {exc}\nResume: bash run_l2.sh --resume-suite {suite_id}',file=sys.stderr,flush=True);return 1


def main():
    argv=sys.argv[1:]
    # Retain explicit single-profile and single-run resume/readiness commands.
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--profile',choices=['auto','smoke','pilot','core'],default='auto')
    selected,_=pre.parse_known_args(argv)
    if any(a=='--resume' or a.startswith('--resume=') for a in argv) or '--check-only' in argv or selected.profile!='auto':
        return runner.main()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--profile',choices=['auto'],default='auto');p.add_argument('--suite-id');p.add_argument('--resume-suite')
    p.add_argument('--require-quality',action='store_true',default=None);p.add_argument('--plan',action='store_true')
    p.add_argument('--config',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--memory-guard',dest='memory_guard',action='store_true',default=None);g.add_argument('--no-memory-guard',dest='memory_guard',action='store_false')
    args=p.parse_args();root=args.root.resolve()
    if args.resume_suite:
        require(args.suite_id is None,'Use --suite-id or --resume-suite');l0_runner(root).validate_run_id(args.resume_suite)
        state=read_json(root/'runs/L2_suites'/args.resume_suite/'summary.json')
        if args.config is None:args.config=Path(state.get('config_file','configs/l2.yaml'))
        memory=state['memory_guard_enabled'] if args.memory_guard is None else args.memory_guard
        quality=state['require_quality'] if args.require_quality is None else args.require_quality
    else:
        memory=bool(args.memory_guard)
        quality=configuration(root,'pilot',args.config)['l2'].get('require_quality_default',False) if args.require_quality is None else args.require_quality
    if args.plan:
        print(json.dumps({'workflow':['pilot','readiness PASS','core'],'quality_required_for_promotion':quality,
                          'pilot':configuration(root,'pilot',args.config)['budget'],'core':configuration(root,'core',args.config)['budget'],
                          'failure_behavior':'No core launch if execution/readiness fails; --require-quality also enforces G3-G6'},indent=2));return 0
    suite_id=args.resume_suite or args.suite_id or f'l2_auto_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    return execute_suite(root,suite_id,bool(args.resume_suite),memory,quality,args.config)

if __name__=='__main__':raise SystemExit(main())
