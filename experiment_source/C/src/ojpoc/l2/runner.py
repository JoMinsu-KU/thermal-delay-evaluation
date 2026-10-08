"""Sequential, immutable L2 experiment orchestration."""
import argparse
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
from ..common import ROOT, require, read_json, write_json, atomic_text, digest, verify
from ..cli import l0_runner, now
from ..recheck import validate_parent, parent_fingerprint
from .contracts import configuration, source_hashes, TRAIN_METHODS, ABLATIONS, assert_frozen

TRAIN_STAGES={f'{i:02d}_{m}':m for i,m in enumerate(TRAIN_METHODS,7)}
STAGES=[('00_preflight','Environment, original specification and completed L1 verification'),
        ('01_frozen_inputs','Copy validation-selected L1 operators with hashes'),
        ('02_generate','Generate disjoint train/validation/calibration/test/OOD episodes'),
        ('03_index','Independent solver, causality, train-only normalization'),
        ('04_observer','Supervised O-small observer and validation fallback'),
        ('05_cache','Modality-specific FV labels and frozen PINO features'),
        ('06_parity','Embedding, gradient, permutation and early full-path timing'),
        *[(s,'Train '+m) for s,m in TRAIN_STAGES.items()],
        ('14_reload','Fresh-process reload parity for all seven trained policies'),
        ('15_freeze','Freeze selected models and preprocessing before calibration'),
        ('16_calibration','Disjoint physics-margin and probability calibration'),
        ('17_evaluation','Sealed test/OOD decision regret and probability calibration'),
        ('18_latency','Fresh-process repeated, path-conditional full E2E timing'),
        ('19_closed_loop','Paired zero-compute-delay test/OOD control'),
        ('20_delayed_loop','Measured-trace delayed control with strict fallback matching'),
        ('21_ablation','Operator and physics/modality diagnostics'),
        ('22_report','Gates, episode bootstrap, resource use and final report')]


def parent(root,cfg):
    p=(root/'runs/L1'/cfg['l2']['parent_run']).resolve()
    require(p.parent==(root/'runs/L1').resolve(),'Invalid parent location')
    return p


def verify_run(root,run):
    require(run.resolve().parent==(root/'runs/L2').resolve(),'Invalid L2 run directory')
    prov=read_json(run/'provenance.json')
    require(prov['source_hashes']==source_hashes(root),'Source/config changed; start a new L2 run')
    require(prov['config_hash']==digest(run/'resolved_config.json'),'Resolved config changed')
    cfg=read_json(run/'resolved_config.json')
    require(prov['parent_fingerprint']==parent_fingerprint(parent(root,cfg)),'Parent lineage changed')
    if (run/'15_freeze/COMPLETE').exists():assert_frozen(run)
    if (run/'16_calibration/COMPLETE').exists():
        verify(run/'16_calibration')
    return cfg


def verify_stage(run,name):
    verify(run/name)
    for path,sha in read_json(run/name/'result.json').get('external_artifacts',{}).items():
        require(digest(run/path)==sha,'External stage artifact changed: '+path)


def child(root,run,output,job,method=None,repeat=None):
    key=job+('_'+method if method else '')+('_'+str(repeat) if repeat is not None else '')
    folder=output/'workers'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE').exists():
        verify(folder);saved=read_json(folder/'result.json')
        for relative,sha in saved.get('artifact_hashes',{}).items():require(digest(run/relative)==sha,'Worker output changed: '+relative)
        return saved
    argv=[sys.executable,'-u','-m','ojpoc.l2.worker','--root',str(root),'--run-dir',str(run),
          '--output',str(output),'--job',job,'--record',str(folder),'--launched-ns',str(time.perf_counter_ns())]
    if method:argv+=['--method',method]
    if repeat is not None:argv+=['--repeat',str(repeat)]
    print('L2 isolated worker '+key,flush=True)
    require(subprocess.call(argv,cwd=root)==0,'Worker failed: '+key)
    verify(folder);return read_json(folder/'result.json')


def execute_stage(name,root,run,cfg,output):
    if name=='00_preflight':
        from ..stages import preflight
        require(subprocess.call(['sha256sum','-c','SHA256SUMS.txt'],cwd=root)==0,'Original specification files changed')
        return {**preflight(root,run,cfg,output),'parent':validate_parent(root,parent(root,cfg))}
    if name=='01_frozen_inputs':
        files={}
        for stage in ['05_fno_data','06_pino_soft','07_pino_balance']:
            src=parent(root,cfg)/stage/'best.pt';dest=output/stage/'best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(src,dest)
            require(digest(src)==digest(dest),'Transferred checkpoint changed')
            files[str(dest.relative_to(run))]=digest(dest)
        return {'external_artifacts':files,'operator_selection':'L1 validation best; frozen transfer, no L2 test selection'}
    if name=='02_generate':
        from .data import generate
        return generate(run,cfg,output)
    if name=='03_index':
        from ..stages import solver_acceptance
        from .data import causal_index
        return {'solver':solver_acceptance(cfg),'data':causal_index(run,cfg,output)}
    if name=='04_observer':
        from .observer import train
        return train(run,cfg,output)
    if name=='05_cache':
        from .cache import prepare
        return prepare(run,cfg,output)
    if name=='06_parity':
        result=child(root,run,output,'parity')
        result['speed_probe']=child(root,run,output,'early_speed')
        return result
    if name in TRAIN_STAGES:
        from .training import train
        result=train(root,run,cfg,output,TRAIN_STAGES[name])
        result['external_artifacts']={str(p.relative_to(run)):digest(p) for p in (run/'models'/TRAIN_STAGES[name]).rglob('*') if p.is_file()}
        return result
    if name=='14_reload':
        result={'models':[child(root,run,output,'reload',m) for m in TRAIN_METHODS], 'resume':child(root,run,output,'resume_probe')}
        if cfg['l2'].get('validation_control_audit',False):
            for method in ['shared_physics','pi','pino_mpc']:child(root,run,output,'validation_control',method)
            from .validation_control import decide
            result['validation_control']=decide(output,cfg)
            # Smoke is plumbing only; pilot/core must demonstrate repeated control.
            require(cfg['profile']=='smoke' or result['validation_control']['passed'],
                    'Validation closed-loop audit failed before calibration/test; inspect 14_reload/validation_control_gate.json')
        return result
    if name=='15_freeze':
        paths=[run/'resolved_config.json',run/'03_index/normalization.json',run/'04_observer/selection.json',run/'04_observer/best.pt']
        paths+=list((run/'01_frozen_inputs').glob('*/best.pt'))
        for method in TRAIN_METHODS:
            verify(run/'models'/method/'best');paths+=list((run/'models'/method/'best').glob('*'))
        files={str(p.relative_to(run)):digest(p) for p in sorted(paths) if p.is_file()}
        result={'files':files,'selection_split':'validation','frozen_before_calibration_and_test':True,
                'methods':cfg['l2']['methods']+ABLATIONS,'test_used_for_selection':False}
        write_json(output/'frozen.json',result);return result
    if name=='16_calibration':
        child(root,run,output,'margins')
        for method in cfg['l2']['methods']:child(root,run,output,'calibrate',method)
        from .calibration import finalize
        return finalize(run,cfg,output)
    if name=='17_evaluation':
        child(root,run,output,'test_cache')
        for method in cfg['l2']['methods']+ABLATIONS:child(root,run,output,'evaluate',method)
        return {'evaluated_methods':cfg['l2']['methods']+ABLATIONS,'calibration_frozen':True}
    if name=='18_latency':
        child(root,run,output,'rings')
        for method in cfg['l2']['methods']+ABLATIONS:
            for repeat in range(cfg['budget']['latency_repeats']):child(root,run,output,'latency',method,repeat)
        return {'methods':cfg['l2']['methods']+ABLATIONS,'fresh_process_repeats':cfg['budget']['latency_repeats']}
    if name in ['19_closed_loop','20_delayed_loop']:
        for method in cfg['l2']['methods']+ABLATIONS:child(root,run,output,'zero' if name=='19_closed_loop' else 'delayed',method)
        return {'methods':cfg['l2']['methods']+ABLATIONS,'test_ood_reported_separately':True}
    if name=='21_ablation':
        result=child(root,run,output,'operators')
        write_json(output/'diagnostics.json',{'retrained_modality_controls':['shared_sensor','shared_thermal'],
                    'inference_physics_controls':ABLATIONS,'regret':'17_evaluation','zero_loop':'19_closed_loop','delayed_loop':'20_delayed_loop',
                    'labels_and_gate_shared_for_physics_arms':True,'modality_labels':'modality-consistent FV-MPC_obs; cross-modality regret is conditional on different observations','interpretation':'Inference ablations are distribution-shift diagnostics, not retrained causal estimates'})
        return result
    if name=='22_report':
        from .report import final_report
        return final_report(run,cfg,output)
    if name=='99_readiness':
        code=subprocess.call([sys.executable,'-m','pytest',str(root/'tests/l2'),'-m','not gpu','-q'],cwd=root)
        require(code==0,'L2 contract checks failed')
        return {'tests':'tests/l2 -m not gpu','experiment_executed':False}
    raise ValueError('Unknown L2 stage '+name)


def stage_main(name):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();run=args.run_dir.resolve();cfg=verify_run(root,run)
    output=run/name;output.mkdir(exist_ok=True);started=now()
    try:
        result=execute_stage(name,root,run,cfg,output)
        write_json(output/'result.json',{**(result or {}),'status':'PASS','started_at':started,'finished_at':now()})
        excluded={'console.log','command.json','resources.jsonl','guard.json','status.json','manifest.json','COMPLETE'}
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name not in excluded and
               'attempts' not in p.relative_to(output).parts and not any(s.startswith('.') for s in p.relative_to(output).parts)}
        write_json(output/'manifest.json',{'files':files,'metadata':{'stage':name,'schema':'thermal-l2-v1'}})
        atomic_text(output/'COMPLETE',digest(output/'manifest.json')+'\n');print('[L2] '+name+' PASS',flush=True)
    except BaseException as exc:
        write_json(output/'result.json',{'status':'FAIL','error':str(exc),'started_at':started,'traceback':traceback.format_exc()});raise


def update(root,run,state):
    write_json(run/'summary.json',state)
    lines=['# L2 execution','',f'Run: {run.name}',f'Status: {state["status"]}; profile: {state["profile"]}',
           '', '| Stage | Status | Seconds |','|---|---|---:|',
           *[f'| {i["name"]} | {i["status"]} | {i.get("seconds",0):.2f} |' for i in state['stages']],
           '', 'Execution completion does not imply scientific success. Report: 22_report/report.md.']
    if state.get('error'):lines+=['',state['error']]
    atomic_text(run/'summary.md','\n'.join(lines)+'\n')
    write_json(root/'logs/L2'/(run.name+'.json'),{'run_directory':str(run),'status':state['status']})


def execute(root,cfg,run_id,resume=False,check_only=False,memory_guard=False):
    helper=l0_runner(root);helper.validate_run_id(run_id)
    with helper.exclusive_lock(root/'locks/l0.lock'):
        run=root/'runs/L2'/run_id
        if resume:
            state=read_json(run/'summary.json');require(verify_run(root,run)==cfg,'Resume configuration changed')
            require(state['check_only']==check_only and state['memory_guard_enabled']==memory_guard,'Resume mode changed')
            require(state['status'] not in ['L2_COMPLETE','L2_SMOKE_PASS','READINESS_PASS'],'Run already completed')
        else:
            validate_parent(root,parent(root,cfg));run.mkdir(parents=True,exist_ok=False);write_json(run/'resolved_config.json',cfg)
            prov={'source_hashes':source_hashes(root),'config_hash':digest(run/'resolved_config.json'),
                  'parent_fingerprint':parent_fingerprint(parent(root,cfg)),'created_at':now(),'python':sys.executable,'argv':sys.argv}
            write_json(run/'provenance.json',prov);write_json(root/'locks/L2'/(run_id+'.json'),prov)
            selected=[STAGES[0],('99_readiness','CPU contract checks')] if check_only else STAGES
            state={'run_id':run_id,'profile':cfg['profile'],'check_only':check_only,'memory_guard_enabled':memory_guard,
                   'started_at':now(),'status':'RUNNING','stages':[{'name':n,'title':t,'status':'PENDING'} for n,t in selected]}
        state['status']='RUNNING';state.pop('error',None);update(root,run,state)
        try:
            for item in state['stages']:
                name=item['name'];output=run/name
                if item['status']=='PASS':verify_stage(run,name);continue
                if output.exists():
                    archive=output/'attempts'/now().replace(':','-');archive.mkdir(parents=True)
                    for old in ['console.log','command.json','resources.jsonl','guard.json','status.json','result.json','manifest.json','COMPLETE']:
                        if (output/old).exists():shutil.move(output/old,archive/old)
                output.mkdir(exist_ok=True);item['status']='RUNNING';update(root,run,state)
                worker=[sys.executable,'-u',str(root/'stages/L2'/name/'run.py'),'--root',str(root),'--run-dir',str(run)]
                command=helper.guard_command(root,output,worker,cfg['poc'],{'guard_interval_s':1.,'guard_grace_s':60.},memory_guard=memory_guard)
                write_json(output/'command.json',{'argv':command,'memory_guard_enabled':memory_guard})
                env=dict(os.environ);env['PYTHONPATH']=str(root/'src');env['HF_HUB_OFFLINE']='1'
                print(f'\n[L2] {name}: {item["title"]}\nOutput: {output}',flush=True);started=time.monotonic()
                code,_=helper.run_child(command,output/'console.log',env);item.update(seconds=time.monotonic()-started,exit_code=code)
                require(code==0,f'{name} exited {code}; inspect {output}/console.log')
                require(read_json(output/'guard.json')['status']=='PASS','Resource monitor failed')
                require(read_json(output/'result.json')['status']=='PASS','Stage result failed')
                verify_stage(run,name);item['status']='PASS';write_json(output/'status.json',item);update(root,run,state)
            state['status']='READINESS_PASS' if check_only else 'L2_SMOKE_PASS' if cfg['profile']=='smoke' else 'L2_COMPLETE'
            if not check_only:state['scientific_status']=read_json(run/'22_report/validation_summary.json')['scientific_status']
            state['finished_at']=now();update(root,run,state)
            report_path=run/"summary.md" if check_only else run/"22_report/report.md"
            print(f'\n[L2] {state["status"]}\nReport: {report_path}',flush=True);return 0
        except BaseException as exc:
            for item in state['stages']:
                if item['status']=='RUNNING':item['status']='FAIL'
                elif item['status']=='PENDING':item['status']='NOT_RUN'
            state.update(status='FAIL',error=str(exc),finished_at=now());update(root,run,state)
            print(f'[L2] FAIL: {exc}\nResume: bash run_l2.sh --resume {run_id}',file=sys.stderr,flush=True);return 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--profile',choices=['smoke','pilot','core']);p.add_argument('--run-id');p.add_argument('--resume')
    p.add_argument('--check-only',action='store_true');p.add_argument('--plan',action='store_true')
    p.add_argument('--config',type=Path)
    guard=p.add_mutually_exclusive_group();guard.add_argument('--memory-guard',dest='memory_guard',action='store_true',default=None)
    guard.add_argument('--no-memory-guard',dest='memory_guard',action='store_false');args=p.parse_args();root=args.root.resolve()
    if args.resume:
        require(args.run_id is None,'Use --resume or --run-id');l0_runner(root).validate_run_id(args.resume)
        run=root/'runs/L2'/args.resume;cfg=verify_run(root,run);state=read_json(run/'summary.json')
        require(args.profile is None or args.profile==cfg['profile'],'Resume profile changed')
        require(args.config is None or str(args.config)==cfg.get('config_source','configs/l2.yaml'),'Resume config changed')
        memory_guard=state['memory_guard_enabled'] if args.memory_guard is None else args.memory_guard
        check_only=state['check_only'] or args.check_only
    else:
        cfg=configuration(root,args.profile,args.config);memory_guard=cfg['l2']['memory_guard'] if args.memory_guard is None else args.memory_guard;check_only=args.check_only
    if args.plan:
        print(json.dumps({'profile':cfg['profile'],'seed':cfg['seed'],'budget':cfg['budget'],'methods':cfg['l2']['methods']+ABLATIONS},indent=2))
        for n,t in STAGES:print(n+': '+t)
        return 0
    run_id=args.resume or args.run_id or f'l2_{cfg["profile"]}_s{cfg["seed"]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    return execute(root,cfg,run_id,bool(args.resume),check_only,memory_guard)

if __name__=='__main__':raise SystemExit(main())
