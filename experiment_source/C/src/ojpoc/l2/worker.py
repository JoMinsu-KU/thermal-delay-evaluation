"""One GPU model per fresh worker process."""
import argparse
from pathlib import Path
import time
import numpy as np
from ..common import read_json,write_json,seal,setup_torch,digest
from .runner import verify_run,TRAIN_STAGES


def work(args,cfg):
    root=args.root;run=args.run_dir;output=args.output;method=args.method
    if args.job=='parity':
        from .models import parity
        return parity(root,run,cfg,output)
    if args.job=='early_speed':
        from .engine import Engine
        from .models import load_policy
        from .contracts import entries,anchors
        from ..data import RawEpisode
        setup_torch(cfg);model=load_policy(root,run,'shared_physics',training=True)
        model.backbone.gradient_checkpointing_disable();model.eval();model.requires_grad_(False)
        if cfg['l2'].get('inference_backend','eager')=='cuda_graph':
            from .graph import GraphPolicy
            model=GraphPolicy(model).eval()
        engine=Engine(root,run,'shared_physics',calibrated=False,policy_override=model)
        entry=entries(run,['validation'])[0];raw=RawEpisode(run/'02_generate/data/raw'/entry['episode_id'])
        step=int(anchors(cfg)[-1]);values=[]
        for i in range(25):
            row=engine.decide(lambda:raw.snapshot(step))
            if i>=5:values.append(row)
        raw.close();engine.close()
        result={'requests':20,'warmup':5,'e2e_ms':{k:float(np.quantile([r['e2e_ms'] for r in values],q)) for k,q in [('p50',.5),('p95',.95),('p99',.99)]},
                'path':values[-1]['fallback_reason'],'valid_tokens':values[-1]['tokens'],
                'scope':'untrained shared wrapper full path; readiness/timing only, not final latency evidence'}
        write_json(output/'early_speed.json',result);return result
    if args.job=='resume_probe':
        from .training import resume_probe
        return resume_probe(root,run,cfg,output)
    if args.job=='reload':
        from .training import reload_check
        return reload_check(root,run,cfg,output,method,next(s for s,m in TRAIN_STAGES.items() if m==method))
    if args.job=='validation_control':
        from .validation_control import audit_method
        return audit_method(root,run,cfg,output,method)
    if args.job=='margins':
        from .calibration import margins
        return margins(run,cfg,output)
    if args.job=='calibrate':
        from .calibration import calibrate_method
        return calibrate_method(root,run,cfg,output,method)
    if args.job=='test_cache':
        from .evaluation import prepare_test
        return prepare_test(run,cfg,output)
    if args.job=='evaluate':
        from .evaluation import evaluate_method
        return evaluate_method(root,run,cfg,output,method)
    if args.job=='rings':
        from .latency import prepare_rings
        return prepare_rings(root,run,cfg,output)
    if args.job=='latency':
        from .latency import trial
        return trial(root,run,cfg,output/'trials'/f'{method}_{args.repeat}',method,args.repeat,args.launched_ns)
    if args.job in ['zero','delayed']:
        from .control import run_method
        return run_method(root,run,cfg,output,method,args.job)
    if args.job=='operators':
        from .report import operator_validation
        return operator_validation(run,cfg,output)
    raise ValueError('Unknown worker job')


def main():
    p=argparse.ArgumentParser()
    for key in ['root','run-dir','output','record']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--job',required=True);p.add_argument('--method');p.add_argument('--repeat',type=int);p.add_argument('--launched-ns',type=int,required=True)
    args=p.parse_args();cfg=verify_run(args.root,args.run_dir);start=time.perf_counter()
    result=work(args,cfg)
    # Seal outputs owned by this worker; completed jobs are verifiable on stage resume.
    output=args.output;method=args.method;job=args.job
    owned=[]
    if job=='latency':owned=[output/'trials'/f'{method}_{args.repeat}']
    elif job in ['zero','delayed']:owned=[output/method,output/(method+'.json')]
    elif job=='evaluate':owned=[output/(method+'.jsonl'),output/(method+'.summary.json')]
    elif job=='calibrate':owned=[output/(method+'.json')]
    elif job=='margins':owned=[output/'margins.json',output/'cache']
    elif job=='test_cache':owned=[output/'cache']
    elif job=='rings':owned=[output/'rings']
    elif job=='resume_probe':owned=[output/'resume_probe.pt',output/'resume_probe.sha256',output/'resume_probe.json']
    elif job=='early_speed':owned=[output/'early_speed.json']
    elif job=='validation_control':owned=[output/('validation_control_'+method+'.json')]
    elif job=='operators':owned=list(output.glob('*_rows.json'))+[output/'operator_validation.json']
    files=[p for item in owned for p in (list(item.rglob('*')) if item.is_dir() else [item]) if p.is_file()]
    write_json(args.record/'result.json',{'job':job,'method':method,'seconds':time.perf_counter()-start,'result':result,
                'artifact_hashes':{str(p.relative_to(args.run_dir)):digest(p) for p in files}})
    seal(args.record,{'fresh_process':True})

if __name__=='__main__':main()
