"""One sequential source/step/method/mode group; original run remains read-only."""
from __future__ import annotations
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from validation_io import (completed_case, read_json, read_trace, read_latency, require,
                           seal, sha256, write_json)
from runtime import utc, environment_check, environment_identity
from plan import C_RUN, D_RUN, STAGES


def save_result(folder, plant, metrics, trace, fixed):
    import numpy as np
    write_json(folder / "metrics.json", metrics)
    with gzip.open(folder / "trace.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        for row in trace:
            if fixed and row["event"] not in ("apply", "deadline_fallback"):
                continue
            f.write(json.dumps(row, allow_nan=False) + "\n")
    with gzip.open(folder / "truth.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(plant.truth_rows[0]))
        writer.writeheader();writer.writerows(plant.truth_rows)
    np.save(folder / "final_T64.npy", plant.field, allow_pickle=False)
    q=plant.quadrature_rows
    np.savez_compressed(folder / "quadrature.npz",
        end_ns=np.asarray([x[0] for x in q], dtype=np.int64), dt_ns=np.asarray([x[1] for x in q], dtype=np.int32),
        mse_k2=np.asarray([x[2] for x in q], dtype=np.float64), command_mse=np.asarray([x[3] for x in q], dtype=np.float64),
        fallback=np.asarray([x[4] for x in q], dtype=bool), peak_c=np.asarray([x[5] for x in q], dtype=np.float64))
    if fixed:
        np.savez_compressed(folder / "fields_1hz.npz", time_ns=np.asarray(plant.field_times,dtype=np.int64),
                            temperature_c=np.asarray(plant.field_frames,dtype=np.float64))


def run(root, output, job_id):
    require(os.environ.get("JEV_EXTENDED_SUPERVISOR_PID") == str(os.getppid()), "Use the supervisor launcher")
    here=Path(__file__).resolve().parent
    protocol=read_json(output / "protocol.json")
    require(protocol["root"] == str(root), "Root differs from protocol")
    ph=sha256(output / "protocol.json")
    reference=read_json(here / "reference.json")
    for name, expected in protocol["harness_sha256"].items():
        require(sha256(here/name)==expected, f"Harness changed: {name}")
    env=environment_check(root)
    require(environment_identity(env)==protocol["environment_identity"], "Environment changed")
    job=next(j for j in protocol["design"]["jobs"] if j["id"]==job_id)
    source=output / "frozen_source" / job["source"]
    for name, expected in reference["sources"][job["source"]]["files"].items():
        require(sha256(source/name)==expected, f"Frozen source changed: {name}")
    sys.path[:0]=[str(source/"src"),str(root/"third_party/Open-Jev")]
    import numpy as np
    import torch
    from ojpoc.common import setup_torch, StopRequest
    from ojpoc.l2.data import Plant
    from ojpoc.physics import advance
    from step_plant import make_plant_class
    from baseline_check import baseline_compare
    fixed=job["phase"]=="fixed"
    parent=root/(C_RUN if fixed else D_RUN)
    cfg=read_json(parent/"resolved_config.json")
    require(cfg["poc"]["runtime"]["actuation_enabled"] is False, "Simulation only")
    setup_torch(cfg)
    stop=StopRequest();engine=None
    plant_class=make_plant_class(Plant,advance)
    entries={e["episode_id"]:e for e in reference["D_entries"]} if not fixed else {reference["C_entry"]["episode_id"]:reference["C_entry"]}
    data=parent/("fresh_confirmation/02_generate/data" if fixed else "02_generate/data")
    latency=None
    try:
        if not fixed:
            from ojpoc.l2.engine import Engine
            from ojpoc.l2.control import simulate
            if job["mode"]=="delayed":
                latency=read_latency(parent,cfg,job["method"])
            print(f"[{job_id}] Load frozen {job['method']} once for this 24-episode group",flush=True)
            engine=Engine(root,parent,job["method"])
        imports={}
        for name,module in list(sys.modules.items()):
            filename=getattr(module,"__file__",None)
            if not filename:
                continue
            p=Path(filename).resolve()
            if name=="ojpoc" or name.startswith("ojpoc."):
                require(p.is_relative_to(source/"src"),f"Unexpected source import: {p}")
            elif name=="jev" or name.startswith("jev.") or name in ("model_checks","common"):
                require(p.is_relative_to(root/"third_party/Open-Jev") or p.is_relative_to(root/"stages/L0"),f"Unexpected helper import: {p}")
                expected=reference["files"].get(p.relative_to(root).as_posix())
                require(expected==sha256(p),f"Unpinned helper import: {p}")
            else:
                continue
            imports[name]={"path":str(p),"sha256":sha256(p)}
        for index, case in enumerate(job["cases"]):
            key=case["key"]
            done=completed_case(output,key,ph)
            if done is not None:
                check=done/"baseline_comparison.json"
                if check.exists():require(read_json(check)["passed"],f"Completed baseline failed: {key}")
                print(f"[{job_id}] {index+1}/{len(job['cases'])} verified completed: {key}",flush=True)
                continue
            folder=output/"cases"/key/("attempt_"+utc().replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:6])
            folder.mkdir(parents=True)
            started=time.perf_counter();plant=None
            write_json(folder/"condition.json",{**{k:v for k,v in job.items() if k!="cases"},**case})
            write_json(folder/"environment.json",env);write_json(folder/"imports.json",imports)
            write_json(folder/"status.json",{"state":"RUNNING","time_utc":utc()})
            write_json(output/"STATUS.json",{"state":"RUNNING","phase":job["phase"],"job":job_id,"case":key,"time_utc":utc()})
            try:
                entry=entries[case["episode"]]
                meta=read_json(data/"raw"/case["episode"]/"meta.json")
                require(meta["actuation_enabled"] is False and entry.get("ood_group") is None,"Expected simulation-only ID episode")
                plant=plant_class(data,entry,60.,max_step_ms=job["h_ms"],truth_stride=1 if fixed else 10,field_stride=100 if fixed else 0)
                if fixed:
                    from replay import replay_commands
                    old=parent/"delay_sensitivity"/f"shared_physics_{case['delay_ms']:.1f}"
                    trace=read_trace(Path(str(old)+".trace.jsonl"))
                    metrics=replay_commands(plant,trace,stop)
                else:
                    count=0
                    decide=engine.decide
                    def progress_decide(observation):
                        nonlocal count
                        r=decide(observation);count+=1
                        if count%200==0:print(f"[{key}] {count} requests",flush=True)
                        return r
                    engine.decide=progress_decide
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        metrics,trace=simulate(plant,engine,job["mode"],latency,stop)
                    finally:
                        engine.decide=decide
                    require(metrics["requests"]+metrics["busy_skipped_ticks"]==600,"Incomplete request-tick coverage")
                    metrics.update(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                   peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                require(metrics["completed"],"Incomplete episode")
                require([r["time_ns"] for r in plant.truth_rows]==list(range(0,60_000_000_001,10_000_000*(1 if fixed else 10))),"Logging grid changed")
                import resource
                usage=resource.getrusage(resource.RUSAGE_SELF)
                metrics.update(wall_seconds=time.perf_counter()-started,process_peak_rss_gib=usage.ru_maxrss/1024**2)
                save_result(folder,plant,metrics,trace,fixed)
                check=None
                if job["h_ms"]==10.:
                    if fixed:
                        original_metrics=read_json(Path(str(old)+".json"))["metrics"]
                        common={k:v for k,v in original_metrics.items() if k in metrics and k not in ("integration","mode","wall_seconds")}
                        check=baseline_compare(metrics,[],common,[])
                        check["scope"]="Core plant metrics under identical archived applications; no policy inference"
                    else:
                        original=parent/STAGES[job["mode"]]/job["method"]/case["episode"]
                        check=baseline_compare(metrics,trace,read_json(original/"metrics.json"),read_trace(original/"control_trace.jsonl"))
                    write_json(folder/"baseline_comparison.json",check)
                write_json(folder/"status.json",{"state":"EXECUTION_COMPLETE","time_utc":utc()})
                seal(folder,{"case":key,"protocol_sha256":ph})
                print(f"[{job_id}] {index+1}/{len(job['cases'])} complete: cost={metrics['episode_cost']:.12g}",flush=True)
                require(check is None or check["passed"],f"Baseline reproduction failed for {key}; inspect sealed evidence")
            except BaseException as exc:
                # A completed-but-nonreproducing baseline remains sealed and cannot
                # be silently replaced on resume.
                if not (folder/"COMPLETE").exists():
                    write_json(folder/"status.json",{"state":"INTERRUPTED" if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else "FAILED",
                                                     "error":repr(exc),"traceback":traceback.format_exc(),"time_utc":utc()})
                raise
            finally:
                plant=None
    finally:
        if engine is not None:engine.close()
        stop.close()
