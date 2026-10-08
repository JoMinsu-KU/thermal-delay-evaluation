#!/usr/bin/env python3
"""Manually run the frozen single-case protocol in the original H200 jev environment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from validation_io import (DELAYS, EPISODE, RUN_REL, STEPS, bundle, case_key, completed_case,
                           fixed_latency, object_hash, read_json, read_latency, read_trace,
                           require, seal, sha256, under, write_json)

HERE = Path(__file__).resolve().parent
REQUIRED_PREFIX = Path("/home/jovyan/.conda/envs/jev")
HARNESS_FILES = ("run_validation.py", "step_plant.py", "analyze_results.py", "validation_io.py",
                 "reference.json", "README_KO.md", "run_validation.sh", "test_validation.py", "verify_archive.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def inventory(root, reference):
    """Only reads file bytes; safe to invoke without NumPy, Torch, or a GPU."""
    failures = []
    checked = 0
    total_bytes = 0
    for name, expected in reference["files"].items():
        path = under(root, name)
        if not path.is_file():
            failures.append({"file": name, "error": "missing"})
            continue
        actual = sha256(path)
        checked += 1
        total_bytes += path.stat().st_size
        if actual != expected:
            failures.append({"file": name, "expected": expected, "actual": actual})
    run = root / RUN_REL
    original = read_json(run / "delay_sensitivity/protocol.json") if (run / "delay_sensitivity/protocol.json").exists() else {}
    source = under(root, reference["source_root_rel"])
    source_keys = {p.relative_to(source).as_posix() for p in (source / "src").rglob("*.py")}
    expected_source = {p for p in original.get("source_hashes", {}) if p.startswith("src/")}
    if source_keys != expected_source:
        failures.append({"source_file_set_changed": sorted(source_keys ^ expected_source)})
    for name in sorted(source_keys & expected_source):
        if sha256(source / name) != original["source_hashes"][name]:
            failures.append({"historical_source_hash_mismatch": name})
    return {"passed": not failures, "checked_files": checked, "checked_bytes": total_bytes,
            "failures": failures, "checked_at_utc": utc(), "root": str(root),
            "scope": "read-only inventory; no inference or simulation executed"}


def environment_check(root):
    require(sys.platform.startswith("linux"), "Experiments must run on the original Linux H200 instance")
    require(Path(sys.prefix).resolve() == REQUIRED_PREFIX.resolve(), f"Activate Conda {REQUIRED_PREFIX}")
    for key in ("JEV_LOAD_8BIT", "JEV_LOAD_4BIT", "JEV_DEVICE_MAP", "JEV_MAX_MEMORY"):
        require(not os.environ.get(key), f"Unset {key}; this protocol preserves original BF16/CUDA placement")
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Use the supplied launcher for offline model loading")
    expected_cache = (root / "cache/huggingface/hub").resolve()
    require(Path(os.environ.get("HF_HUB_CACHE", "")).resolve() == expected_cache, "HF_HUB_CACHE differs from the archived project cache")
    for key in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        require(not os.environ.get(key) or Path(os.environ[key]).resolve() == expected_cache, f"{key} overrides the verified model cache")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible CUDA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    require("H200" in gpu.name, f"Original protocol requires H200; found {gpu.name}")
    packages = {}
    for package in ("torch", "numpy", "transformers", "peft", "accelerate", "safetensors", "PyYAML", "huggingface-hub", "tokenizers"):
        packages[package] = importlib.metadata.version(package)
    result = {
        "checked_at_utc": utc(), "python": sys.version, "prefix": sys.prefix,
        "platform": platform.platform(), "packages": packages, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": gpu.name,
        "gpu_total_memory_gib": gpu.total_memory / 2**30,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED",
            "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")},
    }
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    result["nvidia_smi"] = query.stdout.strip() or query.stderr.strip()
    result["driver_version"] = query.stdout.split(",")[1].strip() if query.returncode == 0 and "," in query.stdout else "unknown"
    return result


def environment_identity(env):
    return {key: value for key, value in env.items() if key not in ("checked_at_utc", "nvidia_smi")}


@contextmanager
def experiment_lock(root):
    import fcntl
    path = root / "locks/l0.lock"
    require(path.parent.is_dir(), f"Missing original lock directory: {path.parent}")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another L0/L1/L2 process holds locks/l0.lock; stop this run and retry later") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "started_utc": utc(), "purpose": "integration_step_sensitivity"}) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def protocol_value(root, reference):
    return {
        "schema": "jev-single-case-integration-step-v1", "root": str(root), "parent_run": RUN_REL,
        "episode": EPISODE, "duration_s": 60., "method": "shared_physics", "post_hoc": True,
        "actuation_enabled": False, "training_or_selection": False,
        "cases": [{"key": case_key(step, delay), "step_ms": step, "delay_ms": delay} for step in STEPS for delay in DELAYS],
        "source": "SHA256-verified 2026-09-26 revision artifact; all src files match the delay experiment protocol",
        "source_root_rel": reference["source_root_rel"],
        "change": "only max_dt of the true-plant FV integrator",
        "fixed": ["FV64 FP64 spatial solver", "selected checkpoint and Qwen revision", "observer and PINO",
                  "calibration", "initial condition and noise seeds", "100 Hz sensors and 5 Hz thermal frames",
                  "100 ms control request grid", "conditional fallback latency replay", "event-driven no-op semantics",
                  "original 10 ms/event endpoint cost and peak evaluation"],
        "baseline_gate": {"metric_atol": 1e-10, "metric_rtol": 1e-8,
                          "trace": "exact semantic event records excluding wall timing and continuous candidate diagnostics",
                          "failure_action": "stop before any 5 or 2.5 ms case; preserve all evidence"},
        "fresh_process_per_case": True, "seed": 3030,
        "reference_sha256": sha256(HERE / "reference.json"),
        "harness_sha256": {name: sha256(HERE / name) for name in HARNESS_FILES},
        "archived_source_sha256": reference["frozen_source"],
        "interpretation": "Descriptive single-case robustness check; no prevalence estimate or automatic convergence threshold",
    }


def prepare_output(root, output, reference, resume, environment):
    protected = root / RUN_REL
    require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "Output overlaps the completed parent run")
    require(not output.is_relative_to(HERE) and not HERE.is_relative_to(output), "Output overlaps validation code")
    protocol = protocol_value(root, reference)
    protocol["environment_identity"] = environment_identity(environment)
    if output.exists():
        require(resume, f"Output exists; use --resume: {output}")
        require(read_json(output / "protocol.json") == protocol, "Resume protocol or harness changed; preserve this output and use a new directory")
        for name, expected in protocol["harness_sha256"].items():
            require(sha256(output / "harness" / name) == expected, f"Copied harness changed: {name}")
        for name, expected in protocol["archived_source_sha256"].items():
            require(sha256(output / "frozen_source" / name) == expected, f"Copied frozen source changed: {name}")
        return
    output.mkdir(parents=True)
    (output / "harness").mkdir()
    for name in HARNESS_FILES:
        shutil.copyfile(HERE / name, output / "harness" / name)
    for name, expected in protocol["archived_source_sha256"].items():
        target = output / "frozen_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / reference["source_root_rel"] / name, target)
        require(sha256(target) == expected, "Frozen source changed while copying")
    write_json(output / "protocol.json", protocol)


def worker(root, output, key):
    # This entrypoint is internal; the supervisor holds the shared experiment lock.
    require(os.environ.get("JEV_STEP_SUPERVISOR_PID") == str(os.getppid()), "Launch via run_validation.sh, not --worker-case")
    protocol = read_json(output / "protocol.json")
    require(protocol["root"] == str(root), "Worker root differs from protocol")
    for name, expected in protocol["harness_sha256"].items():
        require(sha256(HERE / name) == expected, f"Worker harness changed: {name}")
    for name, expected in protocol["archived_source_sha256"].items():
        require(sha256(output / "frozen_source" / name) == expected, f"Frozen import changed: {name}")
    case = next(x for x in protocol["cases"] if x["key"] == key)
    protocol_hash = sha256(output / "protocol.json")
    require(completed_case(output, key, protocol_hash) is None, "Refusing to rerun a completed case")
    attempt = output / "cases" / key / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    write_json(attempt / "status.json", {"state": "RUNNING", "started_utc": utc(), "case": key})
    engine = stop = None
    try:
        env = environment_check(root)
        require(environment_identity(env) == protocol["environment_identity"], "Worker environment differs from frozen protocol")
        write_json(attempt / "environment.json", env)
        sys.path[:0] = [str(output / "frozen_source/src"), str(root / "third_party/Open-Jev")]
        import numpy as np
        import torch
        from ojpoc.common import StopRequest, setup_torch
        from ojpoc.l2.contracts import plan
        from ojpoc.l2.control import simulate
        from ojpoc.l2.data import Plant
        from ojpoc.l2.engine import Engine
        from ojpoc.physics import advance
        from step_plant import make_plant_class
        run = root / RUN_REL
        cfg = read_json(run / "resolved_config.json")
        require(cfg["poc"]["runtime"]["actuation_enabled"] is False, "Simulation only")
        require(cfg["seed"] == 3030, "Unexpected original seed")
        fresh = read_json(run / "fresh_confirmation/protocol.json")["config"]
        entry = next(x for x in plan(fresh) if x["episode_id"] == EPISODE)
        require(entry == read_json(HERE / "reference.json")["entry"], "Episode recipe changed")
        data = run / "fresh_confirmation/02_generate/data"
        require(read_json(data / "raw" / EPISODE / "meta.json")["actuation_enabled"] is False, "Real actuation prohibited")
        latency = fixed_latency(read_latency(run, cfg), case["delay_ms"])
        setup_torch(cfg)
        stop = StopRequest()
        print(f"[{key}] Loading the frozen controller (offline)", flush=True)
        engine = Engine(root, run, "shared_physics")
        # Check imported files after dynamic helper loading, including vendor code.
        imports = {}
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if not filename or not (name == "ojpoc" or name.startswith("ojpoc.") or name == "jev" or name.startswith("jev.") or name in ("model_checks", "common")):
                continue
            path = Path(filename).resolve()
            if name == "ojpoc" or name.startswith("ojpoc."):
                require(path.is_relative_to(output / "frozen_source/src"), f"Imported current root code: {path}")
            elif name == "jev" or name.startswith("jev."):
                require(path.is_relative_to(root / "third_party/Open-Jev"), f"Imported unexpected vendor package: {path}")
            else:
                require(path.is_relative_to(root / "stages/L0"), f"Imported unexpected model helper: {path}")
            imports[name] = {"path": str(path), "sha256": sha256(path)}
        write_json(attempt / "imports.json", imports)
        original_decide = engine.decide
        n_requests = 0

        def logged_decide(make_observation):
            nonlocal n_requests
            result = original_decide(make_observation)
            n_requests += 1
            if n_requests % 100 == 0:
                print(f"[{key}] {n_requests}/600 control requests", flush=True)
            return result

        engine.decide = logged_decide
        plant_class = make_plant_class(Plant, advance)
        plant = plant_class(data, entry, 60., max_step_ms=case["step_ms"])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics, trace = simulate(plant, engine, "delayed", latency, stop)
        metrics.update(wall_seconds=time.perf_counter() - started,
                       peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics.update(process_peak_rss_gib=usage.ru_maxrss / 1024**2,
                       process_user_cpu_s=usage.ru_utime, process_system_cpu_s=usage.ru_stime)
        require(metrics["completed"] and metrics["requests"] + metrics["busy_skipped_ticks"] == 600, "Incomplete control-tick coverage")
        require(metrics["inference_failures"] == 0 and all(not r.get("error") for r in trace), "Inference error")
        require([x["time_ns"] for x in plant.truth_rows] == list(range(0, 60_000_000_001, 10_000_000)), "Changed observation logging grid")
        write_json(attempt / "metrics.json", metrics)
        import gzip
        import csv
        with gzip.open(attempt / "trace.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
            for row in trace:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        with gzip.open(attempt / "truth_100hz.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(plant.truth_rows[0]))
            writer.writeheader()
            writer.writerows(plant.truth_rows)
        np.save(attempt / "final_T64.npy", plant.field, allow_pickle=False)
        if case["step_ms"] == 10.:
            from analyze_results import baseline_compare
            archive = run / "delay_sensitivity" / f"shared_physics_{case['delay_ms']:.1f}"
            check = baseline_compare(metrics, trace, read_json(Path(str(archive) + ".json"))["metrics"],
                                     read_trace(Path(str(archive) + ".trace.jsonl")))
            write_json(attempt / "baseline_comparison.json", check)
        write_json(attempt / "status.json", {"state": "EXECUTION_COMPLETE", "finished_utc": utc(), "case": key})
        seal(attempt, {"case": key, "protocol_sha256": protocol_hash})
        print(f"[{key}] Saved; cost={metrics['episode_cost']:.12g}", flush=True)
    except BaseException as exc:
        write_json(attempt / "status.json", {"state": "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "FAILED",
                                            "finished_utc": utc(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if engine is not None:
            engine.close()
        if stop is not None:
            stop.close()


def run_worker(root, output, key, session):
    log = session / (key + ".log")
    env = dict(os.environ, JEV_STEP_SUPERVISOR_PID=str(os.getpid()))
    cmd = [sys.executable, "-B", "-u", str(output / "harness/run_validation.py"),
           "--root", str(root), "--output", str(output), "--worker-case", key]
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    require(code == 0, f"Worker failed ({code}): {log}")


def evaluate_gate(output):
    checks = {}
    for delay in DELAYS:
        key = case_key(10., delay)
        folder = completed_case(output, key, sha256(output / "protocol.json"))
        if folder is None:
            checks[key] = {"passed": False, "reason": "Missing completed baseline"}
        else:
            checks[key] = read_json(folder / "baseline_comparison.json")
    result = {"passed": all(c["passed"] for c in checks.values()), "cases": checks}
    write_json(output / "baseline_gate.json", result)
    return result


def supervise(args):
    root = args.root.resolve()
    output = (args.output or root / "runs/L2_diagnostics" / ("integration_step_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    reference = read_json(HERE / "reference.json")
    if args.inventory_only:
        result = inventory(root, reference)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    with experiment_lock(root):
        # Validate archived files before creating a runnable output.
        print("Checking frozen inputs, including cached base weights...", flush=True)
        checked = inventory(root, reference)
        if not checked["passed"]:
            print(json.dumps(checked, indent=2))
            return 2
        env = environment_check(root)
        prepare_output(root, output, reference, args.resume, env)
        session = output / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        write_json(session / "inventory_before.json", checked)
        write_json(session / "environment.json", env)
        lock_record = root / "locks/integration_step_sensitivity" / (session.name + ".json")
        log_record = root / "logs/L2/integration_step_sensitivity" / (session.name + ".json")
        audit = {"output": str(output), "session": str(session), "protocol_sha256": sha256(output / "protocol.json"),
                 "source_root_rel": reference["source_root_rel"], "source_sha256": reference["frozen_source"],
                 "environment": env, "actuation_enabled": False}
        write_json(lock_record, audit)
        write_json(log_record, {"state": "PREFLIGHT_PASS", "time_utc": utc(), **audit})
        write_json(output / "STATUS.json", {"state": "PREFLIGHT_PASS", "time_utc": utc(), "output": str(output)})
        if args.check_only:
            print(f"PREFLIGHT_PASS; no inference run. Continue with --resume --output {output}")
            return 0
        state, error = "EXECUTION_COMPLETE", None
        try:
            for step in STEPS:
                for delay in DELAYS:
                    key = case_key(step, delay)
                    if completed_case(output, key, sha256(output / "protocol.json")) is None:
                        write_json(output / "STATUS.json", {"state": "RUNNING", "case": key, "time_utc": utc()})
                        run_worker(root, output, key, session)
                    else:
                        print(f"[{key}] Verified completed case; retained", flush=True)
                if step == 10.:
                    require(evaluate_gate(output)["passed"], "10 ms archived reproduction failed; finer-step runs were not started")
            after = inventory(root, reference)
            write_json(session / "inventory_after.json", after)
            write_json(output / "INPUT_INTEGRITY.json", after)
            require(after["passed"], "Archived input changed during execution; results need an integrity audit")
        except BaseException as exc:
            state = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "FAILED"
            error = {"message": repr(exc), "traceback": traceback.format_exc()}
        write_json(output / "STATUS.json", {"state": state, "time_utc": utc(), "error": error})
        try:
            from analyze_results import analyze
            analyze(output)
        except Exception as exc:
            write_json(output / "analysis_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            state = "FAILED"
            write_json(output / "STATUS.json", {"state": state, "time_utc": utc(), "error": "Analysis failed; see analysis_error.json"})
        archive = bundle(output)
        write_json(log_record, {"state": state, "time_utc": utc(), "archive": str(archive), "error": error, **audit})
        print(f"{state}\nResults: {output}\nReturn this ZIP: {archive}", flush=True)
        if error:
            print(error["message"], file=sys.stderr)
        return 0 if state == "EXECUTION_COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Original Linux project root")
    parser.add_argument("--output", type=Path, help="New output directory, outside the completed parent run")
    parser.add_argument("--resume", action="store_true", help="Verify and retain completed cases; restart incomplete cases in new attempt directories")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Verify files and H200/jev environment without running inference")
    mode.add_argument("--inventory-only", action="store_true", help="Read-only file checks; no NumPy/Torch/GPU imports")
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        require(args.output is not None, "Worker output is required")
        worker(args.root.resolve(), args.output.resolve(), args.worker_case)
        return 0
    return supervise(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
