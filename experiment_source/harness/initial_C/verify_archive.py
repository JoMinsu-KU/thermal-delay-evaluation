"""Read-only archive integration checks; research simulation/model code is never run."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np

from analyze_results import baseline_compare
from run_validation import inventory
from step_plant import make_plant_class
from test_validation import SyntheticPlant, synthetic_euler
from validation_io import (DELAYS, RUN_REL, read_json, read_latency, read_trace,
                           require, sha256)


def verify(root):
    here = Path(__file__).resolve().parent
    reference = read_json(here / "reference.json")
    result = {"scope": "file checks, archived data parsing, and synthetic code-equivalence check only; no research experiment"}
    result["inventory"] = inventory(root, reference)
    require(result["inventory"]["passed"], "Archive inventory mismatch")
    run = root / RUN_REL
    cfg = read_json(run / "resolved_config.json")
    latency = read_latency(run, cfg)
    result["latency_rows"] = len(latency)
    result["latency_paths"] = {key: sum(r["path"] == key for r in latency) for key in sorted({r["path"] for r in latency})}
    result["archived_baseline_self_comparisons"] = {}
    for delay in DELAYS:
        path = run / "delay_sensitivity" / f"shared_physics_{delay:.1f}"
        metrics = read_json(Path(str(path) + ".json"))["metrics"]
        trace = read_trace(Path(str(path) + ".trace.jsonl"))
        comparison = baseline_compare(metrics, trace, metrics, trace)
        require(comparison["passed"], "Archived baseline schema is unsupported")
        result["archived_baseline_self_comparisons"][str(delay)] = {
            "passed": True, "requests": metrics["requests"], "trace_records": len(trace), "cost": metrics["episode_cost"]}
    # Execute only the original event-loop METHOD, on a synthetic y'=-y state.
    # Its global 'advance' is injected as synthetic_euler. No original physics,
    # original Plant constructor, saved initial temperature, or policy is loaded.
    source_path = root / reference["source_root_rel"] / "src/ojpoc/l2/data.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Plant")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "advance_to")
    namespace = {"advance": synthetic_euler, "require": require}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path) + "::synthetic_method_test", "exec"), namespace)
    LegacySynthetic = type("LegacySynthetic", (SyntheticPlant,), {"advance_to": namespace["advance_to"]})
    legacy = LegacySynthetic(None, None, .6)
    refined = make_plant_class(SyntheticPlant, synthetic_euler)(None, None, .6, max_step_ms=10.)
    attributes = ("now", "tick", "error_integral", "effort_integral", "violation_s", "fallback_s", "peak", "overshoot",
                  "initial_violation", "previous_violation", "new_violation", "recovery_s", "emissions", "thermal_times")
    event_ns = (0, 466570, 3_000_000, 13_000_000, 38_000_000, 113_000_000,
                138_000_000, 150_000_000, 299_999_999, 300_000_000, 600_000_000)
    for i, when in enumerate(event_ns):
        legacy.advance_to(when)
        refined.advance_to(when)
        for name in attributes:
            require(getattr(legacy, name) == getattr(refined, name), f"10 ms wrapper changed legacy semantics: {name}")
        require(np.array_equal(legacy.field, refined.field) and np.array_equal(legacy.p, refined.p), "State arithmetic differs")
        legacy.u = np.full(4, (i % 3) * .04)
        refined.u = legacy.u.copy()
        legacy.fallback = refined.fallback = bool(i % 2)
    result["synthetic_legacy_equivalence"] = {"passed": True, "event_times_tested": len(event_ns),
                                               "source_sha256": sha256(source_path), "state": "synthetic 2x2 cooling field"}
    result["passed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
