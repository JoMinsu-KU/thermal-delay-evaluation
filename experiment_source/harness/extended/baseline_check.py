"""Reproduction gate against archived records, not a convergence criterion."""
import math
BASELINE_ATOL=1e-10
BASELINE_RTOL=1e-8
VOLATILE_OR_CONTINUOUS={"e2e_ms","gpu_compute_ms","scope","scores","probabilities","predicted_peak_c"}

def baseline_compare(metrics, trace, reference_metrics, reference_trace):
    failures = []
    for key, old in reference_metrics.items():
        if key in ("integration", "wall_seconds"):
            continue
        new = metrics.get(key)
        if isinstance(old, float) and isinstance(new, (float, int)):
            equal = math.isclose(old, new, abs_tol=BASELINE_ATOL, rel_tol=BASELINE_RTOL)
        else:
            equal = old == new
        if not equal:
            failures.append({"metric": key, "archived": old, "rerun": new})
    changed = []
    continuous = {k: 0.0 for k in ("scores", "probabilities", "predicted_peak_c")}
    if len(trace) != len(reference_trace):
        failures.append({"trace_lengths": [len(reference_trace), len(trace)]})
    for i, (old, new) in enumerate(zip(reference_trace, trace)):
        old_fixed = {k: v for k, v in old.items() if k not in VOLATILE_OR_CONTINUOUS}
        new_fixed = {k: v for k, v in new.items() if k not in VOLATILE_OR_CONTINUOUS}
        if old_fixed != new_fixed:
            changed.append(i)
            if len(changed) == 1:
                failures.append({"first_trace_difference": i, "archived": old_fixed, "rerun": new_fixed})
        for key in continuous:
            a, b = old.get(key), new.get(key)
            if a is not None and b is not None and len(a) == len(b):
                continuous[key] = max(continuous[key], max((abs(x - y) for x, y in zip(a, b)), default=0.))
            elif a != b:
                failures.append({"continuous_field_structure_changed": key, "trace_index": i})
    return {"passed": not failures, "metric_atol": BASELINE_ATOL, "metric_rtol": BASELINE_RTOL,
            "require_exact": "request/action times, commands, candidate choices, masks, fallback paths, replay-row identities",
            "trace_records_changed": len(changed), "continuous_trace_max_abs_difference": continuous,
            "failures": failures}
