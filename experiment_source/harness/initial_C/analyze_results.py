"""Descriptive analysis only: one selected case cannot establish prevalence."""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

from validation_io import (DELAYS, STEPS, case_key, completed_case, read_json,
                           read_trace, require, sha256, write_json)

BASELINE_ATOL = 1e-10
BASELINE_RTOL = 1e-8
VOLATILE_OR_CONTINUOUS = {"e2e_ms", "gpu_compute_ms", "scope", "scores", "probabilities", "predicted_peak_c"}


def baseline_compare(metrics, trace, reference_metrics, reference_trace):
    failures = []
    for key, old in reference_metrics.items():
        if key == "integration":
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


def decision_context(row):
    allowed = row.get("predicted_safe_mask", [])
    scores = row.get("scores")
    ranked = sorted((x for x, ok in zip(scores or [], allowed) if ok), reverse=True)
    return {k: row.get(k) for k in ("request", "snapshot_ns", "ready_ns", "choice", "command", "fallback_reason",
                                   "filter_intervention", "valid_mask", "predicted_safe_mask", "scores", "predicted_peak_c")} | {
        "top_two_allowed_score_gap": ranked[0] - ranked[1] if len(ranked) >= 2 else None}


def compare_traces(left, right):
    a = {r["snapshot_ns"]: r for r in left if r["event"] == "request"}
    b = {r["snapshot_ns"]: r for r in right if r["event"] == "request"}
    keys = sorted(a.keys() & b.keys())
    changed = [k for k in keys if a[k]["command"] != b[k]["command"]]
    masks = sum(a[k].get("predicted_safe_mask") != b[k].get("predicted_safe_mask") for k in keys)
    fallbacks = sum(a[k]["fallback_reason"] != b[k]["fallback_reason"] for k in keys)
    squared = [sum((x - y) ** 2 for x, y in zip(a[k]["command"], b[k]["command"])) / 4 for k in keys]
    def applications(trace, requests):
        times = {r["request"]: t for t, r in requests.items()}
        return {times[r["request"]]: r for r in trace if r["event"] in ("apply", "deadline_fallback")}

    aa, bb = applications(left, a), applications(right, b)
    semantic = lambda r: {k: v for k, v in r.items() if k != "request"} if r is not None else None
    application_changes = [k for k in sorted(aa.keys() | bb.keys()) if semantic(aa.get(k)) != semantic(bb.get(k))]
    return {
        "alignment": "common snapshot timestamps; skipped requests are reported separately",
        "requests_left": len(a), "requests_right": len(b),
        "left_only_request_times_ns": sorted(a.keys() - b.keys()),
        "right_only_request_times_ns": sorted(b.keys() - a.keys()),
        "aligned_requests": len(keys), "command_disagreements": len(changed),
        "command_disagreement_fraction": len(changed) / len(keys) if keys else None,
        "requested_command_rmse": math.sqrt(sum(squared) / len(squared)) if squared else None,
        "safe_mask_disagreements": masks, "fallback_path_disagreements": fallbacks,
        "application_event_disagreements": len(application_changes),
        "first_different_application": {"left": aa.get(application_changes[0]), "right": bb.get(application_changes[0])} if application_changes else None,
        "first_different_requested_command": {"left": decision_context(a[changed[0]]), "right": decision_context(b[changed[0]])} if changed else None,
    }


def load_truth(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return [{k: int(v) if k == "time_ns" else float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def compare_truth(left, right):
    expected = list(range(0, 60_000_000_001, 10_000_000))
    require([r["time_ns"] for r in left] == expected == [r["time_ns"] for r in right], "Incomplete 100 Hz truth logging")
    result = {}
    for key in ("mean_temperature_c", "max_temperature_c"):
        delta = [b[key] - a[key] for a, b in zip(left, right)]
        result[key] = {"rmse_k": math.sqrt(sum(x*x for x in delta) / len(delta)),
                       "max_abs_k": max(abs(x) for x in delta), "final_difference_k": delta[-1]}
    return result


def analyze(output):
    output = Path(output)
    protocol_hash = sha256(output / "protocol.json")
    cases, missing = {}, []
    for step in STEPS:
        for delay in DELAYS:
            key = case_key(step, delay)
            folder = completed_case(output, key, protocol_hash)
            if folder is None:
                missing.append(key)
                continue
            cases[key] = {"folder": folder, "metrics": read_json(folder / "metrics.json"),
                          "trace": read_trace(folder / "trace.jsonl.gz"),
                          "truth": load_truth(folder / "truth_100hz.csv.gz")}
    gate_path = output / "baseline_gate.json"
    gate = read_json(gate_path) if gate_path.exists() else {"passed": False, "reason": "Not evaluated"}
    integrity_path = output / "INPUT_INTEGRITY.json"
    integrity = read_json(integrity_path) if integrity_path.exists() else {"passed": False}
    execution_path = output / "STATUS.json"
    execution = read_json(execution_path) if execution_path.exists() else {"state": "UNKNOWN"}
    complete = not missing and gate["passed"] and integrity["passed"] and execution["state"] == "EXECUTION_COMPLETE"
    result = {
        "scope": "post hoc numerical sensitivity check of one selected episode, l2_090006",
        "status": "EXECUTION_COMPLETE" if complete else "INCOMPLETE_OR_VALIDATION_FAILED",
        "input_integrity_passed": integrity["passed"], "execution_status": execution["state"],
        "missing_cases": missing, "baseline_gate": gate,
        "metrics": {key: x["metrics"] for key, x in cases.items()},
        "cost_orderings": {}, "refinement_comparisons": {}, "delay_comparisons": {},
        "interpretation": "No statistical significance, physical counterexample, or general prevalence is inferred from this single case.",
        "numerical_scope": "Plant temporal refinement only; spatial grid, controller, observation timing, and 10 ms/event cost quadrature are fixed. This is not a proof of solver convergence.",
    }
    rows = []
    for step in STEPS:
        selected = [(delay, cases.get(case_key(step, delay))) for delay in DELAYS]
        if not all(x is not None for _, x in selected):
            continue
        costs = {str(delay): x["metrics"]["episode_cost"] for delay, x in selected}
        order = sorted(DELAYS, key=lambda d: costs[str(d)])
        result["cost_orderings"][str(step)] = {
            "ascending_delay_ms": order, "costs": costs,
            "cost_13_over_38": costs["13.0"] / costs["38.0"],
            "cost_50_over_38": costs["50.0"] / costs["38.0"],
            "38ms_lower_than_both": costs["38.0"] < min(costs["13.0"], costs["50.0"]),
        }
        for delay, x in selected:
            m = x["metrics"]
            rows.append({"step_ms": step, "delay_ms": delay, **{k: m[k] for k in (
                "episode_cost", "tracking_rmse_k", "max_temperature_c", "overshoot_k", "missed_deadlines",
                "filter_interventions", "fallback_s", "requests", "fv_steps", "actual_max_fv_step_ms")}})
        for delay in (13.0, 50.0):
            a, b = cases[case_key(step, 38)], cases[case_key(step, delay)]
            result["delay_comparisons"][f"h{step:g}:38vs{delay:g}"] = {
                "trace": compare_traces(a["trace"], b["trace"]), "truth": compare_truth(a["truth"], b["truth"])}
    for delay in DELAYS:
        for coarse, fine in ((10., 5.), (5., 2.5), (10., 2.5)):
            a, b = cases.get(case_key(coarse, delay)), cases.get(case_key(fine, delay))
            if a is None or b is None:
                continue
            ca, cb = a["metrics"]["episode_cost"], b["metrics"]["episode_cost"]
            result["refinement_comparisons"][f"d{delay:g}:{coarse:g}to{fine:g}"] = {
                "cost_change": cb - ca, "cost_change_percent_of_coarse": 100 * (cb - ca) / ca,
                "trace": compare_traces(a["trace"], b["trace"]), "truth": compare_truth(a["truth"], b["truth"])}
    if complete:
        result["38ms_cost_advantage_persists_on_tested_steps"] = all(x["38ms_lower_than_both"] for x in result["cost_orderings"].values())
        result["full_cost_ordering_identical_on_tested_steps"] = len({tuple(x["ascending_delay_ms"]) for x in result["cost_orderings"].values()}) == 1
    write_json(output / "summary.json", result)
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Integration-step sensitivity: selected delayed-control case", "",
             f"Status: **{result['status']}**", "", result["scope"], "", result["numerical_scope"], "",
             f"Archived 10 ms reproduction gate: **{'PASS' if gate['passed'] else 'NOT PASSED'}**", "",
             "| FV maximum step (ms) | Normal delay (ms) | Episode cost | Tracking RMSE (K) | Peak (C) | Deadline misses |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step_ms']:g} | {row['delay_ms']:g} | {row['episode_cost']:.10f} | {row['tracking_rmse_k']:.6f} | {row['max_temperature_c']:.6f} | {row['missed_deadlines']} |")
    lines += ["", "## Refinement from 5 to 2.5 ms", "",
              "| Delay (ms) | Cost change (%) | Requested command differences | Mean-temperature trajectory RMSE (K) |", "|---:|---:|---:|---:|"]
    for delay in DELAYS:
        v = result["refinement_comparisons"].get(f"d{delay:g}:5to2.5")
        if v:
            lines.append(f"| {delay:g} | {v['cost_change_percent_of_coarse']:.6f} | {v['trace']['command_disagreements']} / {v['trace']['aligned_requests']} | {v['truth']['mean_temperature_c']['rmse_k']:.6f} |")
    lines += ["", "## Interpretation", "", result["interpretation"], "",
              "Costs use identical observation/event endpoint quadrature at all three solver steps. Peak and violation summaries use those same endpoints. Controller inference timing is logged, but archived path-specific replay values determine simulated arrival times.", ""]
    if result["status"] == "EXECUTION_COMPLETE":
        persists = result["38ms_cost_advantage_persists_on_tested_steps"]
        lines.append("The 38 ms case retains lower cost than both 13 and 50 ms on all three tested steps." if persists else
                     "The original 38 ms cost advantage does not persist on all tested steps. The original case should not support a numerically robust non-monotonicity claim.")
        lines += ["", "Assess the 5-to-2.5 ms cost and trajectory changes together with action differences before describing the case as numerically stable. A retained ordering alone does not establish convergence. Substantial changes call for further refinement or a narrower manuscript claim."]
    else:
        lines += ["No robustness conclusion is available. Inspect baseline_gate.json, STATUS.json, and worker logs before resuming."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(analyze(args.output)["status"])
