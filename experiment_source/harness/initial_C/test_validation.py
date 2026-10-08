"""Synthetic software checks only; never load a research model or original Plant."""
import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_results import baseline_compare, compare_traces, compare_truth, analyze
from step_plant import make_plant_class
from validation_io import (bundle, completed_case, fixed_latency, object_hash, read_latency,
                           read_json, seal, sha256, verify_seal, write_json, STEPS, DELAYS, case_key)


class SyntheticPlant:
    def __init__(self, base, entry, duration):
        self.now = self.tick = 0
        self.end_ns = round(duration * 1e9)
        self.duration_s = duration
        self.field = np.full((2, 2), 1., dtype=np.float64)
        self.p = self.u = np.zeros(4)
        self.goal = 0.
        self.nominal = np.array([1., 0., 0., 0., 0., 1., 1., 1., 1.])
        self.condition = None
        self.mismatch_value = None
        self.error_integral = self.effort_integral = self.violation_s = self.fallback_s = 0.
        self.peak = self.overshoot = 1.
        self.initial_violation = self.previous_violation = self.new_violation = False
        self.recovery_s = None
        self.fallback = False
        self.emissions = []
        self.thermal_times = []
        self.emit()
        self.emit()  # Mirrors the base/L2 duplicate initialization emission.

    def emit(self):
        self.emissions.append(self.now)
        if self.tick % 20 == 0:
            self.thermal_times.append(self.now)

    def metrics(self):
        return {"episode_cost": self.error_integral / self.duration_s, "completed": self.now == self.end_ns}


def synthetic_euler(field, p, u, theta, duration, max_dt=.01):
    """y'=-y; its exact solution is exp(-t), unrelated to research physics."""
    count = max(1, int(math.ceil(duration / max_dt)))
    dt = duration / count
    value = field.copy()
    for _ in range(count):
        value -= dt * value
    return value, p.copy()


class StepTests(unittest.TestCase):
    def plant(self, step, duration=.6, advance=synthetic_euler):
        return make_plant_class(SyntheticPlant, advance)(None, None, duration, max_step_ms=step)

    def test_refinement_keeps_sensor_thermal_and_event_times(self):
        grids = []
        for step in (10., 5., 2.5):
            p = self.plant(step)
            for when in (3_000_000, 13_000_000, 38_000_000, 113_000_000, 138_000_000, 600_000_000):
                p.advance_to(when)
                self.assertEqual(p.now, when)
            grids.append([r["time_ns"] for r in p.truth_rows])
            self.assertEqual(sorted(set(p.thermal_times)), [0, 200_000_000, 400_000_000, 600_000_000])
            self.assertLessEqual(p.max_actual_dt_s, step / 1000 + 1e-15)
        self.assertEqual(grids, [list(range(0, 600_000_001, 10_000_000))] * 3)

    def test_10ms_preserves_legacy_state_and_endpoint_quadrature(self):
        p = self.plant(10., duration=.02)
        p.advance_to(3_000_000)
        p.advance_to(20_000_000)
        first = 1 - .003
        second = first * (1 - .007)
        third = second * (1 - .01)
        self.assertAlmostEqual(float(p.field.mean()), third, places=15)
        self.assertAlmostEqual(p.error_integral, first**2 * .003 + second**2 * .007 + third**2 * .01, places=15)
        self.assertEqual(p.fv_steps, 3)
        self.assertEqual(len(p.truth_rows), 3)

    def test_synthetic_analytic_solution_improves_with_refinement(self):
        errors = []
        for step in (10., 5., 2.5):
            p = self.plant(step)
            p.advance_to(p.end_ns)
            errors.append(abs(float(p.field.mean()) - math.exp(-.6)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_integrator_receives_unsplit_original_event_segments(self):
        calls = []

        def logged(*args, **kwargs):
            calls.append(kwargs)
            return synthetic_euler(*args, **kwargs)

        p = self.plant(2.5, duration=.02, advance=logged)
        p.advance_to(3_000_000)
        p.advance_to(20_000_000)
        self.assertEqual(calls, [{"duration": .003, "max_dt": .0025}, {"duration": .007, "max_dt": .0025}, {"duration": .01, "max_dt": .0025}])

    def test_reject_invalid_step_and_clock(self):
        with self.assertRaises(ValueError):
            self.plant(1.)
        p = self.plant(10.)
        p.advance_to(1_000_000)
        for invalid in (0, 600_000_001):
            with self.assertRaises(ValueError):
                p.advance_to(invalid)


class ArchiveTests(unittest.TestCase):
    def test_fallback_delays_are_never_replaced(self):
        rows = [{"fallback_reason": None, "e2e_ms": 999}, {"fallback_reason": "missing", "e2e_ms": .45},
                {"fallback_reason": "no_feasible", "e2e_ms": 3.2}]
        changed = fixed_latency(rows, 13.)
        self.assertEqual([x["e2e_ms"] for x in changed], [13., .45, 3.2])
        self.assertEqual(rows[0]["e2e_ms"], 999)

    def test_sealed_resume_detects_corruption_and_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "cases/k/attempt_1"
            write_json(case / "metrics.json", {"cost": 1.})
            seal(case, {"case": "k", "protocol_sha256": "abc"})
            self.assertEqual(completed_case(root, "k", "abc"), case)
            with self.assertRaises(ValueError):
                completed_case(root, "k", "wrong")
            write_json(case / "metrics.json", {"cost": 2.})
            with self.assertRaises(ValueError):
                completed_case(root, "k", "abc")

    def test_incomplete_attempt_is_retained_but_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "cases/k/attempt_1/status.json", {"state": "FAILED"})
            self.assertIsNone(completed_case(root, "k", "abc"))
            self.assertTrue((root / "cases/k/attempt_1/status.json").exists())

    def test_read_only_latency_loader_matches_interleaving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"budget": {"latency_repeats": 2}}
            for name in ("15_freeze/frozen.json", "16_calibration/calibration.json", "18_latency/rings/both/ring.json"):
                write_json(root / name, {"synthetic": True})
            expected = []
            for repeat in range(2):
                identity = {"frozen": sha256(root / "15_freeze/frozen.json"), "calibration": sha256(root / "16_calibration/calibration.json"),
                            "ring": sha256(root / "18_latency/rings/both/ring.json"), "method": "shared_physics", "repeat": repeat, "budget": cfg["budget"]}
                folder = root / "18_latency/trials" / f"shared_physics_{repeat}"
                folder.mkdir(parents=True)
                entries = []
                for i, path in enumerate(("normal", "missing_observation", "no_predicted_feasible_candidate")):
                    row = {"path": path, "request": repeat * 3 + i, "error": None}
                    entries.append({"index": i, "identity": object_hash(identity), "row": row, "sha256": object_hash(row)})
                (folder / "requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in entries))
                seal(folder, identity)
            before = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            rows = read_latency(root, cfg)
            self.assertEqual([x["request"] for x in rows], [0, 3, 1, 4, 2, 5])
            self.assertEqual(before, {p: (sha256(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()})

    def test_baseline_gates_control_changes_but_not_timing(self):
        old = {"event": "request", "request": 0, "snapshot_ns": 0, "command": [0.] * 4,
               "choice": 0, "e2e_ms": 12., "scores": [1., 2.]}
        new = dict(old, e2e_ms=19., scores=[1.001, 2.001])
        result = baseline_compare({"episode_cost": .1}, [new], {"episode_cost": .1}, [old])
        self.assertTrue(result["passed"])
        self.assertGreater(result["continuous_trace_max_abs_difference"]["scores"], 0)
        for changed in (dict(new, choice=1), dict(new, command=[.04, 0, 0, 0]), dict(new, snapshot_ns=1)):
            self.assertFalse(baseline_compare({"episode_cost": .1}, [changed], {"episode_cost": .1}, [old])["passed"])
        self.assertFalse(baseline_compare({"episode_cost": .11}, [new], {"episode_cost": .1}, [old])["passed"])

    def test_analyzer_does_not_call_missing_or_failed_data_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "protocol.json", {"synthetic": True})
            result = analyze(root)
            self.assertEqual(len(result["missing_cases"]), 9)
            self.assertNotEqual(result["status"], "EXECUTION_COMPLETE")
            self.assertNotIn("38ms_cost_advantage_persists_on_tested_steps", result)

    def test_truth_alignment_rejects_missing_samples(self):
        with self.assertRaises(ValueError):
            compare_truth([{"time_ns": 0}], [{"time_ns": 0}])

    def test_full_report_detects_changed_order_and_failed_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "protocol.json", {"synthetic": True})
            ph = sha256(root / "protocol.json")
            write_json(root / "STATUS.json", {"state": "EXECUTION_COMPLETE"})
            write_json(root / "INPUT_INTEGRITY.json", {"passed": True})
            write_json(root / "baseline_gate.json", {"passed": True})
            for step in STEPS:
                for delay in DELAYS:
                    key = case_key(step, delay)
                    case = root / "cases" / key / "attempt_1"
                    cost = {13.: .03, 38.: .01, 50.: .04}[delay]
                    if step == 2.5 and delay == 38.:
                        cost = .05
                    metrics = {"episode_cost": cost, "tracking_rmse_k": 1., "max_temperature_c": 80., "overshoot_k": 0.,
                               "missed_deadlines": 0, "filter_interventions": 0, "fallback_s": 0., "requests": 600,
                               "fv_steps": 6000, "actual_max_fv_step_ms": step}
                    write_json(case / "metrics.json", metrics)
                    with gzip.open(case / "trace.jsonl.gz", "wt", encoding="utf-8") as stream:
                        for i in range(600):
                            stream.write(json.dumps({"event": "request", "request": i, "snapshot_ns": i * 100_000_000,
                                                     "command": [0.] * 4, "fallback_reason": None}) + "\n")
                    with gzip.open(case / "truth_100hz.csv.gz", "wt", encoding="utf-8", newline="") as stream:
                        w = csv.writer(stream)
                        w.writerow(["time_ns", "mean_temperature_c", "max_temperature_c"])
                        w.writerows((i * 10_000_000, 75., 80.) for i in range(6001))
                    seal(case, {"case": key, "protocol_sha256": ph})
            result = analyze(root)
            self.assertEqual(result["status"], "EXECUTION_COMPLETE")
            self.assertFalse(result["38ms_cost_advantage_persists_on_tested_steps"])
            self.assertFalse(result["full_cost_ordering_identical_on_tested_steps"])
            self.assertEqual(len(result["refinement_comparisons"]), 9)
            self.assertEqual(len(result["delay_comparisons"]), 6)
            write_json(root / "INPUT_INTEGRITY.json", {"passed": False})
            result = analyze(root)
            self.assertNotEqual(result["status"], "EXECUTION_COMPLETE")
            self.assertNotIn("38ms_cost_advantage_persists_on_tested_steps", result)

    def test_first_divergence_uses_requested_command_and_timestamps(self):
        left = [{"event": "request", "request": i, "snapshot_ns": i * 100_000_000,
                 "command": [0.] * 4, "fallback_reason": None} for i in range(3)]
        right = [dict(row) for row in left]
        right[2]["command"] = [.04, 0., 0., 0.]
        result = compare_traces(left, right)
        self.assertEqual(result["command_disagreements"], 1)
        self.assertEqual(result["first_different_requested_command"]["right"]["snapshot_ns"], 200_000_000)

    def test_skipped_ticks_align_by_time_and_report_missing_coverage(self):
        left = [{"event": "request", "request": i, "snapshot_ns": i * 100_000_000,
                 "command": [0.] * 4, "fallback_reason": None} for i in range(3)]
        right = [dict(left[0]), dict(left[2], request=1)]
        result = compare_traces(left, right)
        self.assertEqual(result["aligned_requests"], 2)
        self.assertEqual(result["left_only_request_times_ns"], [100_000_000])
        self.assertEqual(result["command_disagreements"], 0)

    def test_zip_contains_completed_and_failed_attempts_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "output"
            write_json(root / "cases/a/attempt_1/status.json", {"state": "FAILED"})
            write_json(root / "cases/a/attempt_2/metrics.json", {"synthetic": True})
            path = bundle(root)
            import zipfile
            with zipfile.ZipFile(path) as z:
                self.assertEqual(len(z.namelist()), 2)


if __name__ == "__main__":
    unittest.main()
