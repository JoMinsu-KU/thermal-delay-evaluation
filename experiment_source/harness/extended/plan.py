"""Fixed scope; no outcome-dependent episode or checkpoint selection."""
C_RUN = "runs/L2_diagnostics/diagnosis_20260926/full_relative_s3030"
D_RUN = "runs/L2/l2_auto_20260928T043858Z_2f3e14_pilot"
C_SOURCE = "artifacts/L2_revision_final_20260926/source"
D_SOURCE = "artifacts/L2_residual_20260928/source"
METHODS = ("pi", "fv_mpc", "pino_mpc", "shared_physics", "small_physics")
LABELS = {"pi": "PI", "fv_mpc": "FV-MPC", "pino_mpc": "PINO-MPC",
          "shared_physics": "Shared-Residual", "small_physics": "Small-Residual"}
MODES = ("zero", "delayed")
STAGES = {"zero": "19_closed_loop", "delayed": "20_delayed_loop"}
D_STEPS = (10., 5., 2.5)
C_STEPS = (10., 5., 2.5, 1.25)
DELAYS = (13., 38., 50.)
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_RESAMPLES = 20000


def all_jobs(reference):
    jobs = []
    for h in C_STEPS:
        jobs.append({"id": f"fixed_h{h:g}", "phase": "fixed", "source": "C", "h_ms": h,
                     "cases": [{"key": f"fixed_d{d:g}_h{h:g}", "delay_ms": d,
                                "episode": reference["C_entry"]["episode_id"]} for d in DELAYS]})
    for h in D_STEPS:
        for method in METHODS:
            for mode in MODES:
                jobs.append({"id": f"D_h{h:g}_{mode}_{method}", "phase": "baseline" if h == 10 else "refine",
                             "source": "D", "h_ms": h, "method": method, "mode": mode,
                             "cases": [{"key": f"D_h{h:g}_{mode}_{method}_{e['episode_id']}", "episode": e["episode_id"]}
                                       for e in reference["D_entries"]]})
    return jobs


def design(reference):
    return {
        "schema": "jev-extended-numerical-validation-v1", "created_date": "2026-10-08",
        "actuation_enabled": False, "training": False, "model_or_alpha_selection": False,
        "duration_s": 60, "normal_episodes": [e["episode_id"] for e in reference["D_entries"]],
        "jobs": all_jobs(reference), "fixed_command_runs": 12, "D_reproduction_runs": 240,
        "D_additional_runs": 480, "total_runs": 732,
        "plant_change": "Only the actual plant FV maximum time step changes; spatial grid remains 64x64 FP64",
        "unchanged": ["controller/FV-MPC candidate prediction time step", "PINO", "selected weights and residual alpha",
                      "normalization and calibration", "initial state and noise seeds", "100 Hz sensor emission",
                      "5 Hz thermal emission", "100 ms request clock", "path/token-matched archived latency replay rule",
                      "no-op application semantics", "10 ms/event-endpoint cost and peak quadrature"],
        "fixed_command_replay": "C's original 10-ms applied commands, timestamps and fallback flags; no policy inference",
        "D_primary_contrasts": ["FV-MPC delayed minus zero", "Shared minus Small, zero", "Shared minus Small, delayed"],
        "D_secondary_contrasts": ["delay effect for all five methods", "Shared minus PINO in each mode"],
        "statistics": {"unit": "paired normal episode", "n": 24, "resamples": BOOTSTRAP_RESAMPLES,
                       "seed": BOOTSTRAP_SEED, "interval": "percentile 95%", "reporting_tolerances": [1e-10, 1e-8, 1e-6],
                       "scope": "conditional on the frozen systems and archived latency replay; no training or fresh timing uncertainty"},
        "reproduction_gate": "All C fixed-command 10-ms core metrics and all 240 D 10-ms archived metrics/events must reproduce; D gate precedes refinement",
        "scientific_stop_rule": "No automatic success threshold or promotion. Report all cases and changes even if adverse. No extra refinement or retraining starts automatically.",
        "partial_results": "Never present incomplete D coverage as a complete cohort or pool time steps as independent episodes",
        "worker_policy": "One sequential worker per source/step/method/mode; engine reset for each episode as in the original runner",
        "archive_10ms_D_wall_seconds": reference["archive_10ms_D_wall_seconds"],
    }
