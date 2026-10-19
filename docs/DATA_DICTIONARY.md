# Current manuscript data

For the 10 October 2026 manuscript, use `python -B scripts/reproduce_all.py`; current results appear in `supplement/outputs/`. See [the complete map](../supplement/TABLE_FIGURE_MAP.md), [methods](../supplement/METHODS.md), and [supplement README](../supplement/README.md). The new 1,128-case batch reuses 24 episodes. Its full raw records are not included. Original raw assets at v1.0.0 cover the earlier primary/numerical evidence only.

The material below documents the original v1.0.0 data and tools, which are retained for compatibility.

---

# Data dictionary

All temperatures are in degrees Celsius unless a difference or RMSE is explicitly labelled K. Applied commands are dimensionless, with four heater components. Time fields ending in `_ns` are integer nanoseconds; `_s` and `_ms` denote seconds and milliseconds. JSON `null` and empty CSV cells mean unavailable/not applicable, not zero.

## Cohorts and methods

- **A:** zero-delay development comparison on eight validation episodes, including six normal and two initially hot episodes.
- **B:** checkpoint selection on eight reused validation episodes, five synthetic delays, and five scorer families. Candidate trajectories do not create independent episodes.
- **C:** one post hoc selected episode (`l2_090006`) and three normal-path delays, using a direct scorer.
- **D:** residual-comparison cohort. Original records include 24 normal and eight initially hot test episodes, plus 16 OOD episodes. New numerical validation uses only the 24 normal test episodes.
- D method IDs: `pi`, `fv_mpc`, `pino_mpc`, `shared_physics` (Shared-Residual), `small_physics` (Small-Residual).
- `zero` means zero-delay execution. `delayed` is the archived method-specific latency replay, called “replay” in the paper.

## Small repository data

| Path | Contents |
|---|---|
| `data/primary/{mode}_{method}.json` | Original 48-episode result list for one D method/mode; authoritative structured records |
| `data/primary/episode_metrics.csv` | Flattened convenience export of 480 original method/mode/episode records |
| `data/primary/latency_quantiles.csv` | 60 rows: five methods × three paths × three repeats plus pooled summaries |
| `data/primary/latency_audit.json` | Measurement boundaries, environment summary, consumed sequence counts, and timing interpretation |
| `data/primary/fv_deadline_events.json` | FV deadlines, source timing IDs, and zero-command application schedules |
| `data/development/A/` | Paired episode metrics and checkpoint-training summaries |
| `data/development/B/` | Selection records, all 600 candidate metrics and 80 PI/PINO reference metrics; held-out summaries remain separate |
| `data/selected_C/applied_commands.csv` | Applied time and four heater values for three original delay traces, used in Figure 3 |
| `data/selected_C/first_divergence.json` | First differing requested commands, scores, gate masks, and application times |
| `data/numerical/case_index.csv` | All 741 numerical runs, their cohort, cap, mode, episode, and relative raw-data directory |
| `data/numerical/extended_metrics.csv` | All 732 extended-case conditions and metrics; D=720, fixed C=12 |
| `data/numerical/initial_C_metrics.csv` | Nine earlier C closed-loop metrics |
| `data/numerical/contrasts.csv` | 27 comparisons across three caps; paired intervals, ratio of means, wins/losses |
| `data/numerical/refinement.csv` | Change between caps for each D method/mode |
| `data/numerical/episode_sensitivity.csv` | 720 paired episode/step comparisons, including all adverse outcomes |
| `data/numerical/trace_sensitivity.csv` | Command disagreements and temperature-trajectory summary differences |
| `data/numerical/fixed_replay.csv` | 12 C field/cost comparisons across cap pairs, sampled at 1 Hz |
| `data/protocol/` | Preserved experimental settings with machine-identifying metadata removed |
| `data/source_manifest.jsonl` | Original source identifier, original hash, released path/hash, and transformation for each exported source file |

`episode_cost` is the dimensionless composite objective: tracking/10000 + 0.5 × final normalized squared error + 0.002 × command effort + 0.02 × command-increment mean square. It is not physical energy consumption. `max_temperature_c` is a sampled maximum at the original cost/peak endpoints. `new_violation` is distinct from `initial_violation`; initially hot episodes are not pooled with normal episodes.

`ratio_of_means_pct` equals 100 × (mean_B / mean_A − 1). `mean_individual_relative_pct` instead averages individual episode ratios; these quantities differ. `B_wins` counts B costs below A by more than 1e-10. Confidence intervals condition on fixed trained systems and reused timing records.

## Raw release data

### `raw_data/primary/`

- `D/{mode}/{method}/{episode}/metrics.json`: original episode metrics.
- `D/.../control_trace.jsonl`: request, application, deadline fallback, and busy-skip events.
- `latency/{method}_{repeat}/requests.jsonl`: journal wrapper (`identity`, sequential `index`, `row`, `sha256`); the hash covers `json.dumps(row, sort_keys=True, allow_nan=False)` encoded in UTF-8.
- `C/`: original selected-delay records and control traces.

Within latency rows, `e2e_ms` is warm end-to-end wall-clock inference time. `gpu_compute_ms` is CUDA-event elapsed time spanning host/GPU activity and is not pure GPU kernel time. Normal, missing-observation, and no-feasible paths are conditional workloads. `input_id` refers to reused archived inputs, not independent episodes.

### `raw_data/numerical/extended/{case}/`

- `condition.json`, `metrics.json`: run condition and measured outcome.
- `trace.jsonl.gz`: asynchronous control events; `no_change` identifies unchanged command/fallback state.
- `truth.csv.gz`: actual-plant summaries at 100 Hz in C and 10 Hz in D; D sensors still emit at 100 Hz.
- `quadrature.npz`: interval `dt_ns`, `end_ns`, squared tracking error `mse_k2`, command mean square `command_mse`, fallback state, and sampled `peak_c`.
- `final_T64.npy`: final 64 × 64 FP64 field.
- `fields_1hz.npz` (C only): `time_ns` and temperature fields for all 61 integer-second samples.
- `baseline_comparison.json` where present: original baseline comparison outcome.

### `raw_data/numerical/initial_C/{case}/`

Nine earlier closed-loop cases: metrics, compressed events, `truth_100hz.csv.gz`, and final fields. No interval quadrature was saved. The raw audit uses the reported tracking RMSE for these nine cases and identifies that limitation in its output.


## Small initialization update

`seed`: 6060 original or 6061–6063 new; `mode`:zero or delayed (original replay); `episode`:one of 24 reused nominal test IDs. `episode_cost` is the saved full-field composite cost. `cost_reconstructed` and component columns come from the author's raw-record audit, not fresh simulation. `alpha` is validation selected. `validation_quality_pass` checks candidate-cost improvement with zero invalid selections/unexpected fallbacks; 6062 fails and remains included. `small_relative_to_shared_pct` uses Shared as denominator; negative favors Small. `conditional_episode_ci_*` resamples 24 matched episodes, excluding seed-population and new-timing uncertainty. `source_provenance.json` retains raw metric hashes and archive identity; omitted raw inputs are not available for this compact audit.


## Monte Carlo update

`phase`: baseline or monte_carlo. `replicate`: zero-based 0–63 for new runs (blank for baseline). `episode`: one of 24 reused nominal IDs. `fallback_policy`: zero or hold (hold only for FV). `episode_cost`: stored full-field composite cost. `J_tracking`, `J_terminal`, `J_command`, `J_increment`: saved cost components, summed by the compact check. `relative_change_pct`: 100*(repetition mean / method's zero-delay mean - 1). `relative_hold_vs_zero_pct`: 100*(FV hold mean / paired FV zero mean - 1). `q025`/`q975`: empirical repetition percentiles, not confidence limits for a mean. Wilson limits apply to conditional repetition-level events. Ranks/ties use absolute tolerance 1e-10. `inference_failures` is inactive in this harness and cannot estimate reliability. `overshoot_k` is a spatial-peak diagnostic sampled at 10-ms/event endpoints; it is not the 10-Hz spatial-mean overshoot used for original thermal-response analysis. Audit receipts and raw metric hashes describe omitted full records.
