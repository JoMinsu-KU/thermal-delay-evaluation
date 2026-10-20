# Protocol and analysis details

The main manuscript presents the controller design, policy comparisons, and thermal interpretation. This companion retains the detailed chronology, numerical definitions, execution checks, and historical development context.

## Study chronology (Table S3 in the guide)

The original checkpoints and calibration were fixed before held-out testing. The paper synthesizes those results retrospectively. Integration-step, replay/hold, Monte Carlo, and Small initialization protocols were designed after the original outcomes and fixed before their respective executions. Recovery, score-margin and thermal-response summaries were post hoc analyses with definitions fixed before extraction. None of these analyses was externally preregistered. Earlier cohorts A/B and the deliberately selected case C remain supporting development evidence.

## Detailed definitions

`methods/evaluation_protocol_detail.tex` retains the FV candidate-cost difference equation and eligibility convention, latency boundary and environment, input-token definitions, replay matching, statistical estimands, baseline tolerances, event ordering, and command/response definitions. Its labels and references are historical source identifiers; use TABLE_FIGURE_MAP.md for the current manuscript mapping. METHODS.md retains architecture, losses, optimization and selection details. Near-tie margins appear in guide Table S2 and the compact score records. Margins describe the corrected score scale, not numerical-error bounds.

## Execution checks and their scope

- Primary numerical reruns: 720 runs; all 240 10-ms baseline cases reproduced before refinement. Baseline metric tolerances were absolute 1e-10 and relative 1e-8, with semantic request/application agreement excluding wall time and continuous diagnostics.
- Fixed replay/hold batch: 1,128 completed executions including 240 baseline reproductions. Raw cost reconstruction error was at most 4.58e-16; 602,544 consumed latency rows matched the journals. All 168 FV policy pairs had matching actual request/deadline schedules.
- Monte Carlo batch: 9,456 completed executions including 240 baseline reproductions and 9,216 new runs. Raw cost reconstruction error was at most 6.67e-16. All 1,536 FV policy pairs had matching actual request/deadline schedules. Full raw-file hashes and source seals were checked; deterministic draw validation sampled 384 cases (all 64 repetitions and six conditions for the first episode), rather than every trajectory.
- The saved `inference_failures` field is initialized but not incremented by the evaluated scheduler. It is not evidence of a measured zero inference-failure rate. Completion, temperature checks and deadline counts are distinct observations.
- Compact reproduction verifies saved cost components, contrasts, rankings, intervals and figures. It also checks every FV Monte Carlo schedule and draw from the compact original-record projection. It does not rerun raw-field quadrature, other methods' source journals, training, or closed-loop simulations. Author receipts for those checks are included separately.

All scientific input files are unchanged by the balanced manuscript revision. The status table, score-margin table and conceptual related-work comparison were relocated, and main figure/table numbering was updated. Failed validation seed 6062, unfavorable episode outcomes, thermal tradeoffs and nonmonotone refinement effects remain reported.


## All-case FV schedule extension

The October 19 minor revision adds all 3,072 original FV case logs as compact schedule/draw projections, the complete FV timing pool, and the protocol-pinned sampler/scheduler subset. The independent verifier checks all 1,536 pairs and 1,824,628 consumed draws, extending the earlier 384-case sampled resampling audit. Exact all-case comparison to the original ZIP was performed before release. `data/monte_carlo_schedule_audit/README.md` documents the fields, integer-nanosecond checks, source hashes and scope. Original costs and other existing data are unchanged.
