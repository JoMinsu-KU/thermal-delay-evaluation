# Conditional empirical latency Monte Carlo protocol

Frozen implementation date: 2026-10-13. This package contains an unexecuted follow-up experiment; local checks use synthetic fixtures and read archived records only.

## Question and estimand

How often do the five fixed implementations change their aggregate ranking when each of the same 24 held-out nominal episodes receives an independent IID sequence from the implementation's archived latency pool? Does replacing FV-MPC's expired-request zero command by hold change the conditional distribution of cost and rank?

One replicate contains all 24 episodes for five zero-fallback methods and FV-MPC with hold. All cost rankings use the mean across the same 24 episodes. Random timing streams are independent across replicate, episode and method. The two FV policies use identical potential random draws for each request and matched pool. If feedback visits a different fallback path or token pool, the analyzer reports the resulting path and clock divergence. It never selects timing from the wrong path merely to keep clocks equal.

## Frozen sample size and baseline gate

The default is 16 replicates: 2,304 new closed-loop cases plus 240 original zero-delay/interleaved-replay reproductions, total 2,544. `--replicates R` changes the budget only before the first protocol is saved; R must be at least two. Choose at least 64 for tighter Monte Carlo probability precision if resources permit (9,216 new plus 240 baseline cases). At an observed probability of one half, Wilson 95% half-widths are about 0.22 for R=16 and 0.12 for R=64. Neither resolves timing-pool estimation or hardware-population uncertainty.

The 240-case legacy reproduction gate must pass before new cases. The baseline retains the original latency chooser, event order and tolerance (absolute 1e-10, relative 1e-8 for checked numerical metrics; exact semantic events). A failed reproduction halts execution. Finish the saved design regardless of effect direction. A new budget after inspecting outcomes is a separate follow-up protocol and must be disclosed; it cannot resume the old run.

## Timing sampler

1. Read and verify all three original sealed journals with `read_latency`. Its interleaving is preserved as a stable row-index mapping, not interpreted as temporal adjacency.
2. Each method contributes 30,000 normal-path records and 3,000 records for each of two fallback paths before exclusion of timing-error rows. Filter by the exact fallback reason, then by minimum absolute difference to the request token count, including tied nearest counts. Stop if the requested path is absent. There is no normal-latency replacement for a missing fallback path.
3. Draw uniformly with replacement from the matched pool. Counter-based SHA-256 draws use seed, method, episode, replicate, matched-pool membership hash and global request index. Rejection removes modulo bias. The counter has no mutable state: resuming or resetting a case reproduces its potential draws. The request index is the count of issued requests, not the nominal tick count.
4. Record source interleaved index, original repeat/request, row hash, full source row, path, token gap, and consumed delay. Save the potential 600-request plans for all observed path/token pool families. New off-grid token queries still use the same minimum-gap rule and counter, and can be independently reconstructed.
5. Copy the verified timing pools into sealed output folders for independent offline reanalysis. The original archived model/data/source files remain read-only.

IID destroys recorded temporal correlation and pools the three existing timing trials. This is a conditional empirical resampling experiment, not a new hardware measurement, a block bootstrap, or a state-dependent latency model. The 128-input timing ring limits coverage and is not expanded here.

## Unchanged execution and safeguards

Frozen checkpoints, alpha, observations/noise, PINO gate, FV prediction solver, nine candidates, 100-ms decisions/deadlines, 10-ms maximum plant step, sensor timing, and event-bounded cost quadrature remain unchanged. Simulate the full feedback trajectory for each draw plan; do not resample existing final costs and call this Monte Carlo. Hold changes only the command at an expired request; busy execution, skipped ticks, late-result rejection and other fallback paths remain unchanged. No training, real actuation, network delay, or physical safety validation is included.

The launcher requires original Linux H200, `/home/jovyan/.conda/envs/jev`, offline model cache and exactly one visible GPU. It sources the original resource settings, shares `locks/l0.lock`, verifies 1,245 pinned files before/after execution, checks imported source hashes and environment identity, seals each completed attempt, and rejects incompatible resume. Original weights must be present; this package does not download or redistribute them.

## Analysis

- Independently reconstruct cost, tracking RMSE, command effort/increments, fallback duration and event coverage from saved traces, integration quadrature and final fields. Verify every sampled row against the copied timing pool and regenerate its counter draw.
- Report all raw episode metrics, each replicate's five-method ranks under both FV-zero and FV-hold substitution, strict pairwise reversals relative to zero-delay ranking, FV worst/rank deterioration, and first-place proportions. Differences within 1e-10 are ties, excluded from strict reversal events.
- Calculate conditional Monte Carlo proportions and Wilson 95% intervals across complete replicates only. Also report replicate-level cost-effect means, SD, median, 2.5/97.5 percentiles and range. These percentiles describe distribution spread, not a confidence interval for a population effect.
- Report paired FV hold-minus-zero effects and whether each paired run retained identical consumed timing clocks. Partial-run summaries are provisional; final interpretation requires the entire frozen design and all integrity gates.

All uncertainty statements condition on the archived pools, IID assumption, the same 24 episodes and frozen checkpoints. Do not infer general algorithm superiority, physical reliability, pretraining benefit, or performance across hardware/training seeds.

## Outputs

`protocol.json`, source/harness snapshots, before/after inventory and environment records, copied `timing_pools/`, all case traces (including decision scores/masks when returned by the fixed engine), sampled rows/plans, metric quadrature, final fields, seals, and `analysis/` are included in the returned results ZIP. `STATUS.json` distinguishes preflight, phase completion, execution completion and failure; `analysis/summary.json` must say `COMPLETE_VERIFIED` before final scientific reporting.
