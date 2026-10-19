# Saved-log response to M2, M3 and M7

## Scope and reproducibility

This is a post hoc analysis of existing saved trajectories. Definitions were
written in `DEFINITIONS.md` before computing the summaries, after the reviewer
comments and original aggregate results were known. No simulation, inference,
training, model selection, or new hardware measurement was performed.

The analysis verified 552 COMPLETE seals, including 336 FV zero/hold cases
(seven fixed timing settings times 24 episodes times two policies). The
physical-response comparison uses the original five zero-delay and five
replayed pipelines plus original-replay FV hold: 11 conditions times 24
episodes. The additional 288 FV cases support the timing/recovery analysis.

Compact inputs contain all 552 spatial-mean temperature curves (601 samples
each), all application records for the 336 FV cases, and per-case metadata.
They total approximately 6.4 MB. `reproduce_compact.py` independently rebuilt
all event, episode and aggregate CSVs with exact byte matches to extraction
from the sealed traces. NumPy is the only analysis dependency. Matplotlib is
needed only for the optional figure.

```bash
python reproduce_compact.py --output reproduced
python make_mechanism_figure.py
```

## M2: reset and rate-limited command restoration

All 196,752 ordinary FV application transitions across the 336 cases changed
at most one heater by at most 0.04. Consequently, the decision-count lower
bound for restoring a pre-reset vector from zero is

`sum_j ceil(max(0, u_j - 1e-8) / 0.04)`.

It is not the largest single-heater count. The observed bounds ranged from
1 to 100 positive coordinate moves. At a nominal 100-ms decision interval,
100 moves occupy 100 decision slots; this is not an exact 10-s wall-time
bound because ready times vary and decisions can be skipped.

For the already selected example episode at the first original expiration
(6.8 s), the previous vector was [0.64, 0.68, 0.64, 0.68]. Restoring it from
zero requires at least 66 positive coordinate moves. The next expiration
occurred at 7.0 s, so the restoration diagnostic for the first event is
censored at 0.2 s. Hold preserved that vector immediately. This explains a
plausible amplification mechanism but does not quantify how much loss is
caused by the rate constraint: no rate-limit ablation was performed.

Full-vector restoration means that every component first regains at least
its own pre-expiration level. It does not mean temperature settling or a
desired target that the controller ought to pursue. Changing thermal state
may make a smaller command appropriate. Follow-up stops at the next
expiration or 60-s horizon.

There were 1,128 deadline events under each policy. Of the zero-policy
events, 715 had a nonzero prior command; 324 restored the full vector before
the censoring boundary and 391 did not (377 censored at the next expiration,
14 at the horizon). The other 413 zero-policy events already had a zero
vector. All hold events retained their previous vector at expiration by
construction. Neither event count is a count of independent experimental
units.

| Setting | Nonzero resets | Restored | Censored | Median observed restoration (s) | Observed range (s) |
|---|---:|---:|---:|---:|---:|
| Original | 123 | 65 | 58 | 0.478 | 0.175–2.278 |
| R1B1 | 63 | 22 | 41 | 1.675 | 0.287–8.775 |
| R1B2 | 154 | 78 | 76 | 0.927 | 0.282–6.573 |
| R2B1 | 192 | 91 | 101 | 2.079 | 0.287–6.378 |
| R2B2 | 41 | 27 | 14 | 1.474 | 0.376–32.577 |
| R3B1 | 84 | 41 | 43 | 2.276 | 0.278–4.778 |
| R3B2 | 58 | 0 | 58 | — | — |

These are descriptive event summaries with explicit censoring. Completed-only
times are not an estimate of overall mean recovery. Per-episode aggregations
are supplied separately in `deadline_episode_metrics.csv`; no confidence
interval or test treats events as independent.

The original archived implementation assigns zeros at expiration. Its data
generator describes zero as an explicit shutdown/cold-start mode. This
supports describing zero as the inherited benchmark behavior, not as an
optimal heating policy or a validated hardware safety policy. No documented
comparative optimization of the original fallback choice was found in these
files. Discarding a late result while retaining the current command matches
the tested hold action only if the same busy-until-ready and skipped-tick
semantics are preserved. This does not establish equivalence to other named
scheduling strategies. Apply-late or queueing changes the timing/state
contract and remains untested.

## M3: seven fixed schedules, not 24 timing realizations

Within each timing setting, all 24 episodes and both FV fallback policies
share exactly the same complete request/ready/deadline schedule. This was
verified by equality of the request schedule hashes, not only by matching
deadline counts. The original schedule expires at 6.8, 7.0, 25.3, 26.2,
51.3, 54.8, 55.8 and 56.2 s.

| Setting | Expirations per episode | First expiration (s) | Mean prior command, zero | FV zero cost increase (%) | FV hold cost increase (%) |
|---|---:|---:|---:|---:|---:|
| Original | 8 | 6.8 | 0.122 | 25.02 | 2.35 |
| R1B1 | 6 | 8.7 | 0.137 | 18.42 | 2.19 |
| R1B2 | 9 | 3.5 | 0.194 | 62.47 | 5.63 |
| R2B1 | 9 | 2.3 | 0.250 | 79.34 | 5.63 |
| R2B2 | 2 | 1.4 | 0.265 | 11.70 | 4.50 |
| R3B1 | 7 | 2.3 | 0.119 | 19.43 | 6.46 |
| R3B2 | 6 | 2.1 | 0.141 | 46.79 | 5.24 |

Prior command is the mean over four heaters and all expiration events,
followed by equal weighting of the 24 episodes. It includes expirations at
an already zero vector. Cost increases compare each setting's mean FV cost
with the common zero-delay mean. The same 24 episodes are reused throughout.

`figures/deadline_schedule_cost.pdf` shows the temporal positions beside
these conditional cost effects. The earliest first expiration (R2B2, 1.4 s)
does not have the largest effect; it has the smallest zero-fallback increase
among these settings. Expiration count, spacing, command level and thermal
state also differ. No causal share or general timing distribution can be
inferred from seven such settings. The matched zero/hold intervention is
identified under each saved schedule; a distribution over new timing
sequences and a general reversal probability are not. The reviewer's Monte
Carlo request therefore remains an additional-experiment item.

## M7: physically interpretable response and a material tradeoff

All numbers below are equal-weight means over 24 nominal episodes, except
the count settled. Error and overshoot use the spatial-mean temperature,
not the hottest cell or the spatial-field RMSE. Settling uses the first
10-Hz sample after which every remaining sample through 60 s is within
1 K of the setpoint. Non-settling episodes are censored at 60 s.

| Pipeline | Timing/policy | Final-10-s MAE (K) | Mean overshoot (K) | Settled by 60 s |
|---|---|---:|---:|---:|
| PI | zero delay | 2.773 | 4.799 | 7/24 |
| PI | original replay | 2.774 | 4.801 | 7/24 |
| FV-MPC | zero delay | 0.998 | 4.349 | 18/24 |
| FV-MPC | original replay, zero | 0.511 | 2.130 | 21/24 |
| FV-MPC | original replay, hold | 1.089 | 4.641 | 18/24 |
| PINO-MPC | zero delay | 1.103 | 4.572 | 19/24 |
| PINO-MPC | original replay | 1.127 | 4.606 | 18/24 |
| Shared-Residual | zero delay | 1.080 | 4.512 | 19/24 |
| Shared-Residual | original replay | 1.157 | 4.645 | 18/24 |
| Small-Residual | zero delay | 1.089 | 4.536 | 19/24 |
| Small-Residual | original replay | 1.116 | 4.565 | 18/24 |

The mean initial spatial-mean error was 26.142 K. Depending on condition,
80.46–86.99% of the sampled squared spatial-mean tracking error accumulated
in the first 10 s. This supports interpreting the approximately 10-K
whole-episode RMSE as strongly influenced by warm-up. The percentage is a
mean-temperature diagnostic, not a decomposition of the spatial-field
RMSE used in the main cost.

Hold lowers the published finite-horizon composite cost relative to zero
fallback, but it does not dominate these response measures. For the
original replay, zero fallback has smaller late-horizon mean-temperature
error and overshoot and more episodes satisfying the 1-K settling criterion.
The lower integral cost under hold reflects a tradeoff over the whole
trajectory rather than uniform improvement in all control objectives.
Final-10-s error should not be called steady-state error without qualification.

## Suggested manuscript figure caption

Saved FV-MPC expiration schedules and conditional cost effects. (a) Expiration
times are identical across the 24 nominal episodes and both fallback policies
within each setting. (b) Mean cost increase relative to the common zero-delay
FV baseline. The original replay and six within-trial blocks are fixed timing
conditions. Differences across rows are descriptive; they do not estimate a
distribution over independently sampled timing sequences.

## What this analysis identifies

- Exact fallback command changes and subsequent recorded candidate moves.
- The finite-horizon cost contrast of zero and hold under each matched saved
  timing schedule, with the existing model and controller implementation.
- Descriptive command-restoration and mean-temperature response measures.

It does not identify the causal fraction of cost due to the rate constraint,
an optimal fallback policy, hardware safety, behavior under new random timing
draws, a reversal probability, or asymptotic steady-state performance.
