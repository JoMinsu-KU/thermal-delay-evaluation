# Supplementary command diagnostics

## Figure S1. First command divergence in the selected C case

`outputs/figures/first_command_divergence.pdf` and `.png` retain the archived C command diagnostic. The detailed protocol, conditional interpretation, and score-margin discussion are in `methods/selected_C_detail.tex`. C uses a different frozen direct scorer from D. Its episode was selected post hoc; it is not a representative performance estimate for the primary comparison.

## Figure S2. FV-MPC commands around the first deadline miss

![Figure S2](outputs/figures/deadline_command_response.png)

Applied commands for all four heaters during 6.4–9.0 s of episode `l2_240070`, under the original interleaved replay at a 10-ms plant cap. Solid orange curves use zero fallback; dashed blue curves use hold fallback. Dotted vertical lines mark deadline expirations at 6.8 and 7.0 s. Shading spans the interval from the first expiration to the next nonfallback application at 7.177863943 s. Both policies have the same request, deadline, and completion schedules. Curves use actual saved application timestamps with right-continuous holds, without interpolation or new simulation.

Immediately before 6.8 s, the applied command is [0.64, 0.68, 0.64, 0.68]. Zero fallback resets all four values to zero, whereas hold retains them. The next nonfallback command is [0.04, 0, 0, 0] with zero fallback and [0.68, 0.68, 0.64, 0.68] with hold. Subsequent commands are recalculated from each condition's evolving state, so the figure shows the deadline intervention and its downstream feedback effects. It does not isolate every contribution to the final temperature difference or establish a safe hardware policy.

The episode is unchanged from main Figure 5: it was selected as the lower median by zero-delay FV-MPC cost, before the hold curve was added. `data/minor_revision_r2/deadline_command_evidence.json` records the source hashes and author audit scope. The compact script reconstructs command penalties and the plot; it does not repeat the omitted raw trace/seal audit.


## Figure S3. Residual architectures

Shared and Small numerical candidate scorers. Shared uses a frozen released 2B backbone and released LoRA at its selected warmup checkpoint; new thermal encoders/head supply adaptation. Small uses a three-layer width-128 Transformer. Dimensions and losses are in METHODS Sections 1–2. The comparison does not isolate pretraining or backbone-size effects.

## Figure S4. Numerical sensitivity of selected C

(a) Temperature-field RMSE between successive refinements, sampled at 1 Hz with original applied commands fixed. (b) Costs when decisions are recomputed at each plant cap. Colors identify normal-path delay conditions. This single episode was selected post hoc; the changing best delay does not identify an optimal physical delay. C uses a different frozen scorer from D.

## Figure S5. Expiration placement and conditional costs

(a) All saved FV expiration times for each of seven fixed replay settings. Every episode and both policies share that setting's schedule. (b) FV mean cost increase relative to zero delay under zero and hold fallback. The same 24 nominal episodes are used throughout, with a 10-ms plant cap. Timing placement, counts, commands, and thermal states differ together; the figure is descriptive and does not isolate a causal effect of miss position or estimate a random-timing distribution.


## Figure S6. Small initialization sensitivity

`small_seed_cost_comparison.png/pdf`: mean Small-minus-fixed-Shared cost with per-seed conditional 95% episode-bootstrap intervals. Negative values favor Small. Percent labels use the corresponding Shared mean as denominator. Seed 6060 is original; 6061–6063 are additional initializations. Seed 6062 fails the validation improvement criterion and is retained. All 24 episodes and the original timing sequence are reused; intervals omit uncertainty across seeds and timing streams. The new models are tested only at the 10-ms plant cap.
