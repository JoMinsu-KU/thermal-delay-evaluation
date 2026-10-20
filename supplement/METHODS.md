# Supplementary implementation details

Implementation evidence was read from the archived source. Here S means the original snapshot, R is its final D run, and L is its parent PINO run. Source locators document provenance; they are not links to bundled model weights or training inputs. The stored field `regret` is called FV candidate-cost difference in the revised manuscript.

## 1. Open-Jev reuse, initialization, and selected warmup modules

| Component | Verified construction | Evidence |
|---|---|---|
| Model lineage | Qwen/Qwen3.5-2B revision 15852e8c16360a2fea060d615a32b45270f8a8fc; Open-Jev revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 | R/resolved_config.json; S/models/Open-Jev-2B/package/checkpoint/model.json |
| Retained | Text backbone with released checkpoint LoRA, tokenizer, task-token embeddings, model ID/revision | `snapshot/src/ojpoc/policy.py:106`; `snapshot/src/ojpoc/l2/models.py:46` |
| Not used in Shared | Released scalar head is loaded by load_original but not attached to SharedPolicy. The vendor wrapper retains full.model.language_model and deletes the full vision/generation wrapper; Shared does not use native vision, MTP, language-generation head, or candidate text serialization | `snapshot/third_party/Open-Jev/jev/model.py:87`; `snapshot/src/ojpoc/l2/models.py:54` |
| New observation encoder | Sensor Conv1d 48→128 plus 3 residual causal blocks (kernel 5, dilation 1/2/4, GroupNorm 8), pooled to 8 tokens. Thermal Conv2d 6→16→32→64→128, 8 tokens. PLC 12→128→256, 2 tokens. Passport+recipe 14→128→512, 4 tokens. Type embeddings N(0,0.01²) | `snapshot/src/ojpoc/l2/models.py:14` |
| New Shared context | Linear 128→backbone hidden width; summary token N(0,0.01²); task string “Choose the thermal control action.”; final valid summary hidden state→256 with GELU. Active length ≤39, padded to 64; BF16 backbone input, FP32 context/head | `snapshot/src/ojpoc/l2/models.py:54` |
| New candidate networks | Action 80→64→64; physics 32→64→64; combined 384→128→1, with GELU. Residual scalar output layer weight/bias explicitly zero-initialized. Other new layers use framework constructor initialization | `snapshot/src/ojpoc/l2/models.py:63` |
| Small counterpart | Same new observation/candidate paths; 3 newly initialized Transformer layers, width 128, 4 heads, FF 256, dropout 0, norm_first=True; no pretrained backbone or LoRA | models.py 63–69 |
| LoRA | Rank 8, alpha 16, dropout 0, bias none; targets o_proj, in_proj_qkv, v_proj, q_proj, k_proj, out_proj | `snapshot/models/Open-Jev-2B/package/checkpoint/adapter/adapter_config.json:1` |
| Warmup training | Shared trains every nonbackbone parameter. Its base and released LoRA remain frozen. Small trains all parameters because it has no backbone-prefixed module. Gradients still pass through the frozen Shared backbone to train its input encoders | `snapshot/src/ojpoc/l2/residual.py:140` |
| Retained checkpoint evidence | Both selected_step=["warmup",50]. Shared warmup LoRA gradient=0 throughout; warmup_updates lists only nonbackbone modules. Later adapt_updates has 120 LoRA tensors, max change 0.0053252899, discarded after restoring fit_best | `snapshot/runs/L2/l2_auto_20260928T043858Z_2f3e14_pilot/08_shared_physics/training.json:1`; `snapshot/runs/L2/l2_auto_20260928T043858Z_2f3e14_pilot/09_small_physics/training.json:1`; residual.py 186–193 |

The original Yes/No scalar-head initializer in the vendor wrapper is not the initializer of the final thermal scorer.

## 2. Residual training specification

| Item | Actual final-run setting |
|---|---|
| Data | Original 480 anchors, 470 expert-feasible, plus 384 common rows = PI/PINO × 8 training episodes × 24 anchors. Collection alternates zero/38-ms synthetic delay by episode; observed-state FV labels |
| Batch | One batch of 16 decisions: 8 original + 8 common rows, sampled with replacement from feasible-label rows; random candidate permutation each step. It is not 16 accumulated single-sample forwards |
| Matching | All 450 stored batch hashes match across learners, zero mismatches; sampler seed 6060+211. Shared training-data hash d47c82b0d6cc42a86ad16d80373ba4b82f1f6645789278520a2fcdbf58e4d6b3 |
| Optimizer | AdamW, decay 0.01, gradient norm clip 1, foreach=False. New modules LR=1e-4 in both phases; adaptation LoRA LR=3e-5. New optimizer each phase. No LR ramp/scheduler in residual.py |
| Budget/selection | 50 warmup + 400 adaptation steps; validation every 100 or phase end; strictly lower offline selection_score updates fit_best. Checks at warmup 50 and adaptation 100/200/300/400 |
| Correction | Training alpha=1. After restoring fit_best, select alpha∈{0,0.5,1} with zero/38-ms validation rollouts. Selected Shared=1, Small=0.5 |
| Scale | σ=1.828003659389636e-5 from 5,746 feasible training candidates; soft-label temperature τ=4.5804771195429765e-5 from original training data |

Sources: `snapshot/src/ojpoc/l2/residual.py:39`, `snapshot/src/ojpoc/l2/residual.py:108`, `snapshot/src/ojpoc/l2/rollout.py:164`; R/05_residual_data/statistics.json and train/metadata.json; both arms' training.json and selection.json. Alpha selection minimizes reference-relative adverse events, worst all/normal PI/PINO cost ratio, mean ratio, then lower alpha.

### Exact alpha-selection rule

Each alpha in {0, 0.5, 1} is evaluated on the same 16 validation episodes (12 normal and four initially hot), under zero and 38-ms synthetic delays. The two references are PI and PINO-MPC. Selection minimizes the tuple `(adverse count, worst ratio, average ratio, alpha)` in that order. For each reference and matched delay/episode pair, the adverse count increments once if any of these exceed the reference: the new-violation indicator, violation duration by more than 1e-8 s, or inference-failure count. It is not a count of deadline misses.

The eight cost ratios are ratios of mean episode costs, formed from two references × two delays × two strata (all validation episodes and normal episodes). The reference mean has a numerical floor of 1e-8. The second tuple entry is the maximum of these eight ratios, and the third is their arithmetic mean. A remaining tie favors the lower alpha. Recalculation from the saved outcomes selects Shared=1 and Small=0.5 and reproduces every stored candidate key. See `data/minor_revision/*alpha*.json` and `scripts/reproduce_method_details.py`.

### Implemented loss equations

Let \(F_i\) be valid FV-eligible candidates, \(V_i\) valid candidates, and \(C_F(x)_a=x_a-|F_i|^{-1}\sum_{b\in F_i}x_b\).

\[
s_{ia}=-\frac{Q^P_{ia}-\frac19\sum_{b=1}^9Q^P_{ib}}{\sigma}
       -4\alpha\tanh r_\theta(i,a).
\]
The score's PINO center includes **all nine positions**; the regression center includes **FV-eligible valid positions**.

\[
q_{ia}=\operatorname{softmax}_{a\in F_i}(-Q^{FV}_{ia}/\tau),
\qquad
p_{ia}=\operatorname{softmax}_{a\in V_i}(s_{ia}\sigma/\tau),
\]
with q=0 outside F. Then
\[
L_{\rm residual}=
\operatorname{mean}_{i,a\in F_i}
\operatorname{SmoothL1}\!\left(C_F(s_i)_a+
C_F(Q_i^{FV})_a/\sigma\right)
-\frac{0.2}{B}\sum_i\sum_{a:q_{ia}>0}q_{ia}\log p_{ia}.
\]
SmoothL1 uses default beta=1. Regression averages all masked candidate elements across the batch; CE divides by decision batch size. Valid FV-infeasible candidates remain in the CE denominator with zero target mass. See `snapshot/src/ojpoc/l2/residual.py:14`, `snapshot/src/ojpoc/physics.py:105`, `snapshot/src/ojpoc/l2/cache.py:61`.

**Historical A/B objective differs:** valid-mask soft-target CE +0.2 mean softplus pair-ranking loss for valid FV-feasible pairs with \(|Q_a-Q_b|>10^{-6}\). Historical Shared uses new-module LR 3e-4 in warmup and 1e-4 in adaptation, LoRA 3e-5 in adaptation, and a 5% per-phase LR ramp. Do not transfer these settings to D. Sources: `snapshot/src/ojpoc/policy.py:63`, `snapshot/src/ojpoc/l2/training.py:72`.

## 3. PINO data, training, and selected validation

| Item | Verified setting/result |
|---|---|
| Data | Parent L1: 40 train / 8 validation episodes; 5,000 sampled train windows, 128 validation windows; true contiguous FV64 trajectories restricted to 32×32 and actual applied commands. 205 paired states, each with hold and another valid action, independently re-solved in FV64 from the same lifted true T32 and actuator |
| Model | FP32, 3 layers, width 32, Fourier modes 8×8, reflection pad 8, 14 input channels, 90 conditions, 20 outputs, zero-initialized final conv, balance projection |
| Attempted budget | 500 supervised steps, microbatch 16×4 accumulation=64 windows/update, LR 5e-4; then 100 paired steps, 8 pairs×2 accumulation=16 pairs/32 branches, LR 1e-4. AdamW decay 0.01, clip 1, new optimizer per phase, 5% LR ramp |
| Selection | Lowest final-horizon spatial RMSE on 128 validation windows, checks every 100 and phase end, initial model eligible. Selected **supervised 200**; frozen transfer to L2 |
| Selected window metrics | Horizon RMSE **0.0381711449 K**; one-step RMSE **0.0026894664 K**; all-horizon RMSE **0.0236755634 K**; maximum peak prediction error **0.9921951294 K** |
| Selected candidate diagnostics | Candidate-difference RMSE **0.0622588826 K**; significant-pair order accuracy **0.9526474387**, 2,323 pairs; candidate regret **6.8259375071e-7**; candidate peak error **0.9605281586 K** |
| Frozen PINO hash | 03018bae98a0de1f43496fc5489fe46ba331bfc0c410449e3b5ba7344e90c3f4 |

Sources: `snapshot/src/ojpoc/training.py:18`, `snapshot/src/ojpoc/training.py:93`, `snapshot/src/ojpoc/training.py:135`; `snapshot/src/ojpoc/operator.py:33`; `snapshot/runs/L1/l1_pilot_s2026_20260923T154422Z_e829d6/07_pino_balance/training.json:6060`; L/04_operator_data/result.json; L/01_generate/result.json; R/01_frozen_inputs/result.json.

The full attempted PINO loss is
\[
L=\operatorname{SmoothL1}(\widehat T/10,T^{FV}/10)
+0.1\gamma_t\,\operatorname{mean}(\mathcal D/40000)^2
+0.2\,1_{\rm paired}\operatorname{SmoothL1}
((\Delta\widehat T-\Delta T^{FV})/0.25,0),
\]
where \(\gamma_t=\min(1,t/500)\) over cumulative supervised/paired steps, and
\[
\mathcal D=c(T_{n+1}-T_n)/0.1
-\frac{\kappa}{2}(\nabla_h^2T_{n+1}+\nabla_h^2T_n)
-\bar q_n+h[(T_{n+1}+T_n)/2-T_a].
\]
Losses act on balance-projected fields. At retained step 200, the PDE coefficient reaches **0.04**, and the paired term has never been applied. Evidence: training.py 163–171 and operator.py 33–51, 99–113.

## 4. PI gains and missing tuning evidence

\[
v_j=0.03e_j+0.002I_j,\quad
u_j^{PI}=\operatorname{clip}_{[0,1]}
(\operatorname{clip}_{[u_j^{prev}-0.04,u_j^{prev}+0.04]}v_j),
\]
\[
I_j\leftarrow I_j+0.1e_j\,1[(v_j-u_j^{PI})e_j\le0].
\]

Integral state starts at zero and resets each episode. The engine takes four zone means from the observed field, scores each candidate by negative mean squared distance to the PI proposal, and applies the best candidate admitted by the common PINO gate. Thus this is candidate-projected, filtered PI, not unrestricted simultaneous four-heater PI.

Sources: `snapshot/src/ojpoc/physics.py:114`, `snapshot/src/ojpoc/l2/engine.py:35`. `snapshot/POC_SPEC.md:1394` says to fix PI gains on validation. **Missing:** a gain grid/search, tuning objective, and selection log establishing how 0.03/0.002 were obtained. Inspected source, configurations, stage documentation, and relevant L1/L2 records did not resolve this.

## 5. Regret inclusion, counts, and argmax ties

### Implemented rule

FV eligibility is library validity AND maximum FV temperature≤120. Offline training-validation choices instead use library validity AND predicted PINO maximum≤120, without margin. Final held-out choices use the calibrated PINO margin. A regret exists whenever an FV-eligible alternative exists and the scorer returns a choice.

Unusable input or an empty predicted mask produces no regret; it is unexpected fallback only when an FV-eligible candidate exists. If the returned candidate itself is FV-infeasible, its cost difference is still included and its infeasible counter also increments. That hypothetical difference is not clipped and can be negative. Training selection_score is mean regret (or 1 if none) plus (infeasible+unexpected_fallback)/all_rows.

Sources: `snapshot/src/ojpoc/l2/training.py:41`, `snapshot/src/ojpoc/l2/evaluation.py:40`, `snapshot/src/ojpoc/l2/cache.py:50`.

| Actual records | Raw anchors | Defined regrets | FV-infeasible choices | Unexpected fallback |
|---|---:|---:|---:|---:|
| A, selected 250-step | 96 | 94 | 0 | 0 |
| A, selected 1,000-step | 96 | 94 | 0 | 0 |
| B, each Shared-NoPhysics candidate (3) | 96 | 94 | 0 | 0 |
| B, each Shared-Physics candidate (3) | 96 | 94 | 0 | 0 |
| B, each Small-Physics candidate (3) | 96 | 94 | 0 | 0 |
| B, each Shared-Sensor candidate (3) | 96 | 95 | 0 | 0 |
| B, each Shared-Thermal candidate (3) | 96 | 96 | 0 | 0 |
| D, each final learner's validation | 192 | 188 | 0 | 0 |
| D, each of five methods on ID test | 384 | 376 | 0 | 0 |

D ID records show eight expected fallbacks per method. Every OOD subgroup also reports zero FV-infeasible choices and zero unexpected fallbacks for all five methods. Offline zero failure does not establish zero dynamic-limit violations.

Original records: A uses S/runs/L2_diagnostics/diagnosis_20260926/{full_relative_s3030,full_relative_long_s3030}/08_shared_physics/training.json, selected blocks. B uses each of the five arms' rollout/selection.json under S/runs/L2/l2_auto_20260927T033731Z_17e1cf_pilot; all 15 candidates have eligible=true and zero failure counts. D uses each learner's training.json lines 1–27 and R/17_evaluation/{pi,fv_mpc,pino_mpc,shared_physics,small_physics}.summary.json. These locators refer to the original snapshot; the snapshot itself is not included in S1.

Canonical order is hold, heater 0 up/down, heater 1 up/down, heater 2 up/down, heater 3 up/down. Clipped duplicates are removed in this order. NumPy masked argmax takes the first maximum; no epsilon or random tie rule is applied. Training permutation does not change evaluation's canonical order. Sources: `snapshot/src/ojpoc/physics.py:9`, `snapshot/src/ojpoc/l2/training.py:31`, `snapshot/src/ojpoc/l2/engine.py:80`.

The 1e-6 teacher-cost gap is used for significant ranking pairs, soft-label temperature estimation, and expert-equivalence accuracy; it is not an action argmax tolerance. The revision recomputes eligible top-two corrected-score gaps from saved D requests (Section 8). Precision-perturbation stability remains untested; the gap thresholds are descriptive, not estimates of numerical error.


## 6. Injected delays and representative hold response

The two added timing terms in the asynchronous application-time equation in the main manuscript, injected compute and application delay, are both 0 ms for all 240 original nominal-case metric records and all 1,128 returned replay/fallback cases. They are distinct from the nonzero measured/replayed latency. The original simulator separately supports OOD delay-injection conditions; the zero-value statement does not apply to those separate conditions. The compact metric fields retain the names `injected_compute_ms` and `injected_apply_ms`.

The Figure 5 hold curve comes from the sealed original-interleaved-replay, FV-MPC hold case for episode `l2_240070`, sampled at 10 Hz with a 10-ms plant cap. This is the same episode previously selected as the lower median by zero-delay FV cost; selection was not repeated after inspecting hold performance. Its 601 samples are in `data/minor_revision/hold_temperature.csv`. The author's raw-input audit verified the case seal and its 592 request records, including eight expirations, against the original zero-fallback timing schedule. `method_evidence.json` records hashes and scope. The compact package can check the saved curve and corresponding metric values, but does not contain the raw trace or quadrature needed to repeat that complete audit.

The implemented command check limits the command increment at application. The archived metric field `delayed_slew_rejections` is retained for traceability; it does not certify a continuous-time slew-rate bound.

## 7. Nominal slots and application-event normalization

Equation (3) divides the sum of squared command increments by the fixed N=600 nominal decision slots in a 60-s episode. The sum indexes actual command applications, including fallback-induced changes. Skipped requests do not reduce the denominator. Unchanged command vectors contribute zero, including records skipped by the simulator's `no_change` optimization. This definition was preserved; no costs were renormalized.

For the representative FV episode under original replay, both zero and hold have 592 requests and 592 application/deadline records, including no-change records. The numbers of effective `apply` calls differ (314 and 213) because unchanged command/fallback states are omitted internally. Neither count is the normalization denominator. Reconstructing the full saved command increments gives J_increment=0.00001872 for zero and 0.000002666666666666667 for hold, matching the stored metrics. The portable script checks these sums before generating Figure S2.


## 8. Eligible score margins

`data/revision_20261013/neartie/saved_score_vectors.jsonl.gz` contains 57,600 request records from 96 residual-system cases: two frozen systems, two original timing modes, and 24 nominal episodes. Each group contains 14,376 scored requests and 24 initial fallback requests without scores. No scored request has fewer than two eligible candidates. Recomputed argmax choices match all 57,504 saved scored choices. There are no exact top-score ties.

The corrected scores are the dimensionless values before probability-temperature scaling. The margin is the largest minus the second-largest score among candidates admitted by both library validity and the calibrated gate. Fractions use <= thresholds 0, 1e-4, 1e-3, .01, .1, and 1. The descriptive thresholds are unrelated to numerical-error estimates. Requests are dependent; no request-level confidence intervals or significance tests are calculated. The portable script writes grouped and episode-level summaries from the bundled score vectors.

In the score equation, PINO costs are centered by their FP64 arithmetic mean over all nine candidate slots, including duplicate or subsequently masked slots. Masking restricts the final argmax. Subtracting the same scalar does not change the ordering.

## 9. Command restoration and deadline placement

The compact mechanism inputs include 336 FV application traces (seven replay settings, two policies, 24 episodes), plus the spatial-mean curves needed for response measures. At expiration, the historical four-component command is recorded. Restoration means every component reaches or exceeds its pre-expiration value: u_j(t) >= u_pre,j - 1e-8 for all four heaters. It does not require equality to the historical vector. Follow-up ends at the next expiration or the episode horizon. Unobserved restorations are right-censored; medians among observed events are not estimates of overall recovery time. A historical command may cease to be desirable as temperature changes.

An ordinary application changes at most one coordinate by .04. A zero reset therefore requires at least sum_j ceil(u_pre[j]/.04) increasing coordinate moves to restore the vector, allowing 1e-8 grid tolerance. Multiplying this count by .1 s is a nominal-slot equivalent, not a bound on wall time. Subsequent command choices, application intervals, and skipped ticks matter. The compact analysis verifies all 196,752 ordinary transitions. It reports 715 nonzero zero-policy resets, 324 observed restorations, and 391 censored events (377 at the next expiration, 14 at the horizon). Another 413 expirations occur at an already zero command.

Saved schedule equality is verified within each setting across episodes and both policies. Figure S5 juxtaposes seven fixed schedules and their conditional mean-cost effects. It does not estimate a random timing distribution or identify a separate causal effect of deadline position. The input-extraction and source-seal checks are author audit receipts; the compact commands and curves permit the derived metrics to be recomputed.

## 10. Thermal-response measures

All measures use 10-Hz saved spatial-mean temperature y(t), not complete temperature fields. Late MAE is the trapezoidal mean absolute error over 50–60 s. Overshoot is max(0,max_t[y(t)-setpoint]). Settling is the first sampled time after which all remaining samples through 60 s are within +/-1 K; otherwise the episode is censored at 60 s. Observed-only settling means exclude censored cases and must be reported with the settled count. These are finite-horizon diagnostics, not asymptotic steady-state or intersample peak guarantees.

The early-error fraction is computed for each episode as the trapezoidal integral of squared spatial-mean error over 0–10 s divided by its 0–60-s integral, then averaged equally across 24 episodes. The initial mean error is 26.142 K. Across the eleven original conditions (five methods in two modes plus FV hold), the mean episode-wise early-error fraction is 80.46–86.99%. It is not a decomposition of the full-field cost.

Definitions are post hoc, developed after the original cost outcomes were known. No external preregistration is claimed. `DEFINITIONS.md`, all episode/event values, censoring indicators, and `REPORT.md` are supplied under `data/revision_20261013/mechanism/`.


## 11. Additional Small initialization repetitions

Seeds 6061–6063 vary Small initialization while retaining config seed 6060, sampler seed 6271, training data and all 450 batch hashes, FV supervision, observer, PINO, archived calibration and the fixed Shared checkpoint. All 450 steps (50 warmup+400 adapt), batch size 16, completed. Validation-only checkpoint choice uses the original candidate-cost/failure score; alpha 0/.5/1 is chosen by safety, worst reference cost ratio, mean ratio, then smaller alpha across 16 validation episodes at zero/38-ms synthetic delay. Each seed has 96 selection rollouts. Checkpoints and alpha were fixed before its 48 test runs. The original 48 Small conditions reproduced first. The protocol was fixed before this additional execution, after original findings; it was not externally preregistered.

The new models reuse the original Small timing sequence under zero fallback at a 10-ms plant cap. No new latency benchmark, recalibration, Shared retraining, hot/OOD test, or new-checkpoint step refinement was performed. All 24 nominal episodes are reused. Bootstrap uses 20,000 matched episode resamples, seed 20261013. Intervals are conditional per checkpoint pair and timing sequence, not training-seed-population intervals. Relative mean cost in the new table is 100*(Small/Shared-1), with Shared as denominator; previous residual tables use the opposite contrast.

All three new models select warmup step 50; alpha 1/.5/1. Quality requires lower cached validation mean FV candidate-cost difference than the initial reference with zero infeasible selections/unexpected fallbacks. Seed 6062 has 6.3397561e-7 versus 6.2609296e-7 and fails. Its closed-loop-selected alpha=.5 differs from the alpha=1 checkpoint-validation setting. All technically complete seeds are retained. The failed flag is neither excluded nor equated with a new test thermal violation.

`author_raw_audit.json` records reconstruction of 192 test costs from full quadrature, final fields and applied commands (maximum error 2.78e-16); source seals and original input hashes were also checked. This compact distribution permits saved-metric, training-history and alpha-selection recomputation, not those omitted raw audits. The recorded inference_failures counter is not incremented by the scheduler; it is not an independent fault-tolerance measure.


## Independent latency Monte Carlo (19 October update)

`data/monte_carlo/episode_metrics.csv` contains **9,456 cases: 240 baseline reproductions and 9,216 new runs**, comprising 64 repetitions of six conditions on the same 24 nominal episodes. Five original frozen methods use zero fallback; FV-MPC additionally uses hold. The original Small checkpoint (seed 6060) is used. The three additional Small seeds are not crossed with Monte Carlo timing.

```bash
python -B scripts/reproduce_monte_carlo.py
python -B scripts/make_monte_carlo_figure.py
```

The combined driver runs both. The first recomputes 384 group means, 18 distributions, all 39 rank-event probabilities and Wilson intervals, 64 paired FV effects, completion/thermal/deadline counts, and the sum of saved cost components. It checks every summary against the archived outputs. Figure 3 displays all 64 FV repetition means and their paired policy effect. Tables 5–6 use these outputs. There are **24 unique episodes and 64 independent timing repetitions conditional on the empirical pools**, not 9,216 independent physical scenarios.

Delays are sampled IID with replacement from method/path/token-matched pools made from three archived H200 repeats. The fixed 64-repetition design, seeded deterministic keys, and potential draw plans are in `design.json`. FV zero/hold share potential plans. The portable evidence in `data/monte_carlo_schedule_audit/` independently verifies all 1,536 realized pairs, including every consumed latency draw. Serial timing correlation, new hardware measurements, new training, and new plant-step refinements are absent. The empirical 2.5th–97.5th percentiles describe repetition spread; they are not confidence intervals for a mean. Wilson intervals describe conditional rank-event probabilities, not uncertainty over hardware, trained models or episode populations.

FV mean cost increases by 26.012% under zero and 3.722% under hold. The corresponding percentile ranges are 10.185–49.400% and 2.842–4.712%. FV is strictly highest-cost in 61/64 repetitions under zero and rank 1 in 64/64 under hold, comparing against the other four zero-fallback methods. Hold has lower aggregate cost in all repetitions, but episode-level pairs include 1,034 wins, 114 ties and 388 losses (tolerance 1e-10). The mean of the 64 within-repetition hold-versus-zero changes is -17.258%; this differs from the ratio of pooled means.

All 9,216 new runs complete without recorded thermal-limit violations. Missed deadlines total 79 (PI), 22 (PINO), 9,091 for each FV policy, and zero for Shared/Small. The initialized `inference_failures` field is not incremented by this execution harness; its zeros do not measure inference reliability. Raw `overshoot_k` is a spatial-peak metric on 10-ms/event endpoints and must not be conflated with the manuscript's 10-Hz spatial-mean overshoot diagnostic.

`independent_verification.json`, `archive_integrity.json` and `raw_metric_hashes.json` document the author's full-archive audit: all 9,456 raw costs reconstructed (maximum discrepancy 6.67e-16), archive CRC/seals verified, all 1,536 FV schedules paired, and deterministic sampling checked on 384 selected cases. Quadrature and full fields remain outside this compact package. The new FV schedule-audit subset separately supplies all actual FV schedule/draw records, the complete 36,000-row FV pool, and protocol-pinned sampler/scheduler source; those checks are executable here. `outputs/monte_carlo/checks.json` is a fresh saved-metric check with this narrower scope.

Quantiles use NumPy linear interpolation between adjacent ordered repetition values (the default `quantile` method); standard deviations use `ddof=1`. Rank events and episode-level FV win/tie/loss counts use the protocol absolute cost tolerance of 1e-10.


## 12. Evaluation constraint and portable comparisons

The nine-slot command library is a common discrete search space imposed by this evaluation implementation. Its one-coordinate, 0.04 update rule is not a measured physical actuator slew-rate limit. Asynchronous applications can occur at different intervals, and deadline-induced zero commands bypass the ordinary update rule. The reset-recovery effect is conditional on this library.

The thermal plant has linear diffusion, heat-loss and first-order actuator equations for fixed episode parameters. Clipping, candidate selection, the gate and deadline switching introduce nonlinear or hybrid closed-loop behavior. The manuscript therefore describes the plant as spatially distributed.

Timing refers to complete host-side requests, including CPU work and CUDA synchronization on an H200-equipped host. It is not an isolated GPU neural-network kernel measurement. The CPU quota and software/thread settings are archived; a CPU model identifier was not recovered from the existing records.

Frozen-input SHA-256 checks remain byte-exact. Regenerated JSON summaries are compared as parsed objects so CRLF/LF line endings do not cause a false mismatch; regenerated CSV checks remain byte-exact. The regression check accepts equivalent JSON line endings and rejects changed JSON or CSV values.


## Executable FV Monte Carlo schedule evidence (minor revision)

Run `python -B scripts/reproduce_monte_carlo_schedule.py`; the main driver includes this command. No external dependencies are required for this check. The subset contains both policy arms' actual saved schedules and consumed draw indices for all 1,536 pairs (3,072 cases), 3,667,828 events, and 1,824,628 draws. Each policy has 912,314 requests, 9,091 expirations and 9,286 busy skips. The verifier checks integer-nanosecond event equality, strict expiration, all 600 nominal ticks per case, matched pool membership and every deterministic SHA-256 draw. No tolerance is applied to paired event records.

The complete original 36,000-row FV pool, pinned sampler/scheduler source, original case bindings, schema and input hashes are included. `data/monte_carlo_schedule_audit/README.md` distinguishes copied fields from derived checks. Original simulated readiness times reflect the sampled measured latency; they are not new hardware measurements. Full original-ZIP field comparisons and targeted corruption-test receipts are supplied separately from the fresh offline results. This public source subset includes all 20 protocol-pinned harness files, including the original MIT and CC BY 4.0 metadata. The guide also documents the separate anonymous review subset, which withholds those two attribution files.

This addition makes the FV schedule and draw claim independently checkable without the 5.1-GB archive. It does not rerun inference, dynamics, command values or cost quadrature. Existing 804 scientific files remain byte-identical. The CRLF/LF-sensitive derived JSON comparison now compares parsed content; CSV comparisons and frozen input hashes remain byte-exact. A regression check accepts equivalent line endings and rejects a changed numerical value.
