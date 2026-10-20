# Supplement S1: saved-result analysis

Companion to *Inference Delay and Deadline Handling in Thermal Controller Evaluation*. Revision: 2026-10-19.

This public data package contains compact scientific records, implementation details, and scripts to regenerate **five main figures and six supplementary figures**. Version 1.3.4, by Minsu Jo, Namhyun Yoo, Hoansuk Choi and Jinhong Yang. Code/documentation: MIT; data/research figures: CC BY 4.0. See LICENSE, DATA_LICENSE and CITATION.cff. External dependencies are not bundled. Repository: https://github.com/JoMinsu-KU/thermal-delay-evaluation.

## Reproduce

Use Python 3.10+ with NumPy and Matplotlib. From this directory:

```bash
python -m pip install -r requirements.txt
python -B scripts/reproduce_review.py
```

The command verifies SHA-256 hashes, recalculates paired episode contrasts and numerical comparisons, checks 240 cost decompositions and alpha selection, and reproduces all 57 replay/fallback contrasts from 1,128 cases. It also checks all three additional Small initializations, 192 test outcomes, 288 validation selection outcomes, and per-seed paired intervals. It recomputes 57,504 eligible score margins, command restoration from 336 FV traces, and thermal metrics from 552 mean-temperature curves. It checks these new results against the bundled summaries. Outputs go to `outputs/`. No model, training, GPU, or plant simulation is invoked.

## Navigation

The included [Supplement_S1_guide.pdf](Supplement_S1_guide.pdf) presents the supplementary figures and key moved results. [TABLE_FIGURE_MAP.md](TABLE_FIGURE_MAP.md) maps all eleven main tables, five main figures, and six supplementary figures. Historical table numbers in output filenames are preserved for provenance.

| Evidence | Input directory | Reproduction |
|---|---|---|
| Original 24-episode comparison | `data/primary/` | `scripts/reproduce.py` |
| Seven fixed timing settings and FV zero/hold | `data/replay/` | `scripts/reproduce_replay.py` |
| Score margins (Table S2) | `data/revision_20261013/neartie/` | `reproduce_neartie.py` in that directory |
| Restoration and thermal response (Tables 8–9) | `data/revision_20261013/mechanism/` | `reproduce_compact.py` in that directory |
| Numerical refinement (Table 10) | `data/numerical/` | `scripts/reproduce.py` |

The main driver runs each required script with explicit output directories. Individual analysis scripts may default to a sibling `reproduced/` directory; generated outputs are not frozen inputs.

## Evidence and scope

- **Original archive:** `data/primary/`, `data/development/`, and `data/protocol/` document the original frozen systems and execution conditions. D contains 24 nominal test episodes; eight initially hot episodes and other OOD conditions are separate. Nominal describes initial conditions; normal path describes inference with usable inputs and an admitted candidate.
- **Numerical validation:** `data/numerical/` contains nine earlier C closed-loop runs, twelve fixed-command C runs, and 720 D runs at 10/5/2.5 ms. Repeated conditions do not add independent episodes.
- **Replay/fallback batch:** `data/replay/` contains 240 baseline reproductions, 720 alternative replay runs, and 168 FV hold runs. All use the same 24 nominal episodes at a 10-ms plant cap. R1B1–R3B2 reuse three recorded timing repeats; they are not new independent measurements.
- **Physical and method checks:** `data/revision/`, `data/minor_revision/`, and `data/minor_revision_r2/` retain cost decompositions, the unchanged representative episode, alpha-validation outcomes, injected-delay checks, and the selected deadline-command comparison.
- **Current saved-data analyses:** `data/revision_20261013/` adds scored candidate vectors, compact application logs, 10-Hz mean-temperature curves, exact definitions, derived event/episode summaries, and source audit receipts. These support new analyses of existing experiments; these October 13 inputs contain no new experiments. The subsequently completed Small repetitions are supplied separately under `data/small_seed/`; 64 independent latency Monte Carlo repetitions are supplied under `data/monte_carlo/`.
- `METHODS.md` specifies implemented models and measures. `methods/` retains LaTeX fragments for A/B/C, architecture/training details, the original cohort/procedure tables, and cost components moved from the main manuscript. The fragments are supporting source, not standalone TeX documents.

## Interpretation limits

Bootstrap intervals condition on the trained systems and fixed replay settings. Requests and expiration events within an episode are dependent. Command restoration refers to a historical input vector and is censored at the next expiration or 60 s. Thermal metrics use sampled spatial means, so they do not bound intersample or local peak temperatures.

The supplement supports saved-result reanalysis. It omits model weights, complete training inputs, other methods' raw timing journals and full quadrature/temperature-field archives. It cannot independently repeat training, inference, new closed-loop conditions, or the authors' complete source-seal audit. Audit receipts describe checks on the original archive; compact-input checks are executable here.

`SHA256SUMS` covers every distributed input/script/document except itself. Generated `outputs/`, `reproduced/`, and Python caches are excluded. Hashes establish file integrity, not scientific correctness.

## Small initialization repetitions (19 October update)

`data/small_seed/` contains 192 test outcomes (48 original-Small reproductions and 144 new tests), complete recorded training histories, 288 validation-selection outcomes, a fixed Shared reference, and author audit receipts. Run `python -B scripts/reproduce_small_seed.py` for the table/statistics and `python -B scripts/make_small_seed_figure.py` for Figure S6; the main driver runs both.

All three new seeds have lower mean replay cost than the fixed Shared checkpoint, but only one exhibits the original ordering reversal. Seed 6062 fails the original validation improvement criterion and is retained. There are 24 unique test episodes, not 72 independent observations per mode. Timing and calibration are reused; Shared is not retrained. Raw-field/quadrature and source-hash audits are receipts, not independently reproducible from this compact subset. Selected weights and complete training inputs are omitted.

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

Figure 1 was redrawn on 19 October 2026 with monochrome rectangular nodes, consistent straight arrows, and an attached orthogonal plant-to-observation feedback path. Its scientific labels and execution order are preserved. PDF and SVG are vector exports; PNG is 300 dpi. Run `python -B scripts/make_evaluation_flow.py --output outputs/figures` to regenerate this figure alone.


## Balanced manuscript revision

The main paper now presents the matched Monte Carlo policy effect before fixed-replay recovery and sensitivity results. All prior scientific data and numerical outcomes are preserved. Figure 1 now explicitly labels candidate scoring and the PINO feasibility gate. Study chronology and detailed execution checks are collected in PROTOCOL_DETAILS.md; score margins and chronology are guide Tables S2 and S3. TABLE_FIGURE_MAP.md gives the current numbering.


## Final submission consistency update

Main-text citations now use one reference at a time in superscript brackets. METHODS.md uses the current Figure 5 temperature response, Equation (3) cost, and Figure 3 / Tables 5–6 Monte Carlo references. The manuscript distinguishes offline checkpoint selection from closed-loop correction-strength selection and specifies full-field, per-episode tracking RMSE before averaging. Experimental data, analysis scripts, and numerical results are unchanged.


## Executable FV Monte Carlo schedule evidence (minor revision)

Run `python -B scripts/reproduce_monte_carlo_schedule.py`; the main driver includes this command. No external dependencies are required for this check. The subset contains both policy arms' actual saved schedules and consumed draw indices for all 1,536 pairs (3,072 cases), 3,667,828 events, and 1,824,628 draws. Each policy has 912,314 requests, 9,091 expirations and 9,286 busy skips. The verifier checks integer-nanosecond event equality, strict expiration, all 600 nominal ticks per case, matched pool membership and every deterministic SHA-256 draw. No tolerance is applied to paired event records.

The complete original 36,000-row FV pool, pinned sampler/scheduler source, original case bindings, schema and input hashes are included. `data/monte_carlo_schedule_audit/README.md` distinguishes copied fields from derived checks. Original simulated readiness times reflect the sampled measured latency; they are not new hardware measurements. Full original-ZIP field comparisons and targeted corruption-test receipts are supplied separately from the fresh offline results. This public source subset includes all 20 protocol-pinned harness files, including the original MIT and CC BY 4.0 metadata. The guide also documents the separate anonymous review subset, which withholds those two attribution files.

This addition makes the FV schedule and draw claim independently checkable without the 5.1-GB archive. It does not rerun inference, dynamics, command values or cost quadrature. Existing 804 scientific files remain byte-identical. The CRLF/LF-sensitive derived JSON comparison now compares parsed content; CSV comparisons and frozen input hashes remain byte-exact. A regression check accepts equivalent line endings and rejects a changed numerical value.
