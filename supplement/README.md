# Supplement S1: saved-result analysis

Companion to *Inference Delay and Deadline Handling in Thermal Controller Evaluation: A Retrospective Simulation Study*. Revision: 2026-10-19.

This public data package contains compact scientific records, implementation details, and scripts to regenerate **four main figures and six supplementary figures**. Version 1.2.0, prepared 19 October 2026, by Minsu Jo, Namhyun Yoo, Hoansuk Choi, and Jinhong Yang. Code/documentation: MIT; data/research figures: CC BY 4.0. See LICENSE, DATA_LICENSE and CITATION.cff. External dependencies are not bundled. Repository: https://github.com/JoMinsu-KU/thermal-delay-evaluation.

## Reproduce

Use Python 3.10+ with NumPy and Matplotlib. From this directory:

```bash
python -m pip install -r requirements.txt
python -B scripts/reproduce_review.py
```

The command verifies SHA-256 hashes, recalculates paired episode contrasts and numerical comparisons, checks 240 cost decompositions and alpha selection, and reproduces all 57 replay/fallback contrasts from 1,128 cases. It also checks all three additional Small initializations, 192 test outcomes, 288 validation selection outcomes, and per-seed paired intervals. It recomputes 57,504 eligible score margins, command restoration from 336 FV traces, and thermal metrics from 552 mean-temperature curves. It checks these new results against the bundled summaries. Outputs go to `outputs/`. No model, training, GPU, or plant simulation is invoked.

## Navigation

The included [Supplement_S1_guide.pdf](Supplement_S1_guide.pdf) presents the supplementary figures and key moved results. [TABLE_FIGURE_MAP.md](TABLE_FIGURE_MAP.md) maps all twelve main tables, four main figures, and six supplementary figures. Historical table numbers in output filenames are preserved for provenance.

| Evidence | Input directory | Reproduction |
|---|---|---|
| Original 24-episode comparison | `data/primary/` | `scripts/reproduce.py` |
| Seven fixed timing settings and FV zero/hold | `data/replay/` | `scripts/reproduce_replay.py` |
| Score margins (Table 7) | `data/revision_20261013/neartie/` | `reproduce_neartie.py` in that directory |
| Restoration and thermal response (Tables 9–10) | `data/revision_20261013/mechanism/` | `reproduce_compact.py` in that directory |
| Numerical refinement (Table 11) | `data/numerical/` | `scripts/reproduce.py` |

The main driver runs each required script with explicit output directories. Individual analysis scripts may default to a sibling `reproduced/` directory; generated outputs are not frozen inputs.

## Evidence and scope

- **Original archive:** `data/primary/`, `data/development/`, and `data/protocol/` document the original frozen systems and execution conditions. D contains 24 nominal test episodes; eight initially hot episodes and other OOD conditions are separate. Nominal describes initial conditions; normal path describes inference with usable inputs and an admitted candidate.
- **Numerical validation:** `data/numerical/` contains nine earlier C closed-loop runs, twelve fixed-command C runs, and 720 D runs at 10/5/2.5 ms. Repeated conditions do not add independent episodes.
- **Replay/fallback batch:** `data/replay/` contains 240 baseline reproductions, 720 alternative replay runs, and 168 FV hold runs. All use the same 24 nominal episodes at a 10-ms plant cap. R1B1–R3B2 reuse three recorded timing repeats; they are not new independent measurements.
- **Physical and method checks:** `data/revision/`, `data/minor_revision/`, and `data/minor_revision_r2/` retain cost decompositions, the unchanged representative episode, alpha-validation outcomes, injected-delay checks, and the selected deadline-command comparison.
- **Current saved-data analyses:** `data/revision_20261013/` adds scored candidate vectors, compact application logs, 10-Hz mean-temperature curves, exact definitions, derived event/episode summaries, and source audit receipts. These support new analyses of existing experiments; these October 13 inputs contain no new experiments. The subsequently completed Small repetitions are supplied separately under `data/small_seed/`; no independent random-latency Monte Carlo result is included.
- `METHODS.md` specifies implemented models and measures. `methods/` retains LaTeX fragments for A/B/C, architecture/training details, the original cohort/procedure tables, and cost components moved from the main manuscript. The fragments are supporting source, not standalone TeX documents.

## Interpretation limits

Bootstrap intervals condition on the trained systems and fixed replay settings. Requests and expiration events within an episode are dependent. Command restoration refers to a historical input vector and is censored at the next expiration or 60 s. Thermal metrics use sampled spatial means, so they do not bound intersample or local peak temperatures.

The supplement supports saved-result reanalysis. It omits model weights, complete training inputs, full raw timing journals, and full quadrature/temperature-field archives. It cannot independently repeat training, inference, new closed-loop conditions, or the authors' complete source-seal audit. Audit receipts describe checks on the original archive; compact-input checks are executable here.

`SHA256SUMS` covers every distributed input/script/document except itself. Generated `outputs/`, `reproduced/`, and Python caches are excluded. Hashes establish file integrity, not scientific correctness.

## Small initialization repetitions (19 October update)

`data/small_seed/` contains 192 test outcomes (48 original-Small reproductions and 144 new tests), complete recorded training histories, 288 validation-selection outcomes, a fixed Shared reference, and author audit receipts. Run `python -B scripts/reproduce_small_seed.py` for the table/statistics and `python -B scripts/make_small_seed_figure.py` for Figure S6; the main driver runs both.

All three new seeds have lower mean replay cost than the fixed Shared checkpoint, but only one exhibits the original ordering reversal. Seed 6062 fails the original validation improvement criterion and is retained. There are 24 unique test episodes, not 72 independent observations per mode. Timing and calibration are reused; Shared is not retrained. Raw-field/quadrature and source-hash audits are receipts, not independently reproducible from this compact subset. Selected weights and complete training inputs are omitted.
