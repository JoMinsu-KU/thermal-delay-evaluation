# Supplement S1: saved-result analysis

Companion to *Inference Delay and Deadline Handling in Thermal Controller Evaluation: A Retrospective Simulation Study*.

This saved-result package contains compact scientific records, implementation details, and code to reanalyze the saved results and regenerate the six main figures and two supplementary command figures. Public release v1.1.0 by Minsu Jo, Namhyun Yoo, Hoansuk Choi, and Jinhong Yang. Code and documentation are MIT; data and research figures are CC BY 4.0. See LICENSE, DATA_LICENSE, and CITATION.cff. External dependencies retain their own terms. Repository: https://github.com/JoMinsu-KU/thermal-delay-evaluation.

## Reproduce

Use Python 3.10 or later with NumPy and Matplotlib. If needed, install the two dependencies with `python -m pip install -r requirements.txt`, then run:

```bash
python -B scripts/reproduce_review.py
```

The command verifies SHA-256 hashes, recomputes paired episode summaries and the existing numerical contrasts, checks the 240 cost-component records, recomputes all 57 replay/fallback contrasts from 1,128 cases, checks both injected delay values and all six alpha-selection candidates, and writes figures and tables into `outputs/`. The figures are `evaluation_flow`, `residual_architecture`, `paired_test_results`, `temperature_response`, `first_command_divergence`, `numerical_case_C`, `replay_fallback_sensitivity`, and `deadline_command_response`. Figures S1 (`first_command_divergence`) and S2 (`deadline_command_response`) are supplementary; the other six appear in the revised main manuscript. Captions are in SUPPLEMENTARY_FIGURES.md.

## Quick route to Table 9 and Figure 4

| Manuscript item | Input | Run | Output |
|---|---|---|---|
| Table 9 | `data/replay/episode_metrics.csv` | `python -B scripts/reproduce_replay.py` | `outputs/replay_paired_results.json` (57 paired contrasts); compare `data/replay/sequence_summary.csv` and `method_rankings.json` |
| Figure 4 | `data/replay/replay_results.json` (checked by the previous command) | `python -B scripts/make_replay_figure.py` | `outputs/figures/replay_fallback_sensitivity.pdf` and `.png` |
| Supplementary Figure S2 | `data/minor_revision_r2/deadline_commands.csv` | `python -B scripts/make_deadline_figure.py` | `outputs/figures/deadline_command_response.pdf` and `.png` |

The complete [table/figure map](TABLE_FIGURE_MAP.md) covers every main table and figure. The included [Supplement_S1_guide.pdf](Supplement_S1_guide.pdf) is a readable navigation guide with both supplementary figures. This ZIP provides the executable inputs and scripts; the guide does not replace it.

## Contents and limits

- `data/primary`: saved D outcomes and latency quantile summaries. The primary inferential sample is 24 normal test episodes; hot/OOD results are separately labelled.
- `data/development`: A/B checkpoint and validation records. Reused validation trajectories do not establish a selector's independent test benefit.
- `data/numerical`: summaries of 720 D and 12 C executions in the extended batch and nine earlier C closed-loop runs. Repeated settings do not enlarge the independent episode count.
- `data/selected_C`: the supporting selected-episode command diagnostic.
- `data/revision`: the 240 D baseline cost decompositions, physical contrasts, representative 10-Hz temperature records, and hashes identifying the raw inputs used by the authors. The temperature episode is the lower median (12th of 24) by zero-delay FV cost, with episode ID breaking ties.
- `METHODS.md`: model input construction, candidate scoring, losses, retained checkpoints, discrete PI updates, and evaluation definitions.
- `methods/`: detailed A/B text and PINO training settings moved out of the main paper. These are LaTeX fragments, not standalone documents; cross-references refer to the main manuscript.
- `DATA_DICTIONARY.md`: original field names and units. Entries referring to raw archives describe the full evidence archive, which is not included in this compact supplement.

The bootstrap uses 24 paired episodes conditional on the frozen models and fixed replay sequences. The supplement does not include model weights, complete training inputs, raw timing journals, all quadrature arrays, or full temperature fields. Therefore it cannot independently repeat training/inference or the authors' raw-input audit. Stored component values and temperature inputs permit the reported saved-result analyses and figures to be checked. The completed replay/fallback batch contains 240 original-condition reproductions, 720 alternative-replay runs and 168 FV hold runs. All 1,128 cases use the same 24 normal episodes. `data/replay/` supplies compact records, fixed protocol settings, all paired contrasts and an audit receipt. `methods/selected_C_detail.tex` retains the command diagnostic moved out of the main text. The new replay/hold conditions were evaluated at a 10-ms plant cap only.

`SHA256SUMS` covers every distributed input and script except the checksum file itself. Generated `outputs/` are not part of this fixed input manifest.

## Find the evidence used in the manuscript

Start with [TABLE_FIGURE_MAP.md](TABLE_FIGURE_MAP.md), which maps every main table and figure to its input, script, and output. File names containing old table numbers are retained for provenance; the map gives the current numbering.

Three evidence groups must be distinguished:

1. **Original archive:** `data/primary/`, `data/development/`, and `data/protocol/D_*` describe the original frozen systems, development cohorts, original interleaved replay and zero fallback. The primary comparison uses 24 normal episodes. The hot and other OOD records are separate.
2. **Numerical-validation batches:** `data/numerical/` contains 9 earlier C closed-loop runs, 12 C fixed-command runs, and 720 D runs at 10/5/2.5 ms. These establish the reported sensitivity of the original conditions; they do not refine the newly added replay/hold conditions.
3. **Replay/fallback batch:** `data/replay/` contains 1,128 runs: 240 original-condition reproductions, 720 additional-replay runs, and 168 FV hold runs. Each uses the same 24 normal episodes, with a 10-ms plant cap. R1B1–R3B2 reuse three archived timing repeats; they are not six new independent timing measurements.

`data/minor_revision/` adds saved D alpha-validation outcomes, the 601-sample hold temperature curve for the unchanged representative episode, and a method-audit receipt. `scripts/reproduce_method_details.py` recalculates all six alpha keys, checks injected delays in the compact records, and checks the hold input against the saved case metrics. It does not rerun raw trace or file-seal checks; those are recorded author audits.

The main manuscript's method references resolve to METHODS Section 2 (residual loss and alpha selection), Section 3 (PINO loss/training), and Section 6 (injected delays and hold-curve provenance). Detailed A/B protocols are in `methods/development_detail.tex`; PINO settings are also in `methods/pino_training_detail.tex`. `data/replay/paired_contrasts.csv`, `sequence_summary.csv`, and `method_rankings.json` retain every replay-setting comparison.
