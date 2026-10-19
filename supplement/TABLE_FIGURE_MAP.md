# Manuscript-to-evidence map

Paths are relative to Supplement_S1; generated results are under `outputs/`. Run `python -B scripts/reproduce_review.py`. This saved-result analysis calls no model, GPU, or simulator. Numbering below matches the balanced manuscript revision of 19 October 2026.

## Main tables

| Table | Evidence | Script / output |
|---|---|---|
| 1 Plant | `data/protocol/D_resolved_config.json`, `METHODS.md` | Documented frozen settings |
| 2 Latency | `data/primary/latency_quantiles.csv`, `latency_audit.json` | `scripts/reproduce.py` → `table5_latency_all_scopes.csv`; stored quantiles, not fresh raw-journal aggregation |
| 3 Original mean costs | `data/primary/episode_metrics.csv` | `scripts/reproduce.py` → `table6_primary_costs.csv` |
| 4 Paired residual comparison | Same original episode metrics | `scripts/reproduce.py` → `table7_paired_comparisons.csv` |
| 5 Monte Carlo cost distributions | `data/monte_carlo/episode_metrics.csv` | `scripts/reproduce_monte_carlo.py` → `monte_carlo/cost_effect_distribution.csv` |
| 6 Monte Carlo ranking events | Same saved metrics | Same script → `monte_carlo/ranking_probabilities.csv` |
| 7 Replay and policy | `data/replay/episode_metrics.csv`, `sequence_summary.csv`, `method_rankings.json` | `scripts/reproduce_replay.py` → `replay_paired_results.json` (57 contrasts) |
| 8 Command restoration | `data/revision_20261013/mechanism/fv_application_records.csv.gz`, case metadata and curves | Sibling `reproduce_compact.py` → `mechanism/setting_summary.csv`, event/episode CSVs |
| 9 FV thermal responses | `data/revision_20261013/mechanism/truth_curves.csv.gz` and metadata | Same script → `mechanism/physical_summary.csv`, `physical_episode_metrics.csv` |
| 10 Step sensitivity | `data/numerical/extended_metrics.csv`, `contrasts.csv` | `scripts/reproduce.py` → `table8_numerical_sensitivity.csv` |
| 11 Small initialization | `data/small_seed/` | `scripts/reproduce_small_seed.py` → `small_seed/paired_contrasts.csv`, training and selection checks |

Historical `table5`–`table8` output names correspond to current Tables 2, 3, 4, and 10. Table 9 reports original conditions only; all eleven original response conditions are in guide Table S1. Eligible score margins formerly in the main text appear in guide Table S2. Analysis chronology appears in guide Table S3 and PROTOCOL_DETAILS.md. Requests and restoration events are summarized descriptively.

## Figures

| Figure | Basename | Input / generator |
|---|---|---|
| Main 1 | `evaluation_flow` | Documented execution rules; `scripts/make_evaluation_flow.py` via `scripts/make_review_figures.py` |
| Main 2 | `paired_test_results` | `data/revision/cost_components.csv`; same script; panel (c) uses 10^-5 units as Table 4 |
| Main 3 | `monte_carlo_policy_effect` | `data/monte_carlo/replicate_summary.csv`; `scripts/make_monte_carlo_figure.py` |
| Main 4 | `replay_fallback_sensitivity` | `data/replay/replay_results.json`; `scripts/make_replay_figure.py` |
| Main 5 | `temperature_response` | Representative temperature and hold curves; `scripts/make_review_figures.py` |
| S1 | `first_command_divergence` | `data/selected_C/applied_commands.csv`; `scripts/plot_results.py` via `reproduce.py` |
| S2 | `deadline_command_response` | `data/minor_revision_r2/deadline_commands.csv`; `scripts/make_deadline_figure.py` |
| S3 | `residual_architecture` | Frozen model dimensions; `scripts/make_review_figures.py` |
| S4 | `numerical_case_C` | `data/numerical/independent_audit.json`, `initial_C_audit.json`; `scripts/plot_results.py` |
| S5 | `deadline_schedule_cost` | Compact mechanism summaries and schedule audit; `make_mechanism_figure.py` in that directory |
| S6 | `small_seed_cost_comparison` | `data/small_seed/`; `scripts/reproduce_small_seed.py`, `make_small_seed_figure.py` |

All eleven figures are regenerated as PNG/PDF under `outputs/figures/`. Main Figure 5 and Figure S2 retain the same lower-median episode by zero-delay FV cost. Captions and definitions are in SUPPLEMENTARY_FIGURES.md and the guide.

## Supplementary tables and moved details

- Table S1: all eleven original thermal-response conditions, from `data/revision_20261013/mechanism/`.
- Table S2: corrected-score margins, from `data/revision_20261013/neartie/saved_score_vectors.jsonl.gz`; sibling `reproduce_neartie.py` regenerates descriptive threshold fractions.
- Table S3: study chronology, from the original protocols; see PROTOCOL_DETAILS.md.
- `methods/evaluation_protocol_detail.tex`: relocated eligibility/candidate-cost equation, timing environment, detailed replay/statistical rules, baseline checks and analysis-status table. Source labels are historical identifiers.
- `methods/development_detail.tex`, `development_and_selected_C_results.tex`, `cohorts_and_procedure.tex`, and `selected_C_detail.tex`: development and selected-case evidence.
- `METHODS.md` and `methods/residual_architecture_detail.tex`, `residual_training_detail.tex`, `pino_training_detail.tex`: architecture, training, masks, losses and selection.
- `methods/cost_component_table.tex`, `data/revision/physical_contrasts.csv`, and `scripts/reproduce_physical.py`: component differences.
- `scripts/reproduce_method_details.py`: alpha selection, injected delays and hold-curve checks.

Raw source audit receipts and executable compact-input checks have different scopes. Full timing journals, quadrature fields, weights, and training inputs are not distributed here. Figure 1 retains the monochrome vector design and independent generator introduced in the previous revision.

The conceptual related-work comparison is retained in methods/related_work_position.tex; the main Related Work section presents its citations and context in prose. This source fragment is unnumbered supporting material.


## Final submission consistency update

Main-text citations now use one reference at a time in superscript brackets. METHODS.md uses the current Figure 5 temperature response, Equation (3) cost, and Figure 3 / Tables 5–6 Monte Carlo references. The manuscript distinguishes offline checkpoint selection from closed-loop correction-strength selection and specifies full-field, per-episode tracking RMSE before averaging. Experimental data, analysis scripts, and numerical results are unchanged.
