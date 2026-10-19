# Manuscript-to-evidence map

Paths are relative to Supplement_S1; generated results are under `outputs/`. Run `python -B scripts/reproduce_review.py`. This saved-result analysis calls no model, GPU, or simulator.

## Main tables

| Table | Evidence | Script / output |
|---|---|---|
| 1 Related work | Publications cited in the main text | Conceptual comparison; no numerical aggregation |
| 2 Plant | `data/protocol/D_resolved_config.json`, `METHODS.md` | Documented frozen settings |
| 3 Analysis status | Original protocols and retrospective analysis history | `methods/cohorts_and_procedure.tex`; no external preregistration |
| 4 Latency | `data/primary/latency_quantiles.csv`, `latency_audit.json` | `scripts/reproduce.py` → `table5_latency_all_scopes.csv`; stored quantiles, not fresh raw-journal aggregation |
| 5 Original mean costs | `data/primary/episode_metrics.csv` | `scripts/reproduce.py` → `table6_primary_costs.csv` |
| 6 Paired residual comparison | Same original episode metrics | `scripts/reproduce.py` → `table7_paired_comparisons.csv` |
| 7 Eligible score margins | `data/revision_20261013/neartie/saved_score_vectors.jsonl.gz` | Sibling `reproduce_neartie.py` → `neartie/near_tie_summary.csv`, `threshold_fractions.csv`, `episode_threshold_fractions.csv` |
| 8 Replay and policy | `data/replay/episode_metrics.csv`, `sequence_summary.csv`, `method_rankings.json` | `scripts/reproduce_replay.py` → `replay_paired_results.json` (57 contrasts) |
| 9 Command restoration | `data/revision_20261013/mechanism/fv_application_records.csv.gz`, case metadata and curves | Sibling `reproduce_compact.py` → `mechanism/setting_summary.csv`, event/episode CSVs |
| 10 FV thermal responses | `data/revision_20261013/mechanism/truth_curves.csv.gz` and metadata | Same script → `mechanism/physical_summary.csv`, `physical_episode_metrics.csv` |
| 11 Step sensitivity | `data/numerical/extended_metrics.csv`, `contrasts.csv` | `scripts/reproduce.py` → `table8_numerical_sensitivity.csv` |
| 12 Small initialization | `data/small_seed/` | `scripts/reproduce_small_seed.py` → `small_seed/paired_contrasts.csv`, training and selection checks |
| 13 Monte Carlo cost distributions | `data/monte_carlo/episode_metrics.csv` | `scripts/reproduce_monte_carlo.py` → `monte_carlo/cost_effect_distribution.csv` |
| 14 Monte Carlo ranking events | Same saved metrics | Same script → `monte_carlo/ranking_probabilities.csv` |

Historical `table5`–`table8` output names correspond to current Tables 4, 5, 6, and 11. Table 10 reports original conditions only; all eleven original method/condition summaries are retained in the supplement. Tables 7 and 9 use dependent requests/events descriptively, not as independent inferential samples.

## Figures

| Figure | Basename | Input / generator |
|---|---|---|
| Main 1 | `evaluation_flow` | Documented execution rules; `scripts/make_evaluation_flow.py` via `scripts/make_review_figures.py` |
| Main 2 | `paired_test_results` | `data/revision/cost_components.csv`; same script; panel (c) uses 10^-5 units as Table 6 |
| Main 3 | `replay_fallback_sensitivity` | `data/replay/replay_results.json`; `scripts/make_replay_figure.py` |
| Main 4 | `temperature_response` | Representative temperature and hold curves; `scripts/make_review_figures.py` |
| Main 5 | `monte_carlo_policy_effect` | `data/monte_carlo/replicate_summary.csv`; `scripts/make_monte_carlo_figure.py` |
| S1 | `first_command_divergence` | `data/selected_C/applied_commands.csv`; `scripts/plot_results.py` via `reproduce.py` |
| S2 | `deadline_command_response` | `data/minor_revision_r2/deadline_commands.csv`; `scripts/make_deadline_figure.py` |
| S3 | `residual_architecture` | Frozen model dimensions; `scripts/make_review_figures.py` |
| S4 | `numerical_case_C` | `data/numerical/independent_audit.json`, `initial_C_audit.json`; `scripts/plot_results.py` |
| S5 | `deadline_schedule_cost` | Compact mechanism setting summaries and schedule audit; `make_mechanism_figure.py` in that directory |
| S6 | `small_seed_cost_comparison` | `data/small_seed/`; `scripts/reproduce_small_seed.py`, `make_small_seed_figure.py` |

All eleven are regenerated as PNG/PDF under `outputs/figures/`. Main Figure 4 and Figure S2 use the unchanged lower-median episode by zero-delay FV cost. Captions and limitations are in `SUPPLEMENTARY_FIGURES.md` and the guide PDF.

## Moved details

- A/B protocols: `methods/development_detail.tex`; results and selected C numerical checks: `methods/development_and_selected_C_results.tex`.
- Original cohort/procedure tables: `methods/cohorts_and_procedure.tex`.
- Architecture/training: `methods/residual_architecture_detail.tex`, `residual_training_detail.tex`, `pino_training_detail.tex`, and `METHODS.md` Sections 1–3.
- Cost-component contrasts: `methods/cost_component_table.tex`; `data/revision/physical_contrasts.csv`; `scripts/reproduce_physical.py` → `physical_checks.json`.
- Detailed selected C commands: `methods/selected_C_detail.tex`.
- Alpha, injected delays, and hold-curve checks: `scripts/reproduce_method_details.py` → `method_detail_checks.json`.

Raw source audit receipts and portable compact-input checks have different scopes. Full timing journals, quadrature fields, weights, and training inputs are not distributed here.

Figure 1 was redrawn on 19 October 2026 with monochrome rectangular nodes, consistent straight arrows, and an attached orthogonal plant-to-observation feedback path. Its scientific labels and execution order are preserved. PDF and SVG are vector exports; PNG is 300 dpi. Run `python -B scripts/make_evaluation_flow.py --output outputs/figures` to regenerate this figure alone.
