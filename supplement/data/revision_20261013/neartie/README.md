# Eligible top-two score gaps

This is a post hoc reanalysis of saved decisions, not a new closed-loop experiment or a training replication. It covers the 24 nominal held-out episodes, the frozen Shared-Residual and Small-Residual systems, zero delay and the original interleaved replay, and the 10-ms plant cap.

## Reproduce from compact saved inputs

Keep `reproduce_neartie.py` and `saved_score_vectors.jsonl.gz` in the same directory, install NumPy, and run:

```text
python -B reproduce_neartie.py
```

The script writes `reproduced/near_tie_summary.csv`. An optional first argument changes the output directory. The compressed input contains all 57,600 request records: method, execution mode, episode ID, request ID, the nine saved scores, validity and eligibility masks, selected index, and fallback reason. It does not contain model weights, observations, author identities, or machine paths.

For each scored request, the gap is the largest minus second-largest score among candidates with `predicted_safe_mask=True`. This mask already includes duplicate removal (`valid_mask`) and the calibrated predicted-temperature gate. The script verifies that eligibility is a subset of validity and that the stored choice equals the masked argmax. It checks all nine scores for finite values. The initial missing-input request is unscored and is excluded, rather than counted as a tie.

Scores are the dimensionless corrected scores before division by the probability temperature. These gaps are **not** 60-s closed-loop costs, offline FV candidate-cost differences, or calibrated probabilities. Multiplying them by the training scale would only produce a corrected score's cost-unit representation; it would not turn them into measured FV costs or a bound on feedback performance.

## Results

Each of the four groups contains 14,400 requests, 24 unscored initial missing-input fallbacks, and 14,376 scored choices. All scored requests have 5–9 eligible candidates. There are no exact ties between the two largest eligible scores and no gate intervention on the selected argmax in these groups.

| System | Mode | 1st percentile | Median | 95th percentile | Gap <= 0.001 | Gap <= 0.01 |
|---|---|---:|---:|---:|---:|---:|
| Shared | Zero delay | 0.000620 | 0.07355 | 5.09264 | 1.70% | 15.04% |
| Shared | Replay | 0.000523 | 0.09525 | 5.16357 | 1.84% | 15.32% |
| Small | Zero delay | 0.000553 | 0.07657 | 5.11429 | 1.93% | 15.45% |
| Small | Replay | 0.000491 | 0.07755 | 5.13108 | 1.89% | 15.51% |

The descriptive threshold grid `[0, 0.0001, 0.001, 0.01, 0.1, 1]` was written into the analysis script before inspecting the aggregate distribution. It was not preregistered before the original experiments. It has no experimentally validated link to numerical perturbation magnitude or practical control relevance. Full-grid results and episode-level ranges are retained in the accompanying CSVs. For example, the fraction at 0.01 varies from 4.51% to 47.08% across Shared replay episodes, despite a pooled fraction of 15.32%. Requests within an episode are not independent replicates; no request-level confidence intervals are claimed.

These results document the occurrence of close score comparisons. They do not establish that numerical perturbations caused the observed switches, that the system is chaotic, or that the small residual-system ranking difference is practically useful. Retraining variability remains unmeasured.

## Provenance

`analyze_neartie.py` is the original workstation extraction audit; it needs the full restored archive and is not needed for the compact reproduction above. It verified the trace, condition, and metrics SHA-256 entries in all 96 returned case manifests. Every one of the 57,504 scored records was also compared with its original restored trace: nine scores, validity mask, eligibility mask, and selected index all matched exactly. This original-source check is recorded in `near_tie_summary.json`; the compact script does not claim to repeat a check against raw archives that are absent from its package.

`score_gap_records.csv` gives derived per-request gaps and eligibility counts. `threshold_fractions.csv` and `episode_threshold_fractions.csv` retain the threshold counts and episode-level fractions. `near_tie_summary.json` includes source trace hashes, group counts, and quantiles. `implementation_evidence.md` separately documents centering, timing-ring coverage, and the FV predictor's observation boundary.
