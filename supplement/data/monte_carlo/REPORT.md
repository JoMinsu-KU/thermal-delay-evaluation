# Empirical latency Monte Carlo results

State: **COMPLETE_VERIFIED**

Verified cases: 9456/9456; complete replicates: 64/64.

One replicate comprises the same 24 fixed nominal episodes with independent latency draws per episode and method. FV zero and hold share potential draw plans; coupling_audit.csv reports whether the realized timing paths remained identical.
All probabilities and Wilson intervals are conditional on the fixed timing pools, IID assumption, episodes and checkpoints. IID destroys recorded temporal correlation. Intervals quantify Monte Carlo sampling precision, not uncertainty over hardware, new plants, training seeds or the empirical timing distribution.
Partial-run statistics are provisional and must not replace the frozen complete design.
See ranking_probabilities.csv (including strict pairwise reversals), method_rankings.json (five-method rankings and ties), cost_effect_distribution.csv (replicate spread, not confidence intervals), paired_fv_effects.csv, and every raw episode in episode_metrics.csv.
No safety guarantee or positive-result requirement is imposed.
