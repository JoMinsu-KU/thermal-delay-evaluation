# Reproduction scope

Run `python -B scripts/reproduce_all.py` after installing requirements. It verifies distributed inputs, recalculates saved-result statistics and writes five main and six supplementary PNG/PDF figures under `supplement/outputs/`. No GPU or fresh research execution is required.

Monte Carlo inputs contain 9,456 episode metrics with stored cost components. The script verifies the full 64-by-six-by-24 new-run roster, 240 baseline cases, component arithmetic, group means, all archived distributions and rank probabilities (Wilson intervals), and paired FV effects. The absolute ranking/tie tolerance is 1e-10. The same 24 episodes recur across independent timing repetitions conditional on fixed models and empirical IID pools. Percentiles are repetition spread, not mean confidence intervals.

Small inputs retain all 192 tests, 288 validation-selection rollouts and three training histories, including seed 6062's failed improvement criterion. They use original replay and are not factorially crossed with Monte Carlo timing. Their intervals resample matched episodes conditional on each fixed checkpoint pair and timing sequence.

The original and fixed-replay analyses, score margins, command restoration, and sampled thermal summaries remain executable. The complete table/figure map is `supplement/TABLE_FIGURE_MAP.md`. Historical output filenames are retained for provenance.

Full quadrature, temperature fields, model weights and timing pools are omitted from the compact update. Raw-cost, CRC/seal, schedule and sampled resampling audits are author receipts. They cannot be independently repeated from summary files alone. Optional v1.0.0 raw assets support the earlier separate original/numerical audits. The package does not execute new closed-loop conditions or measure a hardware failure rate.
