# Version 1.2.0: Small initialization and saved-trajectory evidence

Companion to the manuscript revision prepared 19 October 2026.

## Added

- All 192 Small test outcomes: 48 original-checkpoint reproductions and 144 tests from three new initializations.
- All three training histories, 288 validation-selection outcomes, selected-alpha evidence, quality flags, and per-seed paired intervals. Seed 6062 fails the validation improvement criterion and is retained.
- The October 13 saved-record analyses of 57,504 scored requests, 336 FV traces, command restoration and thermal responses.
- A nine-page S1 guide, four main figures, six supplementary figures and complete evidence map.
- Portable seed and combined reproduction commands; SHA-256 integrity manifests.

## Interpretation

All three added Small seeds have lower replay mean cost than the fixed Shared checkpoint; zero-delay ordering varies and only one new seed reverses ordering. These are 24 reused episodes, with fixed upstream components, calibration and archived timing. Shared is not retrained. No independently sampled latency Monte Carlo result or general architecture/pretraining claim is included.

## Assets

`thermal-delay-supplement-v1.2.0.zip` contains the compact public supplement. Extract it and run `python -B Supplement_S1/scripts/reproduce_review.py` after installing its requirements. `Supplement_S1_guide.pdf` summarizes the evidence. Check both against SHA256SUMS.

The original v1.0.0 raw-data assets and v1.1.0 release remain unchanged. New Small raw trajectories, weights and full training inputs are omitted. Raw-quadrature and remote-input checks are author audit receipts; compact saved-metric analyses are executable here. Code/documentation: MIT; data/research figures: CC BY 4.0.
