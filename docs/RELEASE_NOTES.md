# Version 1.1.0 — replay and deadline-policy evidence

Companion data for *Inference Delay and Deadline Handling in Thermal Controller Evaluation: A Retrospective Simulation Study*, manuscript revision of 10 October 2026.

## Added

- Compact outcomes for 1,128 completed cases: 240 original-condition reproductions, 720 additional-replay runs, and 168 FV hold runs, using the same 24 normal episodes.
- All 57 paired replay/fallback contrasts, sequence summaries, method rankings, and fixed protocol records.
- Cost-component and physical-effect checks, alpha-selection evidence, and representative hold temperature inputs.
- First-deadline applied-command records, reconstructed increment penalties, and Figure S2.
- Six main figures, two supplementary figures, the complete table/figure map, and a three-page S1 guide.
- Byte-preserved scheduler, latency-matching, protocol, and metric-check source excerpts.

## Reproduce

Install `requirements.txt` and run `python -B scripts/reproduce_all.py`. Current outputs are in `supplement/outputs/`. The standalone ZIP contains equivalent inputs and can be run with `python -B Supplement_S1/scripts/reproduce_review.py` after extraction.

## Assets and scope

This release contains the small supplement ZIP and its guide PDF. Check both against the attached SHA256SUMS. The two large raw-data assets remain at v1.0.0; their bytes and tag are unchanged. The full raw replay/hold batch is not distributed in v1.1.0. Saved-result checks do not rerun training/inference, perform new measurements, or independently repeat the omitted raw-input audit.

Additional replay/hold settings were tested at 10 ms only. Their settings reuse three archived timing repeats, not six independent hardware measurements. Intervals condition on fixed systems and timing sequences.

Code/documentation: MIT. Data/figures: CC BY 4.0. Authors: Minsu Jo, Namhyun Yoo, Hoansuk Choi, Jinhong Yang. The associated manuscript is in preparation; no journal acceptance is claimed.
