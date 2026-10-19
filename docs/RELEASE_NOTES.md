# v1.3.0: Empirical latency Monte Carlo

Prepared 19 October 2026. Adds all 64 timing repetitions: 240 baseline reproductions and 9,216 new runs using five frozen methods plus FV hold on 24 fixed episodes. No outcomes are excluded.

## Added

- Episode metrics/components, full repetition summaries, distributions, rankings, paired effects, sampler design and scoped author audit receipts.
- Portable checks for 9,456 saved costs, 384 means, 18 distributions, 39 ranking events and 64 paired policy effects.
- Main Figure 5 with all 64 outcomes; expanded S1 guide and source map for fourteen main tables and eleven figures.
- Updated README, methods and interpretation of empirical percentile ranges versus Wilson intervals.

FV mean increases are 26.01% under zero and 3.72% under hold. FV is highest-cost in 61/64 zero-policy repetitions and rank 1 in 64/64 hold repetitions. Hold loses 388 individual episode pairs despite improving every repetition mean. Results condition on fixed models, episodes, empirical pools and IID sampling; they do not establish deployment safety or a general architecture ranking. Small seed repetitions and timing repetitions are separate experiments.

## Assets and scope

Extract `thermal-delay-supplement-v1.3.0.zip`, install its requirements and run `python -B Supplement_S1/scripts/reproduce_review.py`. `Supplement_S1_guide.pdf` summarizes the evidence. Check assets against SHA256SUMS. Code/documentation: MIT; data/research figures: CC BY 4.0.

The 5.1-GB raw Monte Carlo archive, full timing pools and weights are omitted. Author raw-audit receipts are distinct from executable saved-metric verification. Earlier releases remain unchanged. No new simulation or publication is performed by the reproduction commands.
