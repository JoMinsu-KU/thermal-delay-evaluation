# v1.3.3: Final submission documentation

Prepared 19 October 2026. This package consolidates the prepared v1.3.2 narrative/evidence-map update and the final submission corrections. The last public version checked during preparation was v1.3.1. Run this package directly; publishing the older prepared v1.3.2 package first is unnecessary.

The manuscript cites each of its 17 references once, in first-appearance order, using bracketed superscripts. Each cited sentence refers to one source. The bibliography retains two Korean journal papers. The sources themselves are unchanged.

Supplement METHODS.md now points to Figure 5 for thermal response, Equation (3) for cost normalization, and Figure 3 / Tables 5–6 for Monte Carlo results. Historical method fragments also use one source per citation. The main text distinguishes offline checkpoint selection from correction-strength selection with closed-loop validation, and specifies full-field per-episode tracking RMSE before averaging. Current table and figure numbering is recorded in supplement/TABLE_FIGURE_MAP.md.

All 804 scientific files under supplement/data, all analysis scripts, and the research figures are unchanged. No new experiment or numerical result is added. The revised guide is included in docs/ and the release assets. Manuscript PDFs, private author/funding records, and internal reviews are excluded from this public package.

Code/documentation: MIT. Data/research figures: CC BY 4.0. The compact package supports saved-result reanalysis and figure regeneration; the scope of raw-archive checks is recorded separately.

---

# v1.3.2: Manuscript evidence map and supplementary documentation

Prepared 19 October 2026. The manuscript now presents the matched Monte Carlo deadline-policy comparison before fixed-replay recovery and sensitivity results. This update synchronizes the title, 11 main-table entries, five main-figure entries, and the supplementary guide. Analysis chronology is Table S3; score margins remain Table S2. PROTOCOL_DETAILS.md and a supporting TeX fragment retain the detailed methods and verification scope relocated from the main text.

All 804 scientific files under supplement/data and all eleven figure images are unchanged. Unfavorable episode outcomes, the zero-policy late-response advantage, failed seed 6062 and nonmonotone numerical sensitivity remain available. No new experiment is added. Figure 1 retains the v1.3.1 monochrome design.

The compact supplement supports saved-result analysis and figure regeneration. Code/documentation use MIT; data/research figures use CC BY 4.0. Earlier releases and assets remain unchanged. Manuscript PDFs, private author records and internal review notes are not included in this public package.

---

# v1.3.1: Monochrome Figure 1

Prepared 19 October 2026. This formatting update replaces Figure 1 with monochrome rectangular nodes, equal-length straight connecting arrows, and one attached orthogonal feedback route from plant to observation. It provides vector PDF/SVG and a 300-dpi PNG, with an independent portable generator called by the existing figure driver.

The five scientific stages, their execution order, experiment data, reported results and other figures are unchanged. All v1.3.0 Monte Carlo and prior analyses are retained. The guide has unchanged scientific content. Code/documentation: MIT; data/research figures: CC BY 4.0.

Extract `thermal-delay-supplement-v1.3.1.zip` and run `python -B Supplement_S1/scripts/reproduce_review.py` with NumPy and Matplotlib installed. For Figure 1 only: `python -B Supplement_S1/scripts/make_evaluation_flow.py --output figure_output`. Verify assets with SHA256SUMS.

Earlier v1.3.0 assets and tags remain unchanged.

---

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
