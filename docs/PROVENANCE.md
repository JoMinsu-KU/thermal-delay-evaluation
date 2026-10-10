# Version 1.1.0 extension

For the 10 October 2026 manuscript, use `python -B scripts/reproduce_all.py`; current results appear in `supplement/outputs/`. See [the complete map](../supplement/TABLE_FIGURE_MAP.md), [methods](../supplement/METHODS.md), and [supplement README](../supplement/README.md). The new 1,128-case batch reuses 24 episodes. Its full raw records are not included. Original raw assets at v1.0.0 cover the earlier primary/numerical evidence only.

The material below documents the original v1.0.0 data and tools, which are retained for compatibility.

---

# Provenance and publication scope

Prepared version: 1.0.0, 8 October 2026.

The package was assembled from the restored research snapshot and two returned numerical-validation archives. The original snapshot and result archives were not modified. Original case seals were checked before copying the selected scientific records. `data/release_scope.json` records input archive hashes and the covered populations; `data/source_manifest.jsonl` maps source files to released files.

## Preserved and normalized content

Scientific raw metrics, request/event journals, quadrature arrays, and temperature fields were copied byte-for-byte. Summary exports preserve numeric values. Absolute metadata paths were normalized, and environment identities, import-path inventories, sessions, and console/resource logs were omitted. As a result, the raw release is a curated export, not a byte-identical copy of the original ZIP. New manifests verify the exported files; the original all-file seals are not represented as complete public seals.

The public package includes source snapshots for the two relevant implementations and their validation harnesses. The portable analysis scripts were prepared from the archived analyses. Their checks operate on saved records only and import no experiment/model/simulator modules.

## Excluded material

- Pretrained or fine-tuned model weights, model caches, generated training inputs, and the full multi-gigabyte snapshot.
- Downloaded research-paper PDFs and literature-search full text.
- Journal classes/fonts and manuscript source/PDF; these are separate author deliverables.
- Author biographies/photos, private machine identity details, internal reviewer conversations, and intermediate manuscript drafts.

No credentials were intentionally included. A bounded pattern scan of staged text and decompressed JSONL/CSV files checked common credential formats and known local machine/path identifiers; no matches remained in the delivered package. This records the performed scan rather than claiming a universal secret-detection guarantee.

## Licenses

The author selected MIT for code and CC BY 4.0 for data. The root MIT text applies to the study-owned analysis and experiment source plus its documentation. `DATA_LICENSE` applies CC BY 4.0 to numeric records, data manifests, research figures, and raw release assets. External dependencies keep their own terms and are not bundled. The CC BY legal text was obtained from the SPDX license-list mirror and checked against the canonical [Creative Commons license](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## Limits of this record

The checks establish consistency of released saved outputs within their stated scope. They do not independently repeat model training, reproduce hardware-dependent inference timing, or demonstrate general numerical convergence. C and D have different controller/checkpoint lineages. Repeated latency prefixes and integration caps do not enlarge the independent episode sample.
