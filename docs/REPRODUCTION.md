# Reproduction scope

## 1. Recalculate summaries using the repository

`python scripts/reproduce.py` reads released episode records, recomputes paired bootstrap intervals with 20,000 resamples and seed 20261007, compares all 27 numerical contrasts against the audited values, and generates the four figures.

The bootstrap resamples episodes as pairs. It preserves the original episode order from `data/protocol/extended_protocol.json`. Percentage changes in Tables 6 and 8 are ratios of means, not means of individual percentage changes. Win/tie/loss uses an absolute reporting tolerance of 1e-10; this is not a solver-error estimate.

The source latency summary is exported at this level. Figure 4's fixed-field differences use the released audited summaries. These two quantities are independently reconstructed from raw files at level 2.

## 2. Reconstruct metrics from raw release records

Extract both raw ZIP assets, then run:

```bash
python scripts/verify.py --raw-dir raw_data
python scripts/audit_raw.py --raw-dir raw_data
```

The audit checks:

1. All 732 extended costs using stored tracking quadrature, command effort, effective command increments, and final temperature fields.
2. All 240 D baseline metric/event reproductions, plus three C fixed-command baseline metric reproductions.
3. Earlier nine C closed-loop costs using reported tracking RMSE, independently reconstructed command terms, and final fields. This earlier batch lacks raw tracking quadrature; its tracking term is not independently reconstructed.
4. C fixed-command field RMSE and maximum pointwise difference at 1 Hz.
5. All 720 D episode/step-pair comparisons of requested commands and logged mean-temperature trajectories at 10 Hz.
6. All 180,000 latency journal row hashes, 60 repeat/pooled summaries, and 71,808 normal held-out replay request mappings.

`verify.py` checks SHA256 manifests. A checksum mismatch stops verification. It does not repair inputs or relax numerical tolerances. Scientific raw records are copied byte-for-byte from the source archives; metadata normalization is recorded in `data/source_manifest.jsonl`.

## 3. Inspect the simulation implementation

`experiment_source/C/` preserves the selected direct-scoring implementation. `experiment_source/D/` preserves the later residual-comparison implementation. The validation harnesses are under `experiment_source/harness/`.

These are provenance snapshots. Executing them requires the original project layout, selected checkpoints, pretrained Open-Jev/Qwen model dependencies, normalized data/caches, and calibration inputs. The original experiment environment was Linux on H200, using the author's Conda `jev` environment and resource limits. Training, inference, or plant simulation was not run as part of public-package validation.

The complete original training package is not redistributed here. This release supports saved-result reanalysis and implementation inspection, not a fresh end-to-end training claim.

## Numerical interpretation

The changed variable is the actual plant's maximum inner FV time step. The spatial grid remains 64 × 64 in FP64, and exact actuator averaging is retained. Controller predictions, sensor emission (100 Hz), thermal frames (5 Hz), requests (100 ms), and the archived latency matching rule remain unchanged. Cost quadrature and peak sampling use original 10-ms/event right endpoints, not every finer inner FV step. An unchanged command/fallback state does not introduce another integration boundary.

The extended validation excludes initially hot and OOD episodes. Aggregate mean comparisons are preserved across tested caps, while individual trajectories can change. Spatial convergence, finer predictor steps, numerical-precision changes, and continuous-time peak bounds remain outside the validation.
