# Monte Carlo schedule evidence: export and verification report

Date: 19 October 2026. Scope: existing FV-MPC Monte Carlo records only.

## Outcome

**PASS.** The compact evidence now supports an executable, all-case comparison of the actual saved zero/hold schedules. It includes both policy arms' records and all consumed latency draw indices, rather than only an author-written summary or a shared potential draw-plan receipt.

| Check | Result |
|---|---:|
| Repetitions × fixed episodes | 64 × 24 |
| Zero/hold pairs | 1,536 |
| Distinct original FV case logs | 3,072 |
| Exported schedule events | 3,667,828 |
| Saved latency draws independently recomputed | 1,824,628 |
| Requests per policy | 912,314 |
| Expired requests per policy | 9,091 |
| Busy-skipped nominal ticks per policy | 9,286 |
| Pairwise schedule mismatches | 0 |
| Pairwise consumed-draw mismatches | 0 |
| Draw/index/source-latency inconsistencies | 0 |
| Complete original FV timing pool | 36,000 rows |
| Distinct consumed source rows with recorded hashes | 31,188 |

For each policy, `912314 + 9286 = 1536 × 600`: every nominal decision tick is represented by either an admitted request or a recorded busy skip. The two policies' separate actual logs support the matching claim.

## Verification performed

1. The extractor read every selected case's original condition, metrics, trace, draw records, draw-plan identity, manifest, and completion seal directly from the ZIP. It checked selected-member SHA-256 values against the original manifests and the case binding to the original protocol. Every selected ZIP member passed the ZIP reader's CRC check.
2. A separately implemented offline verifier read only the compact export. It checked every pair's actual event schedule, every case's 600-tick admission/skip sequence, each request's strict expiration rule and event time, and every consumed draw against an independent SHA-256 rejection-sampling implementation and the complete original FV timing pool. This completed in **49.8 s** for the full local bundle.
3. The verifier's optional raw-archive mode reread **all 3,072 original FV trace/draw logs** and compared each exported field and factored dictionary value directly to those raw records. It also independently recomputed all draws and schedule checks. This completed in **150.1 s**, including archive identity hashing. It was not a sampled spot check.
4. The anonymous review subset passed the same complete offline checks in **48.9 s**. It verified all 18 distributed protocol-pinned harness files, while explicitly accounting for the two withheld attribution metadata files.
5. Targeted corruption checks accepted the unaltered first case and rejected a 1-ns readiness change, a flipped expiry flag, a removed busy-skip event, and a 1-ns application-time change.
6. A decompressed-content scan of the anonymous subset found zero matches for the requested author identities, account identifier, email marker, or GitHub/GitLab repository patterns. Code and source metadata are preserved as described below.

Integer times, flags, identifiers, and paired event records are compared exactly. Floating latencies are preserved by round-trip serialization and compared exactly after parsing; no tolerance is applied. Readiness consistency uses the original operation `ceil(delay_ms × 10^6)` and expiry uses strict `ready_ns > deadline_ns`. The complete schema and distinction between original fields and derived checks are in `README.md`.

## Original archive and code provenance

```text
Archive: latency_monte_carlo_20261013T060637Z_results.zip
Bytes:   5101034440
SHA256:  1f3336baef5be7c4725dd0dbc4fdb677bbb4f429a05d9b6e8ad438d7663d52e5

Original protocol SHA256:
baac429a756844d8a4f5c9d0aa69b895dd61ac25388a67089aeee49b95c97c4c

harness/latency_replay.py:
e9b689d182420ca0db60396027cfcbe4cf8e58381dbcaa61af7dd592a0f28a38
harness/scheduler.py:
a3b601858c8f640b898d6284d90529b2538fc96859d3d80acf7941a0f3ef1d29
harness/worker.py:
6bc3346bb5813d1f4117fa4d232da10d4a1a3d92c465ede82f426d44cbbbea8b
harness/validation_io.py:
6ce7d724614c131b442efbe4b0c2ccbb47816a976174363cc4c67809f186bbf0
```

All members lie under archive root `latency_monte_carlo_20261013T060637Z/`. Actual evidence is read from:

- `cases/monte_carlo_fv_mpc_delayed_mcXXXX_{zero,hold}_l2_XXXXXX/attempt_.../trace.jsonl.gz`
- The corresponding `replay_rows.json.gz`, `condition.json`, `metrics.json`, `manifest.json`, `COMPLETE`, and draw-plan identity
- `timing_pools/fv_mpc/rows.json.gz` and its original manifest/seal
- `harness/` members pinned by `protocol.json`

The original worker verifies the protocol's harness hashes, imports the supplied sampler/scheduler, and writes the saved trace and sampler records. Each case manifest binds its logs to that protocol. Full member paths and file hashes are retained in `cases.jsonl.gz` and `export_provenance.json`.

The saved sampler records explicitly contain `pool_index`, `source_interleaved_row`, `source_repeat`, `source_request`, matched-pool hash/size, seed/method/episode/replicate, and source-row hash. All were present in the original logs. Repeated context and source hashes are losslessly factored into dictionaries; the raw-archive check verifies this factoring for every record.

## Integration

Use the **`anonymous_review/`** variant for blind-review S1. Its contents may be placed at `data/monte_carlo_schedule_audit/`, with execution:

```bash
python -B data/monte_carlo_schedule_audit/verify_mc_audit.py --output outputs/monte_carlo_schedule_audit
```

The full local/public bundle preserves the original attribution metadata. The anonymous variant omits only `archived_source/LICENSE` and `archived_source/DATA_LICENSE`, records their names/original hashes/reasons in provenance, and excludes them from its required-input manifest. All distributed source code is unchanged. The public repository retains the full MIT code and CC BY 4.0 notices at its root. The review variant is an explicitly documented subset, not a full archive copy.

Core exported inputs occupy **32.85 MB** (decimal). The largest files are the schedule CSV/XZ (20.16 MB), draw CSV/XZ (5.39 MB), original FV pool (3.60 MB), original case metadata (2.06 MB), and saved source-row hashes (1.26 MB). Extraction took **266.5 s** and read about 1.09 GB of selected compressed trace/draw members. The direct field representation was retained to keep the evidence simple to inspect; the size is above the preferred 20-MB target.

## Interpretation boundary

The `ready_ns`/completion and expiry fields are original saved simulation-scheduler records, calculated during that experiment from sampled measured latencies. They are not newly measured GPU completion times. This export makes the realized scheduler and sampled-index evidence executable without a GPU or the full archive.

The checks establish consistency and equality of the supplied schedule/draw records. They do not re-execute model inference, thermal dynamics, command values, costs, training, or hardware timing. Unused potential preflight draw plans, full fields, quadrature, weights, and other methods' traces/pools are omitted. Finite-pool uncertainty and serial dependence remain outside the experiment's IID sampling assumptions. Hashes establish integrity and provenance relationships within the supplied archive, not independent authentication of the original experimental run.

No main manuscript, canonical S1, original archive, or GitHub package was modified by this subtask.
