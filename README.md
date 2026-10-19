# Thermal Delay Evaluation

**Data and reproducible analysis of inference delay, deadline commands, and thermal-controller performance.**

Companion to *Inference Delay and Deadline Handling in Thermal Controller Evaluation: A Retrospective Simulation Study*, by **Minsu Jo, Namhyun Yoo, Hoansuk Choi, and Jinhong Yang**. The English manuscript is in preparation for J-KICS; no acceptance is claimed. **Version 1.2.0** corresponds to the revision prepared on **19 October 2026**.

[Quick start](#quick-start) · [New seed results](#small-initialization-results) · [Evidence map](supplement/TABLE_FIGURE_MAP.md) · [S1 guide](docs/Supplement_S1_guide.pdf) · [Scope](docs/REPRODUCTION.md) · [Citation](#citation-and-license)

## What this study examines

Five implementations choose heating commands for the same simulated four-heater plate: PI, FV-MPC, PINO-MPC, Shared-Residual and Small-Residual. Delayed commands interact with a 100-ms deadline and a candidate set that changes at most one heater by 0.04 per decision. The primary comparison uses **24 nominal held-out episodes**. Timing settings and training repeats reuse these episodes.

| Finding | Evidence | Interpretation |
|---|---|---|
| Deadline commands change the observed FV-MPC loss. | Original replay raises mean cost by 25.02% under zero fallback and 2.35% under hold, relative to zero delay. | Matched request/deadline schedules; later states and commands respond to the intervention. |
| Loss varies across fixed timing settings. | Across seven settings, zero increases cost by 11.70–79.34%; hold by 2.19–6.46%. | Recorded blocks are sensitivity cases, not a random-latency distribution. |
| Recovery after forced zero can be prolonged. | 715 nonzero resets; 324 observed full-vector restorations and 391 censored. | Restoration is to a historical command, which need not remain thermally desirable. |
| Small's replay advantage repeats, while its zero-delay ordering varies. | All three added seeds have lower replay mean cost than one fixed Shared checkpoint; only one reverses ordering. | Conditional Small initialization study; one validation-quality failure is retained. |

![FV cost sensitivity to replay and deadline policy](figures/replay_fallback_sensitivity.png)

## Small initialization results

Percentages below are **100×(Small mean / Shared mean−1)**. Negative values favor Small. Each row uses the same 24 nominal episodes and one fixed Shared checkpoint.

| Small initialization | Selected alpha | Zero delay | Original replay | Validation improvement |
|---|---:|---:|---:|---|
|6060 (original)|0.5|+0.275%|−0.373%|Pass|
|6061|1.0|−0.349%|−1.049%|Pass|
|6062|0.5|+0.147%|−0.412%|**Fail; retained**|
|6063|1.0|−0.409%|−1.034%|Pass|

All three new models complete 450 optimizer steps but select warmup step 50. Seed 6062's final validation FV candidate-cost difference is 6.3398e−7, above the initial 6.2609e−7; infeasible selections and unexpected fallbacks are both zero. Its alpha is chosen by the separate, prespecified closed-loop validation rule. The failed outcome is included in every relevant summary.

![Small initialization comparison with conditional episode intervals](figures/small_seed_cost_comparison.png)

The new tests reuse Small's archived latency and calibration. Shared is not retrained. There are **24 unique test episodes**, not 72 independent observations per timing mode. Intervals resample matched episodes conditional on each checkpoint pair and fixed timing. The observed one-of-three reversal count is not a population probability. Architecture, pretraining and selected alpha are not isolated causal effects. The new models are evaluated only at the 10-ms plant cap.

## Quick start

Python 3.10+ with NumPy and Matplotlib is sufficient; no GPU or model download is needed.

```bash
python -m pip install -r requirements.txt
python -B scripts/reproduce_all.py
```

This verifies package hashes and regenerates the manuscript analyses, **four main figures and six supplementary figures**, including per-seed paired contrasts and validation checks. Outputs are written to `supplement/outputs/`; frozen inputs are unchanged.

For only the seed study:

```bash
python -B supplement/scripts/reproduce_small_seed.py
python -B supplement/scripts/make_small_seed_figure.py
```

Inspect `supplement/outputs/small_seed/checks.json` and `paired_contrasts.csv`. The compact inputs and expected values are in [supplement/data/small_seed](supplement/data/small_seed). The complete [table/figure map](supplement/TABLE_FIGURE_MAP.md) identifies every source and generator. Historical output filenames preserve their original numbering.

## Data populations and experimental units

| Evidence | Executions/records | Independent episode scope |
|---|---|---|
| Original nominal comparison | Five methods×two modes×24 episodes | 24 nominal test episodes; eight initially hot episodes reported separately |
| Numerical checks | 720 D runs plus 21 selected-C checks | Same 24 D episodes; C is one post hoc selected case |
| Replay/deadline-policy batch | 1,128 executions, including 240 baseline reproductions | Same 24 episodes; seven fixed timing settings |
| Command/margin/response analysis | 336 FV application traces; 57,504 scored residual requests; 552 curves | Dependent events within reused episodes |
| Small initialization study | 48 baseline tests+144 new tests+288 validation rollouts | Three new Small seeds; 24 test episodes; 16 separate validation episodes |

No independent random-latency Monte Carlo outcomes are included in this version. Existing checkpoint comparisons remain original-checkpoint results unless explicitly identified as new Small repetitions.

## Download and audit the detailed records

Original primary and numerical-validation event logs and temperature arrays remain available in [Release v1.0.0](https://github.com/JoMinsu-KU/thermal-delay-evaluation/releases/tag/v1.0.0).

| Release asset | Size | Contents |
|---|---:|---|
| [Primary experiment records](https://github.com/JoMinsu-KU/thermal-delay-evaluation/releases/download/v1.0.0/thermal-delay-primary-raw-v1.zip) | 113 MiB | Original D metrics/events, 180,000 latency request records, and original selected C traces |
| [Numerical validation records](https://github.com/JoMinsu-KU/thermal-delay-evaluation/releases/download/v1.0.0/thermal-delay-numerical-raw-v1.zip) | 263 MiB | All 741 validation runs, with quadrature for the 732-run extended batch and fixed-command C fields |
| [SHA256SUMS](https://github.com/JoMinsu-KU/thermal-delay-evaluation/releases/download/v1.0.0/SHA256SUMS) | <1 KiB | SHA256 checksums for the two ZIP files |

These are the unchanged v1.0.0 assets. They do not contain the later replay/hold batch. Both ZIPs expand into the Git-ignored `raw_data/` directory. Allow approximately **1 GiB** for the expanded records in addition to the ZIP files.

With [GitHub CLI](https://cli.github.com/):

```bash
gh release download v1.0.0 --repo JoMinsu-KU/thermal-delay-evaluation --pattern "thermal-delay-*-raw-v1.zip" --pattern "SHA256SUMS"
```

<details>
<summary>Linux/macOS: verify, extract, and audit</summary>

```bash
# Linux; on macOS use: shasum -a 256 -c SHA256SUMS
sha256sum -c SHA256SUMS
unzip thermal-delay-primary-raw-v1.zip
unzip thermal-delay-numerical-raw-v1.zip
python scripts/verify.py --raw-dir raw_data
python scripts/audit_raw.py --raw-dir raw_data
```

</details>

<details>
<summary>Windows PowerShell: verify, extract, and audit</summary>

```powershell
Get-Content -LiteralPath SHA256SUMS | ForEach-Object {
    $hash, $name = $_ -split '  ', 2
    if ((Get-FileHash -LiteralPath $name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {
        throw "Checksum mismatch: $name"
    }
}
Expand-Archive -LiteralPath thermal-delay-primary-raw-v1.zip -DestinationPath .
Expand-Archive -LiteralPath thermal-delay-numerical-raw-v1.zip -DestinationPath .
python scripts/verify.py --raw-dir raw_data
python scripts/audit_raw.py --raw-dir raw_data
```

</details>

The raw audit writes to `outputs/raw_audit/`. It checks **732 independent quadrature cost reconstructions**, **243 baseline comparisons**, **12 fixed-field comparisons**, **720 D episode/step trace comparisons**, **180,000 journal row hashes**, and **71,808 replay-request mappings**. For the earlier nine C runs, tracking quadrature was not saved; their cost checks use reported tracking RMSE with independently reconstructed command terms and final fields.

Here, “independent reconstruction” describes recalculation from the saved records. It does not mean independently sampled runs or a new experiment.

## Reproduction scope

The compact repository reanalyses saved episode metrics, training histories, validation selection, application records and temperature summaries. It supports paired statistics, integrity checks and figure regeneration. The seed raw-cost audit reports a maximum reconstruction error of 2.78e−16 across 192 tests; that audit used full quadrature and fields that are **not included** in this compact update. Audit receipts document checks on the author's full archive and are not substitutes for independently repeating them.

The supplied subset omits model weights, complete training inputs, the full Small result ZIP and raw replay/fallback trajectories. It does not rerun training, inference, new plant trajectories, or timing measurements. Original primary/numerical raw releases remain available separately as described above. The seed inference-failure counter is not an independent fault-tolerance metric; the paper reports completed runs, thermal-limit outcomes and deadline counts.

## Repository layout

- `supplement/`: current compact inputs, methods, source map and portable analysis.
- `figures/`: publication figures in PNG/PDF; the map identifies the current ten figures.
- `docs/`: guide, release notes, reproduction scope and data dictionary.
- `data/`, `scripts/`, `experiment_source/`: retained original records and analysis/implementation excerpts.
- `MANIFEST.sha256`: integrity hashes for distributed repository files.

The public data package omits manuscript drafts, review correspondence, author biographies, the journal class and fonts. Those private submission materials are maintained separately.

## Citation and license

Use [CITATION.cff](CITATION.cff) and identify the version and analysis subset used. Authors: Minsu Jo, Namhyun Yoo, Hoansuk Choi and Jinhong Yang. No archive DOI has been assigned.

- **Code and documentation:** [MIT](LICENSE).
- **Data and research figures:** [CC BY 4.0](DATA_LICENSE).
- External software and models retain their own terms and are not redistributed here.

See [release notes](docs/RELEASE_NOTES.md) for changes from v1.1.0.
