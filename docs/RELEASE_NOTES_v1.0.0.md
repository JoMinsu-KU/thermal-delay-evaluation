# Version 1.0.0

Research data and portable analysis accompanying *Inference-Delay-Aware Evaluation of Learning-Based Thermal Controllers: A Retrospective Simulation Study*.

Includes original A/B/C/D evidence, 180,000 latency request records, and 741 numerical-validation runs. The extended validation contains 732 runs; the earlier C closed-loop batch contains nine.

The small repository supports paired episode statistics and figure regeneration. Two optional raw-data release assets support independent quadrature, field, journal, and baseline-event audits. SHA256 manifests and citation metadata accompany the package.

Code: MIT. Data and research figures: CC BY 4.0.

Repository: https://github.com/JoMinsu-KU/thermal-delay-evaluation

Extract both raw-data assets into the repository root, then run `python scripts/verify.py --raw-dir raw_data` and `python scripts/audit_raw.py --raw-dir raw_data`. Check the archive hashes against the attached `SHA256SUMS` before extraction. The repository alone supports `python scripts/reproduce.py` for the small-data analysis and four figures.

Known scope limits: selected checkpoints and reused timing sequences; C has one selected episode; D refinement uses 24 normal test episodes; original model weights and full training inputs are not included. Earlier C tracking quadrature is unavailable.
