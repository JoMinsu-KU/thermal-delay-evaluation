# Reproduction scope

Run `python -B scripts/reproduce_all.py` after installing NumPy and Matplotlib. This runs the compact supplement, verifies hashes, reconstructs paired summaries and selection decisions, and regenerates the four main and six supplementary figures. No model, GPU, training, inference, new simulation or timing measurement is invoked.

For Small repeats, the inputs are 192 saved test metrics, three full recorded training histories, the original training history and fixed Shared reference, 288 alpha-selection outcomes, and provenance/audit receipts. The script independently recomputes costs contrasts, 24-episode conditional bootstrap intervals, win counts, selected checkpoint/alpha and the retained failed quality flag. It does not recompute full-field costs from omitted quadrature or reload omitted weights.

The October 13 compact inputs also support eligible-score margins, restoration from applied commands and sampled thermal-response metrics. `supplement/TABLE_FIGURE_MAP.md` maps all twelve tables and ten figures. Historical filenames and earlier files are retained for provenance; use the map rather than inferring current numbering from filenames.

Full training inputs, model weights, new Small raw trajectories and original timing journals are omitted from the compact package. Optional original primary/numerical raw assets at v1.0.0 support their separate original raw audits. The new Small raw-cost and remote-input integrity findings are author receipts. No random-latency Monte Carlo outcomes are included.
