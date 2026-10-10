# Supplementary command diagnostics

## Figure S1. First command divergence in the selected C case

`outputs/figures/first_command_divergence.pdf` and `.png` retain the archived C command diagnostic. The detailed protocol, conditional interpretation, and score-margin discussion are in `methods/selected_C_detail.tex`. C uses a different frozen direct scorer from D. Its episode was selected post hoc; it is not a representative performance estimate for the primary comparison.

## Figure S2. FV-MPC commands around the first deadline miss

![Figure S2](outputs/figures/deadline_command_response.png)

Applied commands for all four heaters during 6.4–9.0 s of episode `l2_240070`, under the original interleaved replay at a 10-ms plant cap. Solid orange curves use zero fallback; dashed blue curves use hold fallback. Dotted vertical lines mark deadline expirations at 6.8 and 7.0 s. Shading spans the interval from the first expiration to the next nonfallback application at 7.177863943 s. Both policies have the same request, deadline, and completion schedules. Curves use actual saved application timestamps with right-continuous holds, without interpolation or new simulation.

Immediately before 6.8 s, the applied command is [0.64, 0.68, 0.64, 0.68]. Zero fallback resets all four values to zero, whereas hold retains them. The next nonfallback command is [0.04, 0, 0, 0] with zero fallback and [0.68, 0.68, 0.64, 0.68] with hold. Subsequent commands are recalculated from each condition's evolving state, so the figure shows the deadline intervention and its downstream feedback effects. It does not isolate every contribution to the final temperature difference or establish a safe hardware policy.

The episode is unchanged from main Figure 5: it was selected as the lower median by zero-delay FV-MPC cost, before the hold curve was added. `data/minor_revision_r2/deadline_command_evidence.json` records the source hashes and author audit scope. The compact script reconstructs command penalties and the plot; it does not repeat the omitted raw trace/seal audit.
