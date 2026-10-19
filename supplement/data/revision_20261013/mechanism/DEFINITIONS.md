# Post hoc analyses of saved control traces (2026-10-13)

These definitions were recorded before computing the revision summaries. They
are post hoc definitions: the original outcomes and reviewer concerns were
already known. No new simulation, inference, training, or parameter selection
is performed. The 24 nominal held-out episodes remain the sampling units.

## Deadline response

Use every FV-MPC deadline event under zero and hold in the seven saved timing
settings (original replay and six within-trial blocks). For each event retain
the four command components immediately before expiration and after fallback.
Define full-vector restoration as the first subsequent application at which
every command component is at least its own pre-expiration value (tolerance
1e-8). This measures restoration of command levels, not recovery of plant
temperature or optimal control. A controller may appropriately choose a lower
level as temperature changes. An all-zero pre-expiration vector is already
restored and is counted separately. Hold is restored at expiration by design.

For a nonzero reset, stop follow-up immediately before the next deadline
fallback, or at 60 s if no later expiration occurs. If the criterion has not
been met, right-censor at that boundary. Do not replace censored times by the
boundary or report a mean over all events as if recovery were observed.
Record conditional recovery times only among completed restorations and show
the counts of completed and censored events. Events within an episode are
dependent; do not use event-level confidence intervals or hypothesis tests.

The ordinary nine-candidate neighborhood changes at most one heater by 0.04
per selected command. The minimum number of positive coordinate moves from
zero to the previous vector is sum_j ceil((u_j - 1e-8)/0.04), clipped below at
zero. This is a decision-count lower bound, not the largest individual-heater
count. Multiplication by 0.1 s gives equivalent nominal decision slots, not an
exact wall-time bound, because application times vary and decisions may be
skipped. Also retain the first post-expiration nonfallback command time.
Validate the one-coordinate limit in every nonfallback applied transition.

For descriptive association with cost, summarize expiration count, timing,
pre-expiration mean command, and mean-temperature error at expiration by
episode and setting, then show the seven setting means alongside relative
FV cost increases and matched zero-minus-hold cost. Do not estimate a timing
effect causally from these associations. Settings are fixed, not independent
timing draws. The same 24 episodes are reused and uncertainty over new timing
realizations is not identified by this analysis.

## Physical response metrics

Use the saved 10-Hz spatial-mean temperature y(t) and constant setpoint r over
0--60 s. Report the following for the five original zero-delay and replayed
pipelines and original-replay FV hold (11 conditions, 24 nominal episodes):

1. Final-10-s mean absolute tracking error: (1/10) integral_50^60 |y(t)-r| dt,
   trapezoidal quadrature on the saved samples. This describes late-horizon
   error, not a proof that steady state was reached.
2. Sampled mean-temperature overshoot: max(0, max_t[y(t)-r]). This is neither
   the spatial maximum nor the continuous-time peak between 0.1-s samples.
3. Sampled settling time in a fixed +/-1 K band: first saved time t such that
   all samples from t through 60 s satisfy |y-r| <= 1 K. If none, right-censor
   at 60 s. Report the number settled and conditional mean among settled
   episodes; report no fabricated settling time for censored episodes.
4. The absolute initial-to-goal temperature difference and the fraction of
   sampled squared spatial-mean tracking error accrued during 0--10 s are
   descriptive support for the contribution of the warm-up transient.

Means use equal episode weights. These mean-temperature quantities differ
from the main paper's spatial-field tracking RMSE. The eight initially hot
episodes are outside this returned nominal-only batch and are not combined
with these 24 episodes. No bootstrap intervals are introduced for these
post hoc response metrics.
