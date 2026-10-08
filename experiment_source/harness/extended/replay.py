"""Replay archived actual applications without executing a controller."""
import numpy as np
from validation_io import require


def replay_commands(plant, trace, stop=None):
    last_event = 0
    applied = 0
    for row in trace:
        if row["event"] not in ("apply", "deadline_fallback"):
            continue
        t = row["time_ns"]
        require(last_event <= t <= plant.end_ns, "Archived application clock is invalid")
        last_event = t
        u = np.asarray(row["command"], dtype=np.float64)
        unchanged = bool(np.array_equal(u, plant.u) and row["fallback"] == plant.fallback)
        require(unchanged == row["no_change"], "Archived no-op flag differs from applied-command state")
        # Introducing an integration boundary at an original no-op would change
        # the experiment. Exactly match original simulate's no-op handling.
        if not unchanged:
            plant.advance_to(t)
            plant.apply(u, row["fallback"])
            applied += 1
        if stop is not None and stop.signal:
            raise InterruptedError("Fixed-command replay interrupted")
    plant.advance_to(plant.end_ns)
    return {**plant.metrics(), "effective_applications": applied,
            "mode": "fixed_archived_applications", "policy_inference_executed": False}
