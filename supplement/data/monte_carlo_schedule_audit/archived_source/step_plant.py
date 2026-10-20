"""Plant-only time-step refinement; no model imports or global monkey patches."""
from __future__ import annotations

import math


def make_plant_class(base_class, advance):
    """Inject the frozen Plant and FV integrator after source verification."""
    class StepPlant(base_class):
        def __init__(self, base, entry, duration_s, *, max_step_ms, truth_stride=10, field_stride=0):
            if max_step_ms not in (10.0, 5.0, 2.5, 1.25):
                raise ValueError("Unsupported protocol step")
            self.max_dt = max_step_ms / 1000.0
            self.truth_stride, self.field_stride = truth_stride, field_stride
            self.truth_rows = []
            self.field_frames, self.field_times = [], []
            self.quadrature_rows = []
            self.fv_steps = 0
            self.max_actual_dt_s = 0.0
            super().__init__(base, entry, duration_s)

        def emit(self):
            super().emit()
            if self.field_stride and self.tick % self.field_stride == 0:
                if self.field_times and self.field_times[-1] == self.now:
                    self.field_frames[-1] = self.field.copy()
                else:
                    self.field_times.append(self.now)
                    self.field_frames.append(self.field.copy())
            if self.tick % self.truth_stride:
                return
            # L2 Plant emits twice at initialization. Keep one evaluation row.
            row = {
                "time_ns": self.now, "mean_temperature_c": float(self.field.mean()),
                "max_temperature_c": float(self.field.max()),
                "min_temperature_c": float(self.field.min()), "goal_c": float(self.goal),
                **{f"u{j}": float(v) for j, v in enumerate(self.u)},
                **{f"p{j}": float(v) for j, v in enumerate(self.p)},
            }
            if self.truth_rows and self.truth_rows[-1]["time_ns"] == self.now:
                self.truth_rows[-1] = row
            else:
                self.truth_rows.append(row)

        def advance_to(self, end_ns):
            if not self.now <= end_ns <= self.end_ns:
                raise ValueError("Plant clock must be monotone")
            # These boundaries and all metric updates match frozen L2 Plant.
            # Only the max_dt argument to the true-plant FV solver changes.
            while self.now < end_ns:
                next_tick = (self.tick + 1) * 10_000_000
                change = int(self.end_ns // 2)
                following = min(end_ns, next_tick, change if self.now < change else self.end_ns)
                dt = (following - self.now) / 1e9
                theta = self.nominal.copy()
                if self.now >= change:
                    if self.condition == "heater_gain":
                        theta[4] *= self.mismatch_value
                    if self.condition == "surface_loss":
                        theta[2] += self.mismatch_value
                self.field, self.p = advance(
                    self.field, self.p, self.u, theta, duration=dt, max_dt=self.max_dt
                )
                # Diagnostic count follows the frozen integrator's CFL formula.
                c, k, h = theta[:3]
                stable = .4 / (4 * (k / c) / (.5 / self.field.shape[-1]) ** 2 + h / c) if k or h else self.max_dt
                steps = max(1, int(math.ceil(dt / min(self.max_dt, stable))))
                self.fv_steps += steps
                self.max_actual_dt_s = max(self.max_actual_dt_s, dt / steps)
                mse = float(((self.field - self.goal) ** 2).mean())
                effort = float((self.u ** 2).mean())
                self.error_integral += mse * dt
                self.effort_integral += effort * dt
                peak = float(self.field.max())
                self.quadrature_rows.append((following, following-self.now, mse, effort, self.fallback, peak))
                violation = peak > 120
                self.violation_s += dt * violation
                self.fallback_s += dt * self.fallback
                self.peak = max(self.peak, peak)
                self.overshoot = max(self.overshoot, max(0., peak - self.goal))
                if violation and not self.previous_violation:
                    self.new_violation = True
                if self.initial_violation and not violation and self.recovery_s is None:
                    self.recovery_s = following / 1e9
                self.previous_violation = violation
                self.now = following
                if self.now == next_tick:
                    self.tick += 1
                    self.emit()

        def metrics(self):
            return {
                **super().metrics(),
                "integration": "FP64 FV with event-aligned equal substeps; plant max_dt varied only",
                "plant_max_step_ms": self.max_dt * 1000,
                "actual_max_fv_step_ms": self.max_actual_dt_s * 1000,
                "fv_steps": self.fv_steps,
                "cost_quadrature": "original right endpoints: 10 ms sensor ticks plus effective action events",
                "peak_sampling": "original 10 ms/event endpoints, not every inner FV substep",
                "truth_samples": len(self.truth_rows),
                "truth_logging_hz": 100 / self.truth_stride,
            }

    return StepPlant
