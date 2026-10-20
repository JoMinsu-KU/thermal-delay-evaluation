"""Frozen L2 event order with one explicit intervention: expired-request command."""
import numpy as np


def simulate(plant, engine, mode, sampler=None, stop=None, fallback_policy='zero'):
    if mode not in ('zero', 'delayed') or fallback_policy not in ('zero', 'hold'):
        raise ValueError('Unknown simulation condition')
    if mode == 'delayed' and sampler is None:
        raise ValueError('Measured replay is required')
    if plant.entry.get('ood_group') is not None:
        raise ValueError('This protocol contains normal ID episodes only')
    engine.reset(); trace = []; pending = None; busy = 0
    requests = missed = skipped = failed = slew = 0
    if sampler is not None: sampler.reset()
    def until(end):
        nonlocal pending, slew
        if pending is not None and pending['event_ns'] <= end:
            if pending['expired']:
                command = np.zeros(4) if fallback_policy == 'zero' else plant.u.copy()
                fallback = True
            else:
                command = np.array(pending['decision']['command'])
                fallback = pending['decision']['fallback_reason'] is not None
                if not fallback and np.any(abs(command - plant.u) > .04000001):
                    command = np.zeros(4); fallback = True; slew += 1
            unchanged = bool(np.array_equal(command, plant.u) and fallback == plant.fallback)
            if not unchanged:
                plant.advance_to(pending['event_ns']); plant.apply(command, fallback)
            trace.append({'event': 'deadline_fallback' if pending['expired'] else 'apply',
                          'time_ns': pending['event_ns'], 'request': pending['request'],
                          'fallback': fallback, 'command': command.tolist(), 'no_change': unchanged})
            pending = None  # Expired inference output is discarded permanently.
        plant.advance_to(end)
    for step in range(round(plant.duration_s * 10)):
        now = step * 100_000_000; until(now)
        if now < busy:
            skipped += 1; trace.append({'event': 'busy_skip', 'time_ns': now, 'ready_ns': busy}); continue
        decision = engine.decide(lambda: plant.raw().snapshot_ns(now))
        delay, ref, gap = sampler.choose(decision, requests) if mode == 'delayed' else (0., None, None)
        compute_ready = now + int(np.ceil(delay * 1e6))
        ready = compute_ready; deadline = now + 100_000_000; expired = ready > deadline
        missed += int(expired)
        pending = {'request': requests, 'event_ns': min(ready, deadline), 'expired': expired, 'decision': decision}
        busy = compute_ready
        trace.append({'event': 'request', 'request': requests, 'snapshot_ns': now, 'ready_ns': ready,
                      'compute_ready_ns': compute_ready, 'deadline_ns': deadline, 'expired': expired,
                      'delay_ms': delay, 'injected_compute_ms': 0., 'injected_apply_ms': 0.,
                      'trace_request': ref, 'trace_token_gap': gap, **decision})
        requests += 1; until(min((step + 1) * 100_000_000, plant.end_ns))
        if stop is not None and stop.signal: raise InterruptedError('Restart current episode; keep completed episodes')
    until(plant.end_ns)
    return {**plant.metrics(), 'requests': requests, 'missed_deadlines': missed, 'busy_skipped_ticks': skipped,
            'inference_failures': failed, 'delayed_slew_rejections': slew, 'mode': mode, 'deadline_ms': 100,
            'injected_compute_ms': 0., 'injected_apply_ms': 0.,
            'trace_token_gap_max': max((r['trace_token_gap'] for r in trace if r['event'] == 'request' and r['trace_token_gap'] is not None), default=None),
            'filter_interventions': sum(bool(r.get('filter_intervention')) for r in trace)}, trace
