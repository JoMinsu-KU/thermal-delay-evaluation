"""Stateless IID draws from sealed, path/token-matched empirical latency pools."""
import hashlib
import json
import math
from validation_io import require, object_hash


def uniform_index(domain, request_id, size):
    """Deterministic rejection-sampled uniform integer; independent keyed counters."""
    require(size > 0 and request_id >= 0, 'Invalid draw')
    ceiling = (1 << 256) - ((1 << 256) % size)
    counter = 0
    while True:
        payload = json.dumps([domain, int(request_id), counter], sort_keys=True, separators=(',', ':')).encode()
        value = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
        if value < ceiling:
            return value % size
        counter += 1


class ReplaySampler:
    def __init__(self, trace, sequence, legacy_chooser, *, seed=None, method=None, episode=None, replicate=None):
        require(sequence == 'legacy' or (replicate is not None and sequence == f'mc{replicate:04d}'), 'Unknown sequence')
        self.trace, self.sequence, self.legacy_chooser = trace, sequence, legacy_chooser
        self.context = dict(seed=seed, method=method, episode=episode, replicate=replicate)
        if sequence != 'legacy':
            require(all(v is not None for v in self.context.values()), 'MC case identity required')
        self.records = []
        self.cache = {}
        self.reason_pools = {}
        self.row_hashes = {}
        for i, row in enumerate(trace if sequence != 'legacy' else []):
            if row.get('error'):
                continue
            require(math.isfinite(row['e2e_ms']) and row['e2e_ms'] >= 0, 'Invalid empirical latency')
            self.reason_pools.setdefault(row['fallback_reason'], []).append((i, row))

    def reset(self):
        # No RNG stream exists: restarting this case reproduces the same draws.
        self.records = []

    def matched(self, decision):
        key = (decision['fallback_reason'], decision['tokens'])
        if key not in self.cache:
            pool = self.reason_pools.get(key[0], [])
            require(pool, f'Missing exact fallback reason: {key[0]}')
            gap = min(abs(row['tokens']-key[1]) for _, row in pool)
            matched = [(i,row) for i,row in pool if abs(row['tokens']-key[1]) == gap]
            membership = object_hash([i for i,_ in matched])
            domain = {**self.context, 'matched_pool_sha256': membership}
            self.cache[key] = (matched, gap, membership, domain)
        return self.cache[key]

    def choose(self, decision, request_id):
        require(0 <= request_id < 600, 'Request window exhausted')
        if self.sequence == 'legacy':
            delay, ref, gap = self.legacy_chooser(self.trace, decision, request_id, strict=True)
            self.records.append(dict(request=request_id, sequence='legacy', source_request=ref, e2e_ms=delay, token_gap=gap))
            return delay, ref, gap
        matched, gap, membership, domain = self.matched(decision)
        offset = uniform_index(domain, request_id, len(matched))
        index, row = matched[offset]
        if index not in self.row_hashes:
            self.row_hashes[index] = object_hash(row)
        self.records.append({
            'request': request_id, 'sequence': self.sequence, **self.context,
            'matched_pool_sha256': membership, 'matched_pool_size': len(matched), 'pool_index': offset,
            'source_interleaved_row': index, 'source_repeat': row.get('repeat'),
            'source_request': row['request'], 'source_row_sha256': self.row_hashes[index],
            'source_row': row, 'path': row['path'], 'fallback_reason': row['fallback_reason'],
            'tokens': row['tokens'], 'token_gap': gap, 'e2e_ms': row['e2e_ms'],
        })
        return row['e2e_ms'], row['request'], gap

    def preflight(self):
        if self.sequence == 'legacy':
            return {'sequence': 'legacy', 'rule': 'original chooser'}
        path_counts = {}
        expected_reasons = {'normal':None, 'missing_observation':'cold_start_or_missing_observation',
                            'no_predicted_feasible_candidate':'no_predicted_feasible_candidate'}
        for row in self.trace:
            require(row['method'] == self.context['method'], 'Wrong method latency pool')
            require(row['path'] in expected_reasons and row['fallback_reason']==expected_reasons[row['path']], 'Path/reason mismatch')
            path_counts[row['path']] = path_counts.get(row['path'],0)+1
        require(path_counts == {'normal':30000,'missing_observation':3000,'no_predicted_feasible_candidate':3000}, 'Original measured path counts changed')
        plans = []
        # All observed token counts define the pool families. Off-grid queries
        # are matched by the same deterministic minimum-gap rule at runtime.
        keys = sorted({(r['fallback_reason'],r['tokens']) for r in self.trace if not r.get('error')}, key=str)
        for reason, tokens in keys:
            matched, gap, membership, domain = self.matched(dict(fallback_reason=reason,tokens=tokens))
            require(gap == 0, 'Observed token query must exactly match')
            plans.append(dict(fallback_reason=reason, tokens=tokens, size=len(matched),
                              membership_sha256=membership,
                              source_indices=[matched[uniform_index(domain,i,len(matched))][0] for i in range(600)]))
        plan = {'sequence':self.sequence, **self.context, 'path_counts':path_counts,
                'pool_families':plans, 'rule':'stateless IID SHA256 counters'}
        plan['draw_plan_sha256'] = object_hash(plan)
        return plan
