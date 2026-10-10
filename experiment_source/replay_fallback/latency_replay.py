"""Path/token-matched chronological blocks from the archived timing trials."""
from validation_io import require, object_hash
from plan import SEQUENCES


class ReplaySampler:
    def __init__(self, trace, sequence, legacy_chooser):
        require(sequence == 'legacy' or sequence in SEQUENCES, 'Unknown frozen replay setting')
        self.trace, self.sequence, self.legacy_chooser = trace, sequence, legacy_chooser
        self.records = []
        self.cache = {}
        if sequence != 'legacy':
            self.repeat = int(sequence[1]); self.block = int(sequence[-1])
            require(len(trace) % 3 == 0, 'Expected three equal archived measurement trials')
            self.trial = trace[self.repeat::3]

    def reset(self):
        self.records = []

    def choose(self, decision, request_id):
        require(0 <= request_id < 600, 'Replay window exhausted; do not wrap or enlarge it')
        if self.sequence == 'legacy':
            delay, ref, gap = self.legacy_chooser(self.trace, decision, request_id, strict=True)
            self.records.append({'request': request_id, 'sequence': 'legacy', 'source_request': ref,
                                 'e2e_ms': delay, 'token_gap': gap})
            return delay, ref, gap
        key = (decision['fallback_reason'], decision['tokens'])
        if key not in self.cache:
            pool = [(i, row) for i, row in enumerate(self.trial)
                    if row['fallback_reason'] == key[0] and not row.get('error')]
            require(pool, f'Missing recorded fallback path {key[0]}')
            gap = min(abs(row['tokens'] - key[1]) for _, row in pool)
            matched = [(i, row) for i, row in pool if abs(row['tokens'] - key[1]) == gap]
            start = (0, 5000)[self.block] if key[0] is None else (0, 400)[self.block]
            require(len(matched) >= start + 600, f'Insufficient chronological window for {self.sequence}, path/token={key}: {len(matched)}')
            self.cache[key] = (matched, start, gap)
        matched, start, gap = self.cache[key]
        index, row = matched[start + request_id]
        self.records.append({'request': request_id, 'sequence': self.sequence, 'repeat': self.repeat,
                             'block': self.block, 'pool_start': start, 'pool_index': start + request_id,
                             'source_trial_row': index, 'source_request': row['request'],
                             'source_row_sha256': object_hash(row), 'path': row['path'],
                             'source_row': row,
                             'fallback_reason': row['fallback_reason'], 'tokens': row['tokens'],
                             'token_gap': gap, 'e2e_ms': row['e2e_ms']})
        return row['e2e_ms'], row['request'], gap

    def preflight(self):
        if self.sequence == 'legacy':
            return {'sequence': self.sequence, 'rule': 'original chooser'}
        for reason, tokens in sorted({(r['fallback_reason'], r['tokens']) for r in self.trial if not r.get('error')}, key=str):
            self.choose({'fallback_reason': reason, 'tokens': tokens}, 599)
        self.reset()
        return {'sequence': self.sequence, 'repeat': self.repeat, 'matched_windows': len(self.cache), 'window_requests': 600}
