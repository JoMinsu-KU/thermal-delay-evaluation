"""Verify exported FV schedules and every consumed latency draw offline.

Python standard library only. No archived simulation module is imported.
Optionally compare the compact projection directly with the original ZIP.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import lzma
import math
from pathlib import Path
import time
import zipfile

def require(value, message):
    if not value:
        raise ValueError(message)

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest_object(value):
    return digest_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())

def digest_file(path):
    state = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                return state.hexdigest()
            state.update(block)

def independent_draw(domain, request, count):
    # Independently implemented from the documented 256-bit rejection rule.
    limit = 2 ** 256 - (2 ** 256 % count)
    attempt = 0
    while True:
        encoded = json.dumps([domain, request, attempt], sort_keys=True, separators=(',', ':')).encode()
        integer = int(hashlib.sha256(encoded).hexdigest(), 16)
        if integer < limit:
            return integer % count
        attempt += 1

def pair_groups(path):
    with lzma.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        for pair_id, rows in itertools.groupby(reader, lambda r: int(r['case_id']) // 2):
            sides = collections.defaultdict(list)
            for row in rows:
                sides[int(row['case_id'])].append(row)
            yield pair_id, sides

def check_schedule(rows, metrics, context):
    requests = []
    applications = {}
    ticks = {}
    skip_rows = []
    previous_time = -1
    for ordinal, row in enumerate(rows):
        require(int(row['event_index']) == ordinal, context + ': trace order')
        event = row['event']
        if event == 'request':
            request = int(row['request'])
            now = int(row['snapshot_ns'])
            ready = int(row['ready_ns'])
            deadline = int(row['deadline_ns'])
            require(request == len(requests), context + ': request index')
            require(now % 100_000_000 == 0 and 0 <= now < 60_000_000_000, context + ': request clock')
            require(deadline == now + 100_000_000, context + ': deadline clock')
            require(int(row['compute_ready_ns']) == ready == now + math.ceil(float(row['delay_ms']) * 1e6), context + ': ready clock')
            require(int(row['expired']) == int(ready > deadline), context + ': strict expiry')
            require(float(row['injected_compute_ms']) == float(row['injected_apply_ms']) == 0.0, context + ': injected delay')
            require(now not in ticks, context + ': duplicate tick')
            ticks[now] = row
            requests.append(row)
        elif event in ('apply', 'deadline_fallback'):
            now, request = int(row['time_ns']), int(row['request'])
            require(request not in applications, context + ': repeated application')
            require(request < len(requests), context + ': application before request')
            applications[request] = row
        elif event == 'busy_skip':
            now = int(row['time_ns'])
            require(now not in ticks, context + ': duplicate skipped tick')
            ticks[now] = row
            skip_rows.append(row)
        else:
            raise ValueError(context + ': unknown event ' + event)
        require(now >= previous_time, context + ': event time order')
        previous_time = now
    require(set(ticks) == set(range(0, 60_000_000_000, 100_000_000)), context + ': 600-slot coverage')
    busy_until = 0
    for now in sorted(ticks):
        row = ticks[now]
        if now < busy_until:
            require(row['event'] == 'busy_skip' and int(row['ready_ns']) == busy_until, context + ': busy skip')
        else:
            require(row['event'] == 'request', context + ': admissible request')
            busy_until = int(row['compute_ready_ns'])
    require(len(applications) == len(requests), context + ': application count')
    for i, row in enumerate(requests):
        ready, deadline = int(row['ready_ns']), int(row['deadline_ns'])
        app = applications[i]
        require(int(app['time_ns']) == min(ready, deadline), context + ': event clock')
        require(app['event'] == ('deadline_fallback' if ready > deadline else 'apply'), context + ': event kind')
        require(int(app['event_index']) > int(row['event_index']), context + ': request/application order')
        if i + 1 < len(requests) and int(app['time_ns']) == int(requests[i+1]['snapshot_ns']):
            require(int(app['event_index']) < int(requests[i+1]['event_index']), context + ': simultaneous-event priority')
    misses = sum(int(r['expired']) for r in requests)
    require(len(requests) == metrics['requests'], context + ': archived metric request count')
    require(misses == metrics['missed_deadlines'], context + ': archived metric deadline count')
    require(len(skip_rows) == metrics['busy_skipped_ticks'], context + ': archived metric skip count')
    return requests, misses, len(skip_rows)

def raw_check(z, case, schedule, draws, columns, families, saved_hashes, pool):
    base = case['archive_folder'] + '/'
    manifest = json.loads(case['manifest_raw'])
    require(z.read(base+'manifest.json').decode() == case['manifest_raw'], 'Raw manifest identity')
    require(z.read(base+'COMPLETE').decode() == case['complete_raw'], 'Raw COMPLETE identity')
    for name, key in [('condition.json', 'condition_raw'), ('metrics.json', 'metrics_raw')]:
        require(z.read(base+name).decode() == case[key], 'Raw metadata identity: '+name)
    tr = z.read(base+'trace.jsonl.gz')
    dr = z.read(base+'replay_rows.json.gz')
    require(digest_bytes(tr) == manifest['files']['trace.jsonl.gz'], 'Raw trace seal')
    require(digest_bytes(dr) == manifest['files']['replay_rows.json.gz'], 'Raw draw seal')
    original_events = [json.loads(line) for line in gzip.decompress(tr).splitlines()]
    original_draws = json.loads(gzip.decompress(dr))
    require(len(original_events) == len(schedule) and len(original_draws) == len(draws), 'Raw/projection counts')
    for index, (raw, row) in enumerate(zip(original_events, schedule)):
        require(int(row['event_index']) == index, 'Raw event ordinal')
        for field in columns[2:]:
            if field not in raw:
                require(row[field] == '', 'Raw absent field: '+field)
            elif field == 'fallback_reason':
                require(json.loads(row[field]) == raw[field], 'Raw reason')
            elif isinstance(raw[field], bool):
                require(row[field] == ('1' if raw[field] else '0'), 'Raw Boolean')
            else:
                require(row[field] == str(raw[field]), 'Raw event field: '+field)
    for raw, row in zip(original_draws, draws):
        family = families[int(row['pool_family_id'])]
        require(all(raw[k] == v for k, v in case['sampler_context'].items()), 'Raw draw context')
        require(all(raw[k] == v for k, v in family.items()), 'Raw draw family')
        for field in ('request', 'pool_index', 'source_interleaved_row', 'source_repeat', 'source_request', 'token_gap'):
            require(raw[field] == int(row[field]), 'Raw draw field: '+field)
        require(raw['e2e_ms'] == float(row['e2e_ms']), 'Raw draw latency')
        source = int(row['source_interleaved_row'])
        require(raw['source_row_sha256'] == saved_hashes[source], 'Raw saved hash')
        require(raw['source_row'] == pool[source], 'Raw embedded source row')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--raw-archive', type=Path)
    ap.add_argument('--output', type=Path, help='Default: outputs/offline or outputs/raw_archive')
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.output or root / 'outputs' / ('raw_archive' if args.raw_archive else 'offline')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    load = lambda f: json.loads((root/f).read_text(encoding='utf-8'))
    for name, expected in load('INPUT_SHA256.json').items():
        path = (root/name).resolve()
        require(path.is_relative_to(root), 'Unsafe input path')
        require(digest_file(path) == expected, 'Input hash: '+name)
    schema = load('schema.json')
    provenance = load('export_provenance.json')
    protocol_bytes = gzip.decompress((root/'protocol.json.gz').read_bytes())
    protocol = json.loads(protocol_bytes)
    require(digest_bytes(protocol_bytes) == provenance['protocol_sha256'], 'Protocol provenance')
    withheld = provenance.get('withheld_harness_metadata', [])
    withheld_by_name = {r['name']: r for r in withheld}
    require(len(withheld_by_name) == len(withheld), 'Duplicate withheld metadata entry')
    require(not withheld or set(withheld_by_name) == {'LICENSE', 'DATA_LICENSE'}, 'Only the explicitly listed attribution metadata can be withheld')
    for name, expected in protocol['harness_sha256'].items():
        if name in withheld_by_name:
            require(withheld_by_name[name]['original_sha256'] == expected, 'Withheld metadata original hash')
            require(not (root/'archived_source'/name).exists(), 'Anonymous metadata omission must be explicit')
        else:
            require(digest_file(root/'archived_source'/name) == expected, 'Original protocol source hash: '+name)
    pm_raw = (root/'fv_pool_manifest.json').read_bytes()
    pm = json.loads(pm_raw)
    require(digest_bytes(pm_raw) == (root/'fv_pool_COMPLETE').read_text().strip(), 'Original pool seal')
    require(digest_file(root/'fv_timing_pool.json.gz') == pm['files']['rows.json.gz'], 'Original pool file hash')
    with gzip.open(root/'fv_timing_pool.json.gz', 'rt', encoding='utf-8') as stream:
        pool = json.load(stream)
    require(len(pool) == 36000 and digest_object(pool) == pm['metadata']['pool_object_sha256'], 'Original pool object')
    with gzip.open(root/'saved_source_row_hashes.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        saved_hashes = {int(r['source_interleaved_row']): r['source_row_sha256'] for r in csv.DictReader(stream)}
    for i, expected in saved_hashes.items():
        require(digest_object(pool[i]) == expected, 'Saved source-row hash')
    families = load('pool_families.json')
    with gzip.open(root/'cases.jsonl.gz', 'rt', encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream]
    require(len(cases) == 3072, 'Case count')
    reason_indices = collections.defaultdict(list)
    for i, row in enumerate(pool):
        if not row.get('error'):
            reason_indices[row['fallback_reason']].append(i)
    matching_cache = {}
    def match(reason, query_tokens):
        key = (reason, query_tokens)
        if key not in matching_cache:
            candidates = reason_indices[reason]
            gap = min(abs(pool[i]['tokens']-query_tokens) for i in candidates)
            indexes = [i for i in candidates if abs(pool[i]['tokens']-query_tokens) == gap]
            matching_cache[key] = indexes, gap, digest_object(indexes)
        return matching_cache[key]
    z = None
    if args.raw_archive:
        print('Hashing raw archive for direct projection comparison...', flush=True)
        require(digest_file(args.raw_archive) == provenance['archive_sha256'], 'Original ZIP identity')
        z = zipfile.ZipFile(args.raw_archive)
    reports = []
    totals = collections.Counter()
    try:
        schedule_groups = pair_groups(root/'actual_schedules.csv.xz')
        draw_groups = pair_groups(root/'actual_draws.csv.xz')
        for expected_pair, items in enumerate(itertools.zip_longest(schedule_groups, draw_groups)):
            require(None not in items, 'Schedule/draw group coverage')
            (pair_id, schedule_sides), (draw_pair, draw_sides) = items
            require(pair_id == draw_pair == expected_pair, 'Pair ordering')
            require(set(schedule_sides) == set(draw_sides) == {2*pair_id, 2*pair_id+1}, 'Both actual arms')
            results = []
            for case_id in (2*pair_id, 2*pair_id+1):
                case = cases[case_id]
                ctx = case['sampler_context']
                expected_episode = schema['episode_order'][pair_id % 24]
                expected_policy = ('zero', 'hold')[case_id % 2]
                require(case['case_id'] == case_id and case['pair_id'] == pair_id, 'Case identity')
                require(case['episode'] == expected_episode and case['replicate'] == pair_id//24 and case['policy'] == expected_policy, 'Expected grid')
                condition = json.loads(case['condition_raw'])
                manifest = json.loads(case['manifest_raw'])
                metrics = json.loads(case['metrics_raw'])
                require(digest_bytes(case['manifest_raw'].encode()) == case['complete_raw'].strip(), 'Case original manifest seal')
                require(manifest['metadata']['protocol_sha256'] == digest_bytes(protocol_bytes), 'Case protocol binding')
                for filename, key in [('condition.json', 'condition_raw'), ('metrics.json', 'metrics_raw')]:
                    require(digest_bytes(case[key].encode()) == manifest['files'][filename], 'Case original file seal')
                require(condition['fallback_policy'] == expected_policy and condition['episode'] == expected_episode and condition['replicate'] == pair_id//24, 'Case original condition')
                require(ctx == {'seed': 20261013, 'method': 'fv_mpc', 'episode': expected_episode,
                    'replicate': pair_id//24, 'sequence': f'mc{pair_id//24:04d}'}, 'Sampler context')
                schedules, draws = schedule_sides[case_id], draw_sides[case_id]
                for rows, columns, key in [(schedules, schema['schedule_columns'], 'schedule_projection_sha256'),
                                           (draws, schema['draw_columns'], 'draw_projection_sha256')]:
                    require(digest_object([[r[k] for k in columns] for r in rows]) == case[key], 'Projection digest')
                requests, misses, skips = check_schedule(schedules, metrics, condition['key'])
                require(len(draws) == len(requests), 'Every actual request has a saved draw')
                for request, draw in zip(requests, draws):
                    ordinal = int(draw['request'])
                    require(ordinal == int(request['request']), 'Draw request join')
                    family = families[int(draw['pool_family_id'])]
                    reason = json.loads(request['fallback_reason'])
                    indices, gap, membership = match(reason, int(request['tokens']))
                    require(family['matched_pool_sha256'] == membership and family['matched_pool_size'] == len(indices), 'Matched pool identity')
                    domain = {k: ctx[k] for k in ('seed', 'method', 'episode', 'replicate')}
                    domain['matched_pool_sha256'] = membership
                    offset = independent_draw(domain, ordinal, len(indices))
                    require(offset == int(draw['pool_index']), 'Independent uniform offset')
                    source = int(draw['source_interleaved_row'])
                    require(indices[offset] == source, 'Selected original row index')
                    row = pool[source]
                    require(source in saved_hashes, 'Actual saved source-row hash retained')
                    require(family['path'] == row['path'] and family['fallback_reason'] == row['fallback_reason'] and family['tokens'] == row['tokens'], 'Recorded pool family/source row')
                    require(int(draw['source_repeat']) == row['repeat'], 'Recorded source repeat')
                    require(int(draw['source_request']) == row['request'] == int(request['trace_request']), 'Recorded source request')
                    require(float(draw['e2e_ms']) == row['e2e_ms'] == float(request['delay_ms']), 'Actual measured draw latency')
                    require(int(draw['token_gap']) == gap == int(request['trace_token_gap']), 'Nearest token match')
                if z is not None:
                    raw_check(z, case, schedules, draws, schema['schedule_columns'], families, saved_hashes, pool)
                totals['cases'] += 1
                totals['schedule_events'] += len(schedules)
                totals['draws_independently_recomputed'] += len(draws)
                totals[expected_policy+'_requests'] += len(requests)
                totals[expected_policy+'_expired'] += misses
                totals[expected_policy+'_busy_skips'] += skips
                results.append((len(requests), misses, skips))
            clocks = ['event_index', 'event', 'request', 'snapshot_ns', 'ready_ns', 'compute_ready_ns',
                'deadline_ns', 'expired', 'time_ns', 'delay_ms', 'injected_compute_ms', 'injected_apply_ms']
            zero, hold = schedule_sides[2*pair_id], schedule_sides[2*pair_id+1]
            require(len(zero) == len(hold), 'Paired event counts')
            for a, b in zip(zero, hold):
                require(all(a[k] == b[k] for k in clocks), 'Paired actual schedule mismatch')
            require([[r[k] for k in schema['draw_columns'][1:]] for r in draw_sides[2*pair_id]] ==
                    [[r[k] for k in schema['draw_columns'][1:]] for r in draw_sides[2*pair_id+1]], 'Paired actual consumed draws')
            reports.append({'pair_id': pair_id, 'replicate': pair_id//24, 'episode': expected_episode,
                'zero_requests': results[0][0], 'hold_requests': results[1][0],
                'zero_expired': results[0][1], 'hold_expired': results[1][1],
                'zero_busy_skips': results[0][2], 'hold_busy_skips': results[1][2],
                'actual_schedules_equal': True, 'actual_draws_equal': True,
                'all_draws_recomputed': True, 'raw_projection_checked': z is not None})
            if (pair_id+1) % 192 == 0:
                print(f'{pair_id+1}/1536 pairs verified; {time.monotonic()-started:.1f}s', flush=True)
        require(len(reports) == 1536, 'Full pair coverage')
    finally:
        if z is not None:
            z.close()
    with (out/'paired_schedule_checks.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(reports)
    summary = {'status': 'PASS', 'pairs': len(reports), 'replicates': 64, 'episodes': 24,
        **dict(totals), 'raw_projection_comparison': args.raw_archive is not None,
        'all_actual_schedule_pairs_equal': True, 'all_actual_consumed_draw_pairs_equal': True,
        'sampler_and_scheduler_protocol_hashes_match': True,
        'distributed_harness_files_checked': len(protocol['harness_sha256'])-len(withheld),
        'explicitly_withheld_attribution_metadata': sorted(withheld_by_name),
        'elapsed_seconds': round(time.monotonic()-started, 3),
        'scope': 'Observed FV schedule events and actual sampled latency records. No model inference, plant dynamics, costs, command values, thermal fields, or new Monte Carlo runs are recomputed.'}
    (out/'verification_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
