"""Export saved FV schedules/draws; never load a model or simulate a plant."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import lzma
from pathlib import Path
import time
import zipfile

SCHEDULE_FIELDS = ['case_id', 'event_index', 'event', 'request', 'snapshot_ns',
    'ready_ns', 'compute_ready_ns', 'deadline_ns', 'expired', 'time_ns', 'delay_ms',
    'injected_compute_ms', 'injected_apply_ms', 'trace_request', 'trace_token_gap',
    'fallback_reason', 'tokens']
DRAW_FIELDS = ['case_id', 'request', 'pool_family_id', 'pool_index',
    'source_interleaved_row', 'source_repeat', 'source_request', 'token_gap', 'e2e_ms']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def objhash(obj):
    return sha(json.dumps(obj, sort_keys=True, allow_nan=False).encode())

def check(ok, message):
    if not ok:
        raise ValueError(message)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def cell(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)

def schedule_projection(case_id, trace):
    result = []
    for i, record in enumerate(trace):
        row = [str(case_id), str(i)]
        for key in SCHEDULE_FIELDS[2:]:
            value = record.get(key)
            if key == 'fallback_reason' and key in record:
                row.append(json.dumps(value, separators=(',', ':')))
            else:
                row.append('' if key not in record else cell(value))
        result.append(row)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print('Hashing original ZIP...', flush=True)
    archive_hash = filehash(args.archive)
    cases = []
    source_members = []
    family_ids = {}
    families = []
    source_hashes = {}
    counts = {'cases': 0, 'schedule_events': 0, 'consumed_draws': 0}
    with zipfile.ZipFile(args.archive) as z:
        root = z.namelist()[0].split('/')[0] + '/'
        def read(name):
            return z.read(root + name)
        def preserve(name, target):
            raw = read(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            source_members.append({'archive_member': root + name,
                                   'output': target.relative_to(dest).as_posix(),
                                   'sha256': sha(raw), 'bytes': len(raw)})
            return raw
        protocol_raw = read('protocol.json')
        protocol = json.loads(protocol_raw)
        (dest / 'protocol.json.gz').write_bytes(gzip.compress(protocol_raw, mtime=0))
        source_members.append({'archive_member': root + 'protocol.json',
            'output': 'protocol.json.gz', 'uncompressed_sha256': sha(protocol_raw),
            'uncompressed_bytes': len(protocol_raw)})
        for name, digest in protocol['harness_sha256'].items():
            data = preserve('harness/' + name, dest / 'archived_source' / name)
            check(sha(data) == digest, 'Protocol harness seal: ' + name)
        pool_raw = preserve('timing_pools/fv_mpc/rows.json.gz', dest / 'fv_timing_pool.json.gz')
        pool_manifest_raw = preserve('timing_pools/fv_mpc/manifest.json', dest / 'fv_pool_manifest.json')
        pool_complete = preserve('timing_pools/fv_mpc/COMPLETE', dest / 'fv_pool_COMPLETE')
        pool_manifest = json.loads(pool_manifest_raw)
        check(sha(pool_manifest_raw) == pool_complete.decode().strip(), 'Pool manifest seal')
        check(sha(pool_raw) == pool_manifest['files']['rows.json.gz'], 'Pool bytes seal')
        pool = json.loads(gzip.decompress(pool_raw))
        check(len(pool) == 36000, 'FV pool size')
        check(objhash(pool) == pool_manifest['metadata']['pool_object_sha256'], 'Pool object seal')
        pool_row_hashes = [objhash(r) for r in pool]
        members = {}
        for name in z.namelist():
            if '/cases/monte_carlo_fv_mpc_' in name and name.endswith('/condition.json'):
                folder = name.rsplit('/', 1)[0]
                cond = json.loads(z.read(name))
                key = (cond['replicate'], cond['episode'], cond['fallback_policy'])
                check(key not in members, 'Duplicate FV case')
                members[key] = folder
        episodes = protocol['design']['normal_episodes']
        expected = {(r, e, p) for r in range(64) for e in episodes for p in ('zero', 'hold')}
        check(set(members) == expected and len(episodes) == 24, 'Expected case grid')
        with lzma.open(dest / 'actual_schedules.csv.xz', 'wt', encoding='utf-8', newline='', preset=6) as sf, \
             lzma.open(dest / 'actual_draws.csv.xz', 'wt', encoding='utf-8', newline='', preset=6) as df:
            sw, dw = csv.writer(sf, lineterminator='\n'), csv.writer(df, lineterminator='\n')
            sw.writerow(SCHEDULE_FIELDS)
            dw.writerow(DRAW_FIELDS)
            for pair_id, (replicate, episode) in enumerate(itertools.product(range(64), episodes)):
                pair_schedules, pair_draws = [], []
                for policy in ('zero', 'hold'):
                    folder = members[replicate, episode, policy]
                    raw = {f: z.read(folder + '/' + f) for f in (
                        'condition.json', 'manifest.json', 'COMPLETE', 'trace.jsonl.gz',
                        'replay_rows.json.gz', 'draw_plan.json', 'metrics.json')}
                    manifest = json.loads(raw['manifest.json'])
                    check(sha(raw['manifest.json']) == raw['COMPLETE'].decode().strip(), folder + ': manifest')
                    for f, data in raw.items():
                        if f not in ('manifest.json', 'COMPLETE'):
                            check(sha(data) == manifest['files'][f], folder + ': ' + f)
                    check(manifest['metadata']['protocol_sha256'] == sha(protocol_raw), 'Case protocol seal')
                    condition = json.loads(raw['condition.json'])
                    trace = [json.loads(s) for s in gzip.decompress(raw['trace.jsonl.gz']).splitlines()]
                    records = json.loads(gzip.decompress(raw['replay_rows.json.gz']))
                    plan = json.loads(raw['draw_plan.json'])
                    context = {k: plan[k] for k in ('sequence', 'seed', 'method', 'episode', 'replicate')}
                    check(context == {'sequence': f'mc{replicate:04d}', 'seed': 20261013,
                        'method': 'fv_mpc', 'episode': episode, 'replicate': replicate}, 'Draw context')
                    request_rows = [r for r in trace if r['event'] == 'request']
                    check(len(records) == len(request_rows), 'Trace/draw count')
                    case_id = len(cases)
                    schedule_rows = schedule_projection(case_id, trace)
                    draw_rows = []
                    for position, (record, request) in enumerate(zip(records, request_rows)):
                        check(record['request'] == request['request'] == position, 'Draw/request order')
                        check(all(record[k] == v for k, v in context.items()), 'Saved draw context')
                        index = record['source_interleaved_row']
                        check(record['source_row'] == pool[index], 'Actual embedded source row')
                        check(record['source_row_sha256'] == pool_row_hashes[index], 'Actual saved source-row hash')
                        source_hashes[index] = record['source_row_sha256']
                        family = {k: record[k] for k in ('matched_pool_sha256', 'matched_pool_size',
                            'path', 'fallback_reason', 'tokens')}
                        signature = json.dumps(family, sort_keys=True)
                        if signature not in family_ids:
                            family_ids[signature] = len(families)
                            families.append(family)
                        family_id = family_ids[signature]
                        row = [str(case_id), str(record['request']), str(family_id)]
                        row.extend(cell(record[k]) for k in DRAW_FIELDS[3:])
                        draw_rows.append(row)
                    metrics = json.loads(raw['metrics.json'])
                    entry = {'case_id': case_id, 'pair_id': pair_id, 'replicate': replicate,
                        'episode': episode, 'policy': policy, 'archive_folder': folder,
                        'condition_raw': raw['condition.json'].decode(),
                        'manifest_raw': raw['manifest.json'].decode(),
                        'complete_raw': raw['COMPLETE'].decode(),
                        'metrics_raw': raw['metrics.json'].decode(), 'sampler_context': context,
                        'draw_plan_sha256': sha(raw['draw_plan.json']),
                        'draw_plan_object_sha256': plan['draw_plan_sha256'],
                        'trace_gzip_sha256': sha(raw['trace.jsonl.gz']),
                        'replay_rows_gzip_sha256': sha(raw['replay_rows.json.gz']),
                        'exported_schedule_rows': len(schedule_rows), 'exported_draw_rows': len(draw_rows),
                        'schedule_projection_sha256': objhash(schedule_rows),
                        'draw_projection_sha256': objhash(draw_rows)}
                    cases.append(entry)
                    pair_schedules.append(schedule_rows)
                    pair_draws.append(draw_rows)
                    counts['cases'] += 1
                    counts['schedule_events'] += len(schedule_rows)
                    counts['consumed_draws'] += len(draw_rows)
                # Adjacent arms compress well. Each row is independently read from its own original log.
                for group in itertools.zip_longest(*pair_schedules):
                    for row in group:
                        if row is not None:
                            sw.writerow(row)
                for group in itertools.zip_longest(*pair_draws):
                    for row in group:
                        if row is not None:
                            dw.writerow(row)
                if (pair_id + 1) % 96 == 0:
                    print(f'{pair_id + 1}/1536 pairs exported; {time.monotonic()-started:.1f}s', flush=True)
        with gzip.open(dest / 'cases.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as stream:
            for entry in cases:
                stream.write(json.dumps(entry, separators=(',', ':')) + '\n')
        with gzip.open(dest / 'saved_source_row_hashes.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['source_interleaved_row', 'source_row_sha256'])
            writer.writerows(sorted(source_hashes.items()))
        dump(dest / 'pool_families.json', families)
        dump(dest / 'schema.json', {'schema': 'saved-fv-monte-carlo-schedules-v1',
            'schedule_columns': SCHEDULE_FIELDS, 'draw_columns': DRAW_FIELDS,
            'episode_order': episodes, 'replicates': 64, 'policies': ['zero', 'hold'],
            'schedule_empty_cell': 'Field absent in original event; not a zero.',
            'fallback_reason_encoding': 'JSON: null is distinct from an absent field.',
            'dictionary_normalization': 'Each saved draw context is in cases.sampler_context; family hash/size/path/reason/tokens are in pool_families; saved source-row hashes are in saved_source_row_hashes.csv.gz. All repeated values were checked before factoring.'})
        dump(dest / 'export_provenance.json', {'archive_name': args.archive.name,
            'archive_bytes': args.archive.stat().st_size, 'archive_sha256': archive_hash,
            'archive_root': root, 'protocol_sha256': sha(protocol_raw),
            'source_members': source_members, 'counts': counts,
            'saved_source_row_hashes': len(source_hashes), 'pool_rows': len(pool),
            'elapsed_seconds': time.monotonic()-started,
            'scope': 'All 3072 FV MC cases; only listed schedule/sampler/metadata members read. Full fields, quadrature, models, and all non-FV case traces are omitted. Every selected member passed ZIP CRC and original case SHA-256 checks. Raw-source reproduction requires the original ZIP.'})
    inputs = ['actual_schedules.csv.xz', 'actual_draws.csv.xz', 'cases.jsonl.gz',
        'saved_source_row_hashes.csv.gz', 'pool_families.json', 'schema.json',
        'export_provenance.json', 'fv_timing_pool.json.gz', 'fv_pool_manifest.json',
        'fv_pool_COMPLETE', 'protocol.json.gz']
    paths = [dest / f for f in inputs] + sorted((dest / 'archived_source').glob('*'))
    dump(dest / 'INPUT_SHA256.json', {p.relative_to(dest).as_posix(): filehash(p) for p in paths})
    print(json.dumps({'status': 'EXPORTED', **counts, 'bytes': sum(p.stat().st_size for p in paths),
        'elapsed_seconds': time.monotonic()-started}), flush=True)

if __name__ == '__main__':
    main()
