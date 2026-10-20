"""Targeted corruption checks for the timing evidence validator."""
import copy
import gzip
import json
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
api = runpy.run_path(str(root/'verify_mc_audit.py'))
with gzip.open(root/'cases.jsonl.gz', 'rt', encoding='utf-8') as stream:
    case = json.loads(next(stream))
groups = api['pair_groups'](root/'actual_schedules.csv.xz')
_, sides = next(groups)
groups.close()
original = sides[0]
metrics = json.loads(case['metrics_raw'])
api['check_schedule'](original, metrics, 'unaltered fixture')
results = {'unaltered_case_passes': True}

def fails(name, mutate):
    rows = copy.deepcopy(original)
    mutate(rows)
    try:
        api['check_schedule'](rows, metrics, name)
    except ValueError:
        results[name] = True
    else:
        raise AssertionError('Corruption was not rejected: '+name)

def change_ready(rows):
    row = next(r for r in rows if r['event']=='request')
    row['ready_ns'] = str(int(row['ready_ns'])+1)

def change_expiry(rows):
    row = next(r for r in rows if r['event']=='request')
    row['expired'] = '1' if row['expired']=='0' else '0'

def remove_skip(rows):
    i = next(i for i,r in enumerate(rows) if r['event']=='busy_skip')
    rows.pop(i)
    for j,row in enumerate(rows):
        row['event_index'] = str(j)

def shift_application(rows):
    row = next(r for r in rows if r['event']=='apply')
    row['time_ns'] = str(int(row['time_ns'])+1)

fails('one_nanosecond_ready_change_rejected', change_ready)
fails('flipped_expiry_rejected', change_expiry)
fails('missing_busy_skip_rejected', remove_skip)
fails('one_nanosecond_application_change_rejected', shift_application)
out=root/'outputs'
out.mkdir(exist_ok=True)
(out/'checker_failure_tests.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
