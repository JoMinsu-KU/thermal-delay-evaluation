"""Check reported delay settings, alpha selection, and saved hold-curve input."""
from pathlib import Path
import csv, json
import numpy as np

P=Path(__file__).resolve().parents[1]
R=P/'data/minor_revision'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8')))
refs=read(R/'alpha_validation_references.json')
def key(rows):
    actual={(r['delay'],r['episode']):r for r in rows}; adverse=0; ratios=[]
    for reference in refs.values():
        expected={(r['delay'],r['episode']):r for r in reference}
        assert actual.keys()==expected.keys()
        for k,a in actual.items():
            b=expected[k]
            adverse+=int(a['new_violation']>b['new_violation'] or a['violation_s']>b['violation_s']+1e-8 or a['inference_failures']>b['inference_failures'])
        for delay in sorted({r['delay'] for r in rows}):
            for stratum in ['all','normal']:
                keys=[k for k,r in actual.items() if r['delay']==delay and (stratum=='all' or r['stratum']==stratum)]
                ratios.append(float(np.mean([actual[k]['episode_cost'] for k in keys])/max(np.mean([expected[k]['episode_cost'] for k in keys]),1e-8)))
    return [adverse,max(ratios),float(np.mean(ratios))]
alphas={}
for arm in ['shared_physics','small_physics']:
    record=read(R/(arm+'_alpha_selection.json'))
    for c in record['candidates']:
        assert np.allclose(key(c['rows']),c['key'],atol=1e-14,rtol=0)
    selected=min(record['candidates'],key=lambda c:(*key(c['rows']),c['alpha']))['alpha']
    assert selected==record['selected_alpha'];alphas[arm]=selected
replay=csvread(P/'data/replay/episode_metrics.csv');assert len(replay)==1128
assert all(float(r['injected_compute_ms'])==float(r['injected_apply_ms'])==0 for r in replay)
primary=csvread(P/'data/revision/cost_components.csv')
episodes={r['episode'] for r in primary};count=0
for path in (P/'data/primary').glob('*.json'):
    if not path.name.startswith(('zero_','delayed_')):continue
    saved=read(path)
    # Original per-method metric files are lists of test and OOD records.
    for row in saved:
        if row['episode'] in episodes:
            assert row['injected_compute_ms']==row['injected_apply_ms']==0
            count+=1
assert count==240,count
hold=csvread(R/'hold_temperature.csv')
assert len(hold)==601
assert [int(r['time_ns']) for r in hold]==list(range(0,60_000_000_001,100_000_000))
assert all(r['method']=='fv_mpc' and r['mode']=='hold' for r in hold)
assert all(float(r['min_temperature_c'])<=float(r['mean_temperature_c'])<=float(r['max_temperature_c']) for r in hold)
original=csvread(P/'data/revision/representative_temperature.csv')
zero=[r for r in original if r['method']=='fv_mpc' and r['mode']=='zero']
assert [r['time_ns'] for r in hold]==[r['time_ns'] for r in zero]
assert [r['goal_c'] for r in hold]==[r['goal_c'] for r in zero]
for k in ['mean_temperature_c','min_temperature_c','max_temperature_c']:
    assert hold[0][k]==zero[0][k]
receipt=read(R/'method_evidence.json')['hold_temperature']
saved=next(r for r in replay if r['key']==receipt['case'])
for k,v in receipt['metrics'].items():assert np.isclose(float(saved[k]),v,rtol=0,atol=1e-12)
out={'passed':True,'original_normal_cases_with_zero_injected_delays':count,
     'returned_cases_with_zero_injected_delays':len(replay),'recomputed_selected_alpha':alphas,
     'hold_curve_samples':len(hold),'raw_trace_and_seal_audit_repeated':False,
     'scope':'Saved compact records only. The raw-file seal and timing checks are author audit receipts.'}
(P/'outputs/method_detail_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print('PASS: delay values, six alpha candidates, and 601 saved hold temperature samples')
