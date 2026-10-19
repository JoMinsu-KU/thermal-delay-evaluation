"""Portable saved-score-only reanalysis; place beside saved_score_vectors.jsonl.gz.

Usage: python reproduce_neartie.py [output-directory]
No inference, simulation, learned weights, or original absolute paths required.

Writes near_tie_summary.csv, threshold_fractions.csv, and
episode_threshold_fractions.csv. The two threshold tables are compared with
their bundled reference CSVs when those references are present.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv,gzip,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'reproduced'
if OUT.resolve()==HERE:
    raise ValueError('Choose an output directory separate from the bundled inputs.')
OUT.mkdir(parents=True,exist_ok=True)
GRID=[0.,1e-4,1e-3,.01,.1,1.]
groups=defaultdict(list); counts=defaultdict(Counter)
episodes=defaultdict(lambda:defaultdict(list))
with gzip.open(HERE/'saved_score_vectors.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line); key=(r['method'],r['mode']);counts[key]['requests']+=1
        if r['scores'] is None:
            assert r['request']==0 and r['fallback_reason']=='cold_start_or_missing_observation'
            counts[key]['unscored']+=1;continue
        s=np.array(r['scores'],float); v=np.array(r['valid_mask'],bool);a=np.array(r['predicted_safe_mask'],bool)
        assert np.isfinite(s).all() and not (a&~v).any() and a.sum()>=2
        assert r['choice']==int(np.where(a,s,-np.inf).argmax())
        z=np.sort(s[a]);gap=float(z[-1]-z[-2])
        groups[key].append(gap);episodes[key][r['episode']].append(gap)
rows=[];thresholds=[];episode_rows=[]
for key,x in sorted(groups.items()):
    a=np.array(x)
    assert len(a)==14376 and counts[key]['requests']==14400 and counts[key]['unscored']==24
    rows.append({'method':key[0],'mode':key[1],'scored':len(a),'unscored':24,
                 **{f'q_{q}':float(v) for q,v in zip([.01,.05,.5,.95,.99],np.quantile(a,[.01,.05,.5,.95,.99]))},
                 **{f'fraction_le_{t}':float((a<=t).mean()) for t in GRID}})
    assert len(episodes[key])==24 and all(len(v)==599 for v in episodes[key].values())
    for threshold in GRID:
        episode_fractions=[]
        for episode,x_episode in sorted(episodes[key].items()):
            ea=np.array(x_episode);fraction=float((ea<=threshold).mean())
            episode_fractions.append(fraction)
            episode_rows.append({'method':key[0],'mode':key[1],'episode':episode,
                                 'threshold':threshold,'n':len(ea),'count':int((ea<=threshold).sum()),
                                 'fraction':fraction})
        thresholds.append({'method':key[0],'mode':key[1],'threshold':threshold,
                           'count':int((a<=threshold).sum()),'n':len(a),
                           'fraction':float((a<=threshold).mean()),
                           'episode_fraction_min':min(episode_fractions),
                           'episode_fraction_max':max(episode_fractions)})

def verify_reference(filename,actual):
    """Compare exact parsed numeric values; no tolerance or rounded comparison."""
    path=HERE/filename
    if not path.exists():
        return 'reference not supplied'
    with path.open(newline='',encoding='utf-8') as f:
        expected=list(csv.DictReader(f))
    assert len(expected)==len(actual),f'{filename}: row count differs'
    for row_number,(reference,computed) in enumerate(zip(expected,actual),start=2):
        assert list(reference)==list(computed),f'{filename}: columns differ'
        for field,value in computed.items():
            same=(reference[field]==value if isinstance(value,str)
                  else float(reference[field])==value)
            assert same,f'{filename}:{row_number}: {field} differs'
    return {'rows':len(actual),'numeric_fields_equal':True,'identity_fields_equal':True}

checks={name:verify_reference(name,table) for name,table in
        [('threshold_fractions.csv',thresholds),('episode_threshold_fractions.csv',episode_rows)]}
for filename,table in [('near_tie_summary.csv',rows),('threshold_fractions.csv',thresholds),
                       ('episode_threshold_fractions.csv',episode_rows)]:
    with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(json.dumps({'summary':rows,'reference_checks':checks},indent=2))
