"""Recompute physical summaries from compact saved episode records."""
from pathlib import Path
import csv,json
import numpy as np
P=Path(__file__).resolve().parents[1];R=P/'data/revision';O=P/'outputs';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((R/'cost_components.csv').open(encoding='utf-8')))
assert len(rows)==240
expected=json.loads((R/'physical_results.json').read_text(encoding='utf-8'))
fields=['episode_cost','tracking_rmse_k','max_temperature_c','action_effort','action_delta_mean_square','J_tracking','J_terminal','J_command','J_increment']
for r in rows:
 assert np.isclose(sum(float(r[k]) for k in fields[-4:]),float(r['episode_cost']),atol=1e-12,rtol=1e-10)
 means=None
cells={}
for key,values in expected['means'].items():
 mode,method=key.split('/');rs=[r for r in rows if r['mode']==mode and r['method']==method]
 assert len(rs)==24 and len({r['episode'] for r in rs})==24
 cells[key]={k:float(np.mean([float(r[k]) for r in rs])) for k in fields}
 for k,v in cells[key].items():assert np.isclose(v,values[k],atol=1e-12,rtol=1e-10),(key,k)
result={'passed':True,'saved_episode_records':240,'cell_means':cells,'raw_quadrature_reconstruction':False,
 'scope':'Reanalysis of compact components; original quadrature inputs are identified by source_hashes.json and are not included in this compact supplement.'}
(O/'physical_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('PASS: 240 cost-component sums and all physical cell means')
