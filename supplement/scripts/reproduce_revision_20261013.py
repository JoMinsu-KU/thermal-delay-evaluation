"""Recompute saved score-margin, recovery and thermal-response diagnostics."""
from pathlib import Path
import json,subprocess,sys,shutil
P=Path(__file__).resolve().parents[1]
D=P/'data/revision_20261013';O=P/'outputs'
N=D/'neartie';M=D/'mechanism'
subprocess.run([sys.executable,'-B',str(N/'reproduce_neartie.py'),str(O/'neartie')],check=True)
subprocess.run([sys.executable,'-B',str(M/'reproduce_compact.py'),'--input',str(M),'--output',str(O/'mechanism')],check=True)
checked=[]
for name in ('physical_episode_metrics.csv','deadline_episode_metrics.csv','deadline_events.csv','physical_summary.csv','setting_summary.csv','summary.json'):
    assert (M/name).read_bytes()==(O/'mechanism'/name).read_bytes(),name
    checked.append(name)
for name in ('threshold_fractions.csv','episode_threshold_fractions.csv'):
    assert (N/name).read_bytes()==(O/'neartie'/name).read_bytes(),name
    checked.append(name)
# The schedule audit records the archived raw extraction; copy it next to the
# recomputed setting summary solely for the descriptive figure.
shutil.copy2(M/'extraction_audit.json',O/'mechanism/extraction_audit.json')
subprocess.run([sys.executable,'-B',str(M/'make_mechanism_figure.py'),'--input',str(O/'mechanism'),'--output',str(O/'figures')],check=True)
(O/'revision_20261013_checks.json').write_text(json.dumps({'passed':True,'byte_equal_derived_files':checked,'scope':'Saved-data reanalysis only; no training, inference or simulation'},indent=2),encoding='utf-8')
print('PASS: saved score margins, restoration metrics, and thermal responses agree with bundled derived data.')
