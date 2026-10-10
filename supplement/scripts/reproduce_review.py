"""Verify compact results and regenerate all six main figures and two supplementary command figures."""
from pathlib import Path
import subprocess,sys
P=Path(__file__).resolve().parents[1]
for name in ('verify_files.py','reproduce.py','reproduce_physical.py','reproduce_method_details.py','make_review_figures.py','reproduce_replay.py','make_replay_figure.py','make_deadline_figure.py'):
 subprocess.run([sys.executable,'-B',str(P/'scripts'/name)],check=True)
print('PASS: outputs/ contains the reanalysis and six main figures and two supplementary command figures.')
