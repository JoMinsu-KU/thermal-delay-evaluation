"""Verify compact results and regenerate four main and six supplementary figures."""
from pathlib import Path
import subprocess,sys
P=Path(__file__).resolve().parents[1]
for name in ('verify_files.py','reproduce.py','reproduce_physical.py','reproduce_method_details.py','make_review_figures.py','reproduce_replay.py','make_replay_figure.py','make_deadline_figure.py','reproduce_revision_20261013.py','reproduce_small_seed.py','make_small_seed_figure.py'):
 subprocess.run([sys.executable,'-B',str(P/'scripts'/name)],check=True)
assert len(list((P/'outputs/figures').glob('*.png')))==10
print('PASS: outputs/ contains saved-data reanalyses, four main figures and six supplementary figures.')
