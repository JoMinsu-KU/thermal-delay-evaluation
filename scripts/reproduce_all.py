"""Verify public inputs and reanalyse saved outcomes; no research execution."""
from pathlib import Path
import subprocess,sys
P=Path(__file__).resolve().parents[1]
for script in [P/'scripts/verify.py',P/'supplement/scripts/reproduce_review.py']:
    subprocess.run([sys.executable,'-B',str(script)],cwd=P,check=True)
print('PASS: manuscript tables and ten figures are reproduced under supplement/outputs/.')
