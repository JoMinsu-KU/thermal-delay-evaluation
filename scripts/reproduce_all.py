"""Verify the public package and reproduce the current manuscript (MIT)."""
from pathlib import Path
import subprocess, sys
P = Path(__file__).resolve().parents[1]
for path in (P/'scripts/verify.py', P/'supplement/scripts/reproduce_review.py'):
    subprocess.run([sys.executable, '-B', str(path)], cwd=P, check=True)
print('PASS: current manuscript outputs are in supplement/outputs/.')
