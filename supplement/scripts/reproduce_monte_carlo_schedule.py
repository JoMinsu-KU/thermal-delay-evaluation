"""Audit the exported actual FV schedules and consumed latency draws."""
from pathlib import Path
import subprocess
import sys

P=Path(__file__).resolve().parents[1]
subprocess.run([
    sys.executable,'-B',str(P/'data/monte_carlo_schedule_audit/verify_mc_audit.py'),
    '--output',str(P/'outputs/monte_carlo_schedule_audit'),
],check=True)
