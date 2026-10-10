from pathlib import Path
import hashlib
P=Path(__file__).resolve().parents[1]
count=0
for line in (P/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
 digest,name=line.split('  ',1)
 f=(P/name).resolve()
 assert f.is_relative_to(P) and hashlib.sha256(f.read_bytes()).hexdigest()==digest,name
 count+=1
print(f'PASS: {count} supplement file hashes')
