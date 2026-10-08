"""Verify package SHA256 manifests without running analysis or experiments (MIT)."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    manifest=root/'MANIFEST.sha256'
    count=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1)
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes root: '+name)
        if not p.is_file() or digest(p)!=expected:
            raise ValueError('Missing or changed file: '+name)
        count+=1
    return count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir',type=Path)
    args=ap.parse_args()
    result={'repository_files_verified':verify(ROOT)}
    if args.raw_dir:
        for name in ('primary','numerical'):
            result[name+'_files_verified']=verify(args.raw_dir/name)
    print(json.dumps({'passed':True,**result},indent=2))


if __name__=='__main__':
    main()
