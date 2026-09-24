#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'docs/LFS_POINTER_OIDS.json').read_text())
for r in records:
    path=ROOT/'checkpoints'/r['checkpoint']/r['file']
    real=path.exists() and path.stat().st_size>1000000
    print(f"{r['checkpoint']:10} {r['file']:28} {'RECOVERED' if real else 'MISSING'} expected_size={r['original_size']} oid={r['oid_sha256']}")
