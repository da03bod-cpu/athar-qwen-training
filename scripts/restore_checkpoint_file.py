#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('source')
p.add_argument('checkpoint',choices=['base','meta','specialist','matcher','specialist-v2'])
p.add_argument('--name',default='adapter_model.safetensors')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
src=Path(a.source)
if not src.is_file(): raise SystemExit(f'not found: {src}')
if a.name=='adapter_model.safetensors' and src.stat().st_size<10_000_000:
    raise SystemExit('source is too small; this looks like an LFS pointer, not real weights')
dst=root/'checkpoints'/a.checkpoint/a.name
dst.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(src,dst)
print('restored',dst,'bytes=',dst.stat().st_size)
