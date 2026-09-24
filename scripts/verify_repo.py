#!/usr/bin/env python3
from pathlib import Path
import json, re, sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def rows(rel):
    p=ROOT/rel
    with p.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def ok(label, cond, detail=''):
    print(('OK  ' if cond else 'FAIL'), label, detail)
    if not cond:
        global failures
        failures += 1

failures=0
prompts=sorted((ROOT/'prompts/advisors').glob('advisor_*.md'))
ok('35 advisor prompts', len(prompts)==35, str(len(prompts)))
reg=json.loads((ROOT/'advisors/advisors_registry_35.json').read_text(encoding='utf-8'))
advisors=reg.get('advisors',reg)
ok('35-advisor registry', isinstance(advisors,list) and len(advisors)==35, str(len(advisors) if isinstance(advisors,list) else type(advisors)))
expected={
'data/train.jsonl':670,'data/validation.jsonl':220,
'data/train_meta_balanced.jsonl':131,'data/validation_meta.jsonl':30,
'data/train_specialist.jsonl':580,'data/validation_specialist.jsonl':190,
'data/train_matcher_v1.jsonl':252,'data/validation_matcher_v1.jsonl':67,
'data/train_matcher_v2.jsonl':256,'data/validation_matcher_v2.jsonl':128,
'data/train_specialist_v2_generated.jsonl':250,
'data/validation_specialist_v2_generated.jsonl':50,
'data/compiled/train_specialist_v2_all35_350.jsonl':350,
'data/compiled/validation_specialist_v2_all35_70.jsonl':70,
}
for rel,n in expected.items():
    try:c=len(rows(rel)); ok(rel,c==n,f'{c}/{n}')
    except Exception as e:ok(rel,False,str(e))
for rel in ['handler.py','handler_advisory_council.py']:
    try:
        compile((ROOT/rel).read_text(encoding='utf-8'),rel,'exec');ok(f'{rel} syntax',True)
    except Exception as e:ok(f'{rel} syntax',False,str(e))
for ck in ['base','meta','specialist','matcher']:
    p=ROOT/'checkpoints'/ck/'adapter_model.safetensors'
    real=p.exists() and p.stat().st_size>=10_000_000
    print(('OK  ' if real else 'MISS'),f'{ck} real adapter', p if real else 'real LFS object not recovered')
print('\nVERDICT:', 'repository files verified' if failures==0 else f'{failures} verification failure(s)')
sys.exit(1 if failures else 0)
