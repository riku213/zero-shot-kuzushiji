import pickle
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/final_codebook.pkl')
print('File:', p)
try:
    print('Size:', p.stat().st_size, 'bytes')
except Exception as e:
    print('Size: (error)', e)

with p.open('rb') as fh:
    payload = pickle.load(fh)

print('Top-level type:', type(payload))
if isinstance(payload, dict):
    print('Keys:', list(payload.keys()))
    if 'codebook' in payload and isinstance(payload['codebook'], dict):
        cb = payload['codebook']
        print('codebook is dict, entries:', len(cb))
        sample_keys = list(cb.keys())[:20]
        print('sample keys (up to 20):', sample_keys)
        for k in sample_keys:
            v = cb[k]
            print(' - key:', k, 'type:', type(v), 'len?', getattr(v, '__len__', lambda:None)())
    if 'entries' in payload and isinstance(payload['entries'], list):
        entries = payload['entries']
        print('entries list length:', len(entries))
        for i, item in enumerate(entries[:10]):
            print(' entry', i, 'type', type(item))
            if isinstance(item, dict):
                print('  keys', list(item.keys()))
                # try to show label-like fields
                for kk in ('unicode','class','label'):
                    if kk in item:
                        print('   ', kk, '=>', item[kk])

# fallback: if payload is dict of arrays
if isinstance(payload, dict):
    simple = {k: v for k, v in payload.items() if isinstance(v, (list, tuple))}
    if simple:
        print('Detected many list/tuple-valued items; sample count:', len(simple))
        sample_keys = list(simple.keys())[:20]
        for k in sample_keys:
            print(' -', k, 'len', len(simple[k]))

# if payload is list
if isinstance(payload, list):
    print('Top-level is list, length', len(payload))
    for i, item in enumerate(payload[:10]):
        print(' item', i, 'type', type(item))
        if isinstance(item, dict):
            print('  keys', list(item.keys()))

print('Done')
