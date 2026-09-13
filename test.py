from pathlib import Path
import pickle, sys
from tqdm import tqdm

cb = pickle.load(open("outputs/final_codebook.pkl","rb"))
# cb may be dict or wrapped; normalize to dict of string keys
if isinstance(cb, dict) and "codebook" in cb and isinstance(cb["codebook"], dict):
    cb = cb["codebook"]
if isinstance(cb, dict) and "entries" in cb and isinstance(cb["entries"], list):
    new={}
    for it in cb["entries"]:
        k = it.get("unicode") or it.get("class") or it.get("label")
        v = it.get("code") or it.get("vector") or it.get("embedding")
        if k and v: new[str(k)] = v
    cb = new
codebook_keys = set(str(k) for k in cb.keys())

def unicode_aliases(text):
    a=set()
    t=str(text).strip()
    if not t: return a
    a.add(t)
    if t.upper().startswith("U+"):
        try:
            cp=int(t[2:],16); a.add(chr(cp))
        except: pass
    else:
        try: a.add(f"U+{ord(t):04X}".upper())
        except: pass
        if len(t)==1: a.add(f"U+{ord(t):04X}".upper())
    return a

manifest = Path("outputs/manifests/pretrain_manifest.txt")
if not manifest.exists():
    print("manifest not found", file=sys.stderr); sys.exit(1)

total=0; matched=0
examples=[]
alias_map = {}
for k in codebook_keys:
    for a in unicode_aliases(k):
        alias_map[str(a)] = k

for i,line in enumerate(tqdm(manifest.open(encoding="utf-8"), desc="Checking manifest", unit="lines")):
    p=Path(line.strip())
    total+=1
    # candidates: filename stem, parent name, grandparents up to manifest root 6 levels
    cand=[p.stem, p.parent.name]
    cand += [pp.name for pp in p.parents[:6]]
    ok=False
    for c in cand:
        if c in codebook_keys or c in alias_map:
            ok = True
            break
    if ok:
        matched+=1
        if len(examples)<10: examples.append((p, cand))
    if i>200000 and total%100000==0:
        print("scanned", total, "matched", matched)
print("total entries:", total)
print("matched entries:", matched)
print("examples:", examples[:10])