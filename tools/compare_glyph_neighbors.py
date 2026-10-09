from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/261007_correct_dict/all_glyph_codes.pkl"
DEFAULT_INVENTORY = ROOT / "outputs/radical_inventory.pkl"
DEFAULT_OUTPUT = ROOT / "outputs/261007_correct_dict/glyph_neighbors.html"
DEFAULT_FONT = ROOT / "dataset/fonts/HanaMinA.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare glyph neighbors in source features and 64-bit FaRE codes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--neighbors", type=int, default=12, help="Neighbors shown for each representation.")
    parser.add_argument("--batch-size", type=int, default=128, help="Query rows processed at once.")
    parser.add_argument("--interactive", action="store_true", help="Compare glyph neighbors in the terminal.")
    parser.add_argument("--all-glyphs", action="store_true", help="Include additional whole-character records in terminal results.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_payload(input_path: Path, inventory_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Input pickle must contain a 'records' list.")

    radical_set: set[str] | None = None
    if inventory_path.is_file():
        with inventory_path.open("rb") as handle:
            inventory = pickle.load(handle)
        radical_set = {str(value) for value in inventory.get("radicals", [])}

    normalized_records = []
    for record in records:
        if not isinstance(record, dict) or record.get("radical") is None:
            continue
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        normalized_records.append(
            {
                "glyph": str(record["radical"]),
                "codepoint": str(record.get("codepoint", "")),
                "image_path": image_path,
                "is_component": radical_set is None or str(record["radical"]) in radical_set,
                "feature": record.get("feature"),
                "code": record.get("fare_code"),
            }
        )
    if len(normalized_records) < 2:
        raise ValueError("At least two glyph records are required.")

    feature_rows = [record["feature"] for record in normalized_records]
    code_rows = [record["code"] for record in normalized_records]
    if any(row is None for row in feature_rows) or any(row is None for row in code_rows):
        raise ValueError("Every record must contain both 'feature' and 'fare_code'.")
    features = np.asarray(feature_rows, dtype=np.float32)
    codes = np.asarray(code_rows, dtype=np.float32)
    if features.ndim != 2 or codes.shape != (len(normalized_records), 64):
        raise ValueError(f"Unexpected vector shapes: features={features.shape}, codes={codes.shape}.")
    if not np.isfinite(features).all() or not np.isin(codes, (-1, 1)).all():
        raise ValueError("Features must be finite and codes must contain only -1/+1 values.")
    return normalized_records, features, codes


def nearest_tables(
    features: np.ndarray,
    codes: np.ndarray,
    component_mask: np.ndarray,
    neighbors: int,
    batch_size: int,
) -> tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]:
    count = len(features)
    neighbors = min(neighbors, count - 1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    source_tables: list[list[list[float]]] = [[] for _ in range(count)]
    code_tables: list[list[list[float]]] = [[] for _ in range(count)]
    source_component_tables: list[list[list[float]]] = [[] for _ in range(count)]
    code_component_tables: list[list[list[float]]] = [[] for _ in range(count)]

    def rank(scores: np.ndarray, allowed: np.ndarray, largest: bool) -> np.ndarray:
        candidate_ids = np.flatnonzero(allowed)
        candidate_scores = scores[candidate_ids]
        amount = min(neighbors, len(candidate_ids))
        if amount == 0:
            return np.empty(0, dtype=np.int64)
        selected = np.argpartition(-candidate_scores if largest else candidate_scores, amount - 1)[:amount]
        selected = selected[np.argsort(-candidate_scores[selected] if largest else candidate_scores[selected])]
        return candidate_ids[selected]

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        similarities = normalized[start:end] @ normalized.T
        agreements = codes[start:end] @ codes.T
        hamming = np.rint((64.0 - agreements) * 0.5).astype(np.int16)
        for local_index, row_index in enumerate(range(start, end)):
            similarities[local_index, row_index] = -np.inf
            hamming[local_index, row_index] = 65
            all_candidates = np.ones(count, dtype=bool)
            all_candidates[row_index] = False
            component_candidates = component_mask.copy()
            component_candidates[row_index] = False
            source_ids = rank(similarities[local_index], all_candidates, largest=True)
            code_ids = rank(hamming[local_index], all_candidates, largest=False)
            source_component_ids = rank(similarities[local_index], component_candidates, largest=True)
            code_component_ids = rank(hamming[local_index], component_candidates, largest=False)
            source_tables[row_index] = [[int(index), round(float(similarities[local_index, index]), 5)] for index in source_ids]
            code_tables[row_index] = [[int(index), int(hamming[local_index, index])] for index in code_ids]
            source_component_tables[row_index] = [
                [int(index), round(float(similarities[local_index, index]), 5)] for index in source_component_ids
            ]
            code_component_tables[row_index] = [
                [int(index), int(hamming[local_index, index])] for index in code_component_ids
            ]
        print(f"Ranked neighbors for {end:,}/{count:,} glyphs", flush=True)
    return source_tables, code_tables, source_component_tables, code_component_tables


HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>字形の最近傍比較</title>
<style>
@font-face{font-family:HanaMinA;src:url("__FONT__") format("truetype");font-display:swap}
:root{color-scheme:light;--ink:#202522;--muted:#68736c;--line:#d6ddd7;--paper:#f4f6f2;--red:#b14f37;--teal:#26756a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 "Yu Gothic UI","Meiryo",sans-serif}
header{display:flex;align-items:center;gap:20px;padding:15px 22px;background:white;border-bottom:1px solid var(--line)}h1{margin:0;font-size:19px;font-weight:650}
.controls{display:flex;align-items:center;gap:10px;margin-left:auto}input{width:min(340px,42vw);height:36px;padding:0 11px;border:1px solid #b9c4bc;border-radius:4px;font:inherit}
label{display:flex;align-items:center;gap:7px;color:#455149;white-space:nowrap}input[type=checkbox]{width:16px;height:16px;accent-color:var(--teal)}#status{color:var(--muted);font-size:12px;min-width:100px;text-align:right}
main{max-width:1500px;margin:0 auto;padding:20px 22px 36px}.query{display:flex;align-items:center;gap:18px;margin-bottom:18px;padding:12px 0;border-bottom:1px solid var(--line)}.query img{width:88px;height:88px;object-fit:contain;background:white;border:1px solid var(--line)}.query .glyph{font:64px/1 HanaMinA,"Yu Mincho",serif}.query .label{color:var(--muted);font-size:12px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:28px}section{min-width:0}h2{display:flex;align-items:baseline;gap:10px;margin:0 0 10px;font-size:16px}h2 small{font-size:12px;color:var(--muted);font-weight:400}.neighbor-list{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.item{display:grid;grid-template-columns:66px minmax(0,1fr);gap:9px;align-items:center;min-height:84px;padding:7px;background:white;border:1px solid var(--line);border-radius:4px}.item img{width:64px;height:64px;object-fit:contain;background:#fbfcfa}.item .glyph{font:34px/1 HanaMinA,"Yu Mincho",serif;text-align:center}.item .char{font:20px/1.2 HanaMinA,"Yu Mincho",serif}.item .codepoint{margin-top:3px;color:var(--muted);font-size:10px}.item .distance{margin-top:6px;font-size:11px;color:var(--teal)}.item.code .distance{color:var(--red)}.empty{padding:22px;color:var(--muted);background:white;border:1px dashed var(--line)}
@media(max-width:900px){.columns{grid-template-columns:1fr;gap:26px}.neighbor-list{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:580px){header{align-items:flex-start;flex-direction:column;padding:12px}.controls{width:100%;margin:0;flex-wrap:wrap}input{flex:1;width:auto}main{padding:14px 12px}.neighbor-list{grid-template-columns:1fr}.query{gap:12px}.query img{width:64px;height:64px}.query .glyph{font-size:48px}}
</style></head><body>
<header><h1>字形の最近傍比較</h1><div class="controls"><input id="search" list="glyph-options" placeholder="漢字または U+XXXX を入力" aria-label="漢字またはcodepointを検索"><datalist id="glyph-options"></datalist><label><input id="components-only" type="checkbox" checked>IDS構成要素のみ</label><span id="status"></span></div></header>
<main><div class="query" id="query-panel"><div class="glyph">?</div><div><strong>比較する文字を検索</strong><div class="label">元特徴と64ビットコードそれぞれの近傍を表示します</div></div></div>
<div class="columns"><section><h2>元特徴 <small>cosine類似度が高い順</small></h2><div class="neighbor-list" id="source-list"><div class="empty">文字を検索してください</div></div></section>
<section><h2>64ビットFaREコード <small>Hamming距離が小さい順</small></h2><div class="neighbor-list" id="code-list"><div class="empty">文字を検索してください</div></div></section></div></main>
<script>
const records=__RECORDS__,sourceNeighbors=__SOURCE__,codeNeighbors=__CODES__,sourceComponentNeighbors=__SOURCE_COMPONENTS__,codeComponentNeighbors=__CODE_COMPONENTS__;
const options=document.getElementById('glyph-options'),search=document.getElementById('search'),status=document.getElementById('status'),onlyComponents=document.getElementById('components-only');
const byGlyph=new Map(records.map((r,i)=>[r.glyph.toLowerCase(),i])),byCode=new Map(records.map((r,i)=>[r.codepoint.toLowerCase(),i]));
for(const r of records){const option=document.createElement('option');option.value=r.glyph;option.label=r.codepoint;options.appendChild(option)}
let selected=-1;
function resolve(query){const key=query.trim().toLowerCase();if(!key)return -1;if(byGlyph.has(key))return byGlyph.get(key);if(byCode.has(key))return byCode.get(key);return records.findIndex(r=>r.glyph.toLowerCase().includes(key)||r.codepoint.toLowerCase().includes(key))}
function card(index,score,kind){const r=records[index],item=document.createElement('article');item.className=`item ${kind}`;const media=r.image?document.createElement('img'):document.createElement('div');if(r.image){media.src=r.image;media.alt=r.glyph;media.onerror=()=>{media.replaceWith(glyphElement(r.glyph))}}else{media.className='glyph';media.textContent=r.glyph}item.appendChild(media);const text=document.createElement('div'),char=document.createElement('div'),cp=document.createElement('div'),dist=document.createElement('div');char.className='char';char.textContent=r.glyph;cp.className='codepoint';cp.textContent=r.codepoint;dist.className='distance';dist.textContent=kind==='source'?`cosine ${score.toFixed(4)}`:`Hamming ${score}/64`;text.append(char,cp,dist);item.appendChild(text);return item}
function glyphElement(glyph){const el=document.createElement('div');el.className='glyph';el.textContent=glyph;return el}
function render(){if(selected<0)return;const r=records[selected],query=document.getElementById('query-panel');query.replaceChildren();const image=r.image?document.createElement('img'):glyphElement(r.glyph);if(r.image){image.src=r.image;image.alt=r.glyph;image.onerror=()=>image.replaceWith(glyphElement(r.glyph))}query.appendChild(image);const info=document.createElement('div'),glyph=glyphElement(r.glyph),label=document.createElement('div');label.className='label';label.textContent=`${r.codepoint} · ${r.isComponent?'IDS構成要素':'追加文字'}`;info.append(glyph,label);query.appendChild(info);
for(const [id,table,componentTable,kind] of [['source-list',sourceNeighbors,sourceComponentNeighbors,'source'],['code-list',codeNeighbors,codeComponentNeighbors,'code']]){const list=document.getElementById(id);list.replaceChildren();const candidates=(onlyComponents.checked?componentTable:selected<0?[]:table)[selected];for(const [index,score] of candidates||[]){list.appendChild(card(index,score,kind))}if(!candidates?.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='条件に合う候補はありません';list.appendChild(empty)}}
status.textContent=`${selected+1} / ${records.length}`}
search.addEventListener('input',()=>{const index=resolve(search.value);if(index>=0){selected=index;render()}else status.textContent=search.value?'該当なし':`${records.length.toLocaleString()}字`});onlyComponents.addEventListener('change',render);
</script></body></html>"""


def make_html(
    records: list[dict[str, Any]],
    source_neighbors: list[list[list[float]]],
    code_neighbors: list[list[list[float]]],
    source_component_neighbors: list[list[list[float]]],
    code_component_neighbors: list[list[list[float]]],
    output_path: Path,
    font_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_target = output_path.with_name(f"{output_path.stem}.HanaMinA.ttf")
    if font_target.resolve() != font_path.resolve():
        shutil.copy2(font_path, font_target)
    serializable_records = []
    for record in records:
        try:
            image = record["image_path"].relative_to(output_path.parent)
        except ValueError:
            import os

            image = Path(os.path.relpath(record["image_path"], output_path.parent))
        serializable_records.append(
            {
                "glyph": record["glyph"],
                "codepoint": record["codepoint"],
                "image": image.as_posix() if record["image_path"].is_file() else "",
                "isComponent": record["is_component"],
            }
        )
    html = HTML.replace("__FONT__", font_target.name)
    html = html.replace("__RECORDS__", json.dumps(serializable_records, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__SOURCE__", json.dumps(source_neighbors, separators=(",", ":")))
    html = html.replace("__CODES__", json.dumps(code_neighbors, separators=(",", ":")))
    html = html.replace("__SOURCE_COMPONENTS__", json.dumps(source_component_neighbors, separators=(",", ":")))
    html = html.replace("__CODE_COMPONENTS__", json.dumps(code_component_neighbors, separators=(",", ":")))
    output_path.write_text(html, encoding="utf-8")


def terminal_search(records: list[dict[str, Any]], features: np.ndarray, codes: np.ndarray, neighbors: int, all_glyphs: bool) -> None:
    norms = np.linalg.norm(features, axis=1)
    normalized = features / np.maximum(norms[:, None], 1e-12)
    allowed = np.ones(len(records), dtype=bool) if all_glyphs else np.asarray(
        [record["is_component"] for record in records], dtype=bool
    )
    lookup: dict[str, int] = {}
    for index, record in enumerate(records):
        lookup.setdefault(record["glyph"].casefold(), index)
        lookup.setdefault(record["codepoint"].casefold(), index)

    print("文字またはU+XXXXを入力してください。終了するには q を入力します。", flush=True)
    while True:
        try:
            query = input("検索文字> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.casefold() in {"q", "quit", "exit"}:
            return
        index = lookup.get(query.casefold())
        if index is None:
            print(f"見つかりません: {query}\n", flush=True)
            continue

        target = records[index]
        source_scores = normalized @ normalized[index]
        hamming_distances = np.count_nonzero(codes != codes[index], axis=1)
        candidates = np.flatnonzero(allowed)
        candidates = candidates[candidates != index]
        source_order = candidates[np.argsort(-source_scores[candidates], kind="stable")[:neighbors]]
        code_order = candidates[np.argsort(hamming_distances[candidates], kind="stable")[:neighbors]]

        print(f"\n対象: {target['glyph']} ({target['codepoint']})")
        print(f"表示対象: {'全レコード' if all_glyphs else 'IDS構成要素'} / 上位{neighbors}件")
        print("\n[元特徴: cosine類似度 高い順]")
        for rank, candidate in enumerate(source_order, start=1):
            record = records[int(candidate)]
            print(f"{rank:2d}. {record['glyph']}  {record['codepoint']}  cosine={source_scores[candidate]:.5f}")
        print("\n[64ビットFaREコード: Hamming距離 小さい順]")
        for rank, candidate in enumerate(code_order, start=1):
            record = records[int(candidate)]
            distance = int(hamming_distances[candidate])
            print(f"{rank:2d}. {record['glyph']}  {record['codepoint']}  Hamming={distance}/64")
        print(flush=True)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    inventory_path = resolve_path(args.inventory)
    output_path = resolve_path(args.output)
    font_path = resolve_path(args.font)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    records, features, codes = load_payload(input_path, inventory_path)
    print(f"Loaded {len(records):,} glyphs: source features {features.shape[1]}D, FaRE codes {codes.shape[1]}D.", flush=True)
    if args.interactive:
        terminal_search(records, features, codes, args.neighbors, args.all_glyphs)
        return
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    component_mask = np.asarray([record["is_component"] for record in records], dtype=bool)
    source_neighbors, code_neighbors, source_component_neighbors, code_component_neighbors = nearest_tables(
        features, codes, component_mask, args.neighbors, args.batch_size
    )
    make_html(
        records,
        source_neighbors,
        code_neighbors,
        source_component_neighbors,
        code_component_neighbors,
        output_path,
        font_path,
    )
    component_count = sum(record["is_component"] for record in records)
    print(f"IDS components: {component_count:,}; additional glyphs: {len(records) - component_count:,}.", flush=True)
    print(f"Interactive comparison: {output_path}", flush=True)


if __name__ == "__main__":
    main()