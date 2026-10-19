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
DEFAULT_OUTPUT = ROOT / "outputs/261007_correct_dict/radical_map.html"
DEFAULT_FONT = ROOT / "dataset/fonts/HanaMinA.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive 2D map of 64-dimensional glyph codes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input all_glyph_codes.pkl.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output interactive HTML.")
    parser.add_argument("--method", choices=("umap", "tsne", "pca"), default="umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="Font copied beside the generated HTML.")
    parser.add_argument("--no-cache", action="store_true", help="Recompute the 2D coordinates.")
    return parser.parse_args()


def load_data(input_path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    codes = payload.get("radical_to_code")
    records = payload.get("records")
    if not isinstance(codes, dict) or not isinstance(records, list):
        raise ValueError("Input pickle must contain 'radical_to_code' and 'records'.")

    record_by_radical = {
        str(record["radical"]): record
        for record in records
        if isinstance(record, dict) and record.get("radical") is not None
    }
    radicals = [str(radical) for radical in codes if str(radical) in record_by_radical]
    vectors = np.asarray([codes[radical] for radical in radicals], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 64:
        raise ValueError(f"Expected 64-dimensional vectors, got shape {vectors.shape}.")
    if len(radicals) < 3:
        raise ValueError("At least three glyphs are required for a 2D embedding.")

    output_records = []
    for radical in radicals:
        record = record_by_radical[radical]
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        output_records.append(
            {
                "glyph": radical,
                "codepoint": str(record.get("codepoint", "")),
                "source": str(record.get("source", "")),
                "image": image_path,
            }
        )
    return output_records, vectors


def get_coordinates(vectors: np.ndarray, input_path: Path, output_path: Path, method: str, seed: int, use_cache: bool) -> np.ndarray:
    cache_path = output_path.with_suffix(".coordinates.npy")
    metadata_path = output_path.with_suffix(".coordinates.json")
    stat = input_path.stat()
    metadata = {"input": str(input_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "method": method, "seed": seed}
    if use_cache and cache_path.exists() and metadata_path.exists():
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")) == metadata:
                coordinates = np.load(cache_path, allow_pickle=False)
                if coordinates.shape == (len(vectors), 2):
                    print(f"Using cached coordinates: {cache_path}", flush=True)
                    return coordinates
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="hamming",
            init="random",
            random_state=seed,
        )
        coordinates = reducer.fit_transform(vectors)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        coordinates = TSNE(n_components=2, init="random", learning_rate="auto", random_state=seed).fit_transform(vectors)
    else:
        from sklearn.decomposition import PCA

        coordinates = PCA(n_components=2, random_state=seed).fit_transform(vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, coordinates)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return coordinates


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>部首コードの2次元マップ</title>
<style>
@font-face { font-family: HanaMinA; src: url("__FONT__") format("truetype"); font-display: swap; }
:root { color-scheme: light; --ink: #202522; --muted: #69736d; --line: #d7ddd8; --paper: #f7f8f4; --accent: #bd4d32; --green: #1d756c; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--paper); font: 14px/1.45 "Yu Gothic UI", "Meiryo", sans-serif; }
header { display: flex; align-items: center; gap: 20px; min-height: 64px; padding: 12px 22px; border-bottom: 1px solid var(--line); background: #fff; }
h1 { margin: 0; font-size: 18px; font-weight: 650; white-space: nowrap; }
.toolbar { display: flex; align-items: center; gap: 8px; flex: 1; }
input { min-width: 150px; width: min(360px, 40vw); height: 34px; padding: 0 10px; border: 1px solid #bfc9c2; border-radius: 4px; background: white; font: inherit; }
button { height: 34px; padding: 0 11px; border: 1px solid #aebbb3; border-radius: 4px; background: white; color: var(--ink); font: inherit; cursor: pointer; }
button:hover { border-color: var(--green); color: var(--green); }
#status { margin-left: auto; color: var(--muted); white-space: nowrap; font-size: 12px; }
main { position: relative; height: calc(100vh - 65px); min-height: 360px; overflow: hidden; background-color: #f7f8f4; background-image: radial-gradient(#dce2dc 0.75px, transparent 0.75px); background-size: 19px 19px; }
canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
#tooltip { position: absolute; z-index: 2; display: none; max-width: 300px; padding: 10px 12px; border: 1px solid #bac5bd; border-radius: 4px; background: rgba(255,255,255,.97); box-shadow: 0 5px 18px #24332b24; pointer-events: none; }
#tooltip.pinned { border: 2px solid var(--accent); }
.tip-main { display: flex; align-items: center; gap: 10px; }
.glyph { font-family: HanaMinA, "Yu Mincho", serif; font-size: 42px; line-height: 1; }
.tip-image { width: 46px; height: 46px; object-fit: contain; image-rendering: auto; border: 1px solid #e1e5e1; background: white; }
.meta { margin-top: 7px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.legend { position: absolute; left: 14px; bottom: 12px; padding: 6px 9px; border: 1px solid var(--line); border-radius: 3px; background: #ffffffdc; color: var(--muted); font-size: 11px; pointer-events: none; }
@media (max-width: 650px) { header { align-items: flex-start; flex-wrap: wrap; gap: 8px; padding: 10px 12px; } h1 { width: 100%; } .toolbar { width: 100%; flex-wrap: wrap; } input { flex: 1; width: auto; } #status { margin-left: 0; } main { height: calc(100vh - 108px); } }
</style>
</head>
<body>
<header>
  <h1>部首コードの2次元マップ</h1>
  <div class="toolbar">
    <input id="search" type="search" placeholder="漢字または codepoint で検索" aria-label="漢字または codepoint で検索">
    <button id="reset" type="button" title="表示範囲をリセット">表示リセット</button>
    <button id="unpin" type="button" title="ポップアップを閉じる">固定解除</button>
    <span id="status"></span>
  </div>
</header>
<main id="stage">
  <canvas id="map"></canvas>
  <div id="tooltip"><div class="tip-main"><span class="glyph"></span><img class="tip-image" alt="字形画像"></div><div class="meta"></div></div>
  <div class="legend">点にホバーで表示・クリックで固定 / ドラッグで移動・ホイールで拡大縮小</div>
</main>
<script>
const points = __DATA__;
const canvas = document.getElementById('map'), ctx = canvas.getContext('2d');
const stage = document.getElementById('stage'), tooltip = document.getElementById('tooltip');
const search = document.getElementById('search'), status = document.getElementById('status');
let width = 0, height = 0, zoom = 1, panX = 0, panY = 0, hovered = -1, pinned = -1, dragging = false, moved = false;
let lastX = 0, lastY = 0, searchIndex = -1;
const bounds = points.reduce((b, p) => ({ minX: Math.min(b.minX, p.x), maxX: Math.max(b.maxX, p.x), minY: Math.min(b.minY, p.y), maxY: Math.max(b.maxY, p.y) }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
const spanX = Math.max(bounds.maxX - bounds.minX, 1e-9), spanY = Math.max(bounds.maxY - bounds.minY, 1e-9);
const cells = new Map(), cellSize = 0.025;
for (let i = 0; i < points.length; i++) {
  points[i].nx = (points[i].x - bounds.minX) / spanX;
  points[i].ny = (points[i].y - bounds.minY) / spanY;
  const key = `${Math.floor(points[i].nx / cellSize)},${Math.floor(points[i].ny / cellSize)}`;
  if (!cells.has(key)) cells.set(key, []);
  cells.get(key).push(i);
}
function baseSize() { return Math.max(1, Math.min(width - 56, height - 56)); }
function screen(p) { const s = baseSize(); return { x: (width - s) / 2 + p.nx * s * zoom + panX, y: (height - s) / 2 + p.ny * s * zoom + panY }; }
function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const radius = Math.max(1.45, Math.min(3.2, 2.7 - Math.log10(points.length / 1000 + 1) * .45));
  for (let i = 0; i < points.length; i++) {
    const p = points[i], pos = screen(p);
    if (pos.x < -4 || pos.y < -4 || pos.x > width + 4 || pos.y > height + 4) continue;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, i === hovered || i === pinned || i === searchIndex ? 5 : radius, 0, Math.PI * 2);
    ctx.fillStyle = i === pinned ? '#bd4d32' : i === searchIndex ? '#1d756c' : (p.renderable ? '#526a5d' : '#a5aaa6'); ctx.globalAlpha = i === hovered || i === pinned || i === searchIndex ? 1 : .72; ctx.fill();
    if (i === hovered || i === pinned || i === searchIndex) { ctx.globalAlpha = 1; ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke(); }
  }
  ctx.globalAlpha = 1;
}
function resize() { const rect = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; width = rect.width; height = rect.height; canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); draw(); }
function nearest(x, y) {
  const s = baseSize() * zoom, nx = (x - (width - baseSize()) / 2 - panX) / s, ny = (y - (height - baseSize()) / 2 - panY) / s;
  const range = Math.ceil((12 / s) / cellSize) + 1, cx = Math.floor(nx / cellSize), cy = Math.floor(ny / cellSize); let best = -1, bestDistance = 144;
  for (let gx = cx - range; gx <= cx + range; gx++) for (let gy = cy - range; gy <= cy + range; gy++) {
    for (const i of cells.get(`${gx},${gy}`) || []) { const p = screen(points[i]), d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bestDistance) { best = i; bestDistance = d; } }
  }
  return best;
}
function showTip(index, x, y, isPinned) {
  if (index < 0) { tooltip.style.display = 'none'; return; }
  const p = points[index]; tooltip.querySelector('.glyph').textContent = p.glyph || '�';
    const image = tooltip.querySelector('.tip-image'); image.src = p.image; image.style.display = p.image ? 'block' : 'none';
    image.onerror = () => { image.style.display = 'none'; };
  tooltip.querySelector('.meta').textContent = `${p.codepoint || ''}${p.source ? ' · ' + p.source : ''}`;
  tooltip.classList.toggle('pinned', isPinned); tooltip.style.display = 'block';
  const left = Math.max(8, Math.min(width - 310, x + 14)), top = Math.max(8, Math.min(height - 116, y + 14));
  tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
}
function eventPosition(event) { const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
canvas.addEventListener('pointerdown', event => { const p = eventPosition(event); dragging = true; moved = false; lastX = p.x; lastY = p.y; canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener('pointermove', event => {
  const p = eventPosition(event);
  if (dragging) { const dx = p.x - lastX, dy = p.y - lastY; if (Math.abs(dx) + Math.abs(dy) > 2) moved = true; panX += dx; panY += dy; lastX = p.x; lastY = p.y; draw(); return; }
  hovered = nearest(p.x, p.y); if (pinned < 0) showTip(hovered, p.x, p.y, false); draw();
});
canvas.addEventListener('pointerup', event => {
  const p = eventPosition(event); dragging = false;
  if (!moved) { const hit = nearest(p.x, p.y); if (hit === pinned) pinned = -1; else pinned = hit; showTip(pinned >= 0 ? pinned : hovered, p.x, p.y, pinned >= 0); draw(); }
});
canvas.addEventListener('pointerleave', () => { if (!dragging) { hovered = -1; if (pinned < 0) tooltip.style.display = 'none'; draw(); } });
canvas.addEventListener('wheel', event => { event.preventDefault(); const p = eventPosition(event), oldZoom = zoom; zoom = Math.max(.6, Math.min(32, zoom * Math.exp(-event.deltaY * .001))); const factor = zoom / oldZoom; panX = p.x - (p.x - panX) * factor; panY = p.y - (p.y - panY) * factor; draw(); }, { passive: false });
document.getElementById('reset').addEventListener('click', () => { zoom = 1; panX = 0; panY = 0; pinned = -1; tooltip.style.display = 'none'; draw(); });
document.getElementById('unpin').addEventListener('click', () => { pinned = -1; tooltip.style.display = 'none'; draw(); });
search.addEventListener('input', () => {
  const query = search.value.trim().toLowerCase(); searchIndex = query ? points.findIndex(p => p.glyph.toLowerCase() === query || p.codepoint.toLowerCase() === query) : -1;
  if (searchIndex < 0 && query) searchIndex = points.findIndex(p => p.glyph.toLowerCase().includes(query) || p.codepoint.toLowerCase().includes(query));
  status.textContent = searchIndex >= 0 ? `${searchIndex + 1} / ${points.length}` : (query ? '該当なし' : `${points.length.toLocaleString()} 字`);
  if (searchIndex >= 0) { zoom = Math.max(zoom, 2.5); const pos = screen(points[searchIndex]); panX += width / 2 - pos.x; panY += height / 2 - pos.y; pinned = searchIndex; showTip(pinned, width / 2, height / 2, true); }
  draw();
});
window.addEventListener('keydown', event => { if (event.key === 'Escape') { pinned = -1; tooltip.style.display = 'none'; draw(); } });
new ResizeObserver(resize).observe(stage); status.textContent = `${points.length.toLocaleString()} 字`; resize();
</script>
</body>
</html>"""


def write_html(records: list[dict[str, Any]], coordinates: np.ndarray, output_path: Path, font_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_target = output_path.with_name(f"{output_path.stem}.HanaMinA.ttf")
    if font_target.resolve() != font_path.resolve():
        shutil.copy2(font_path, font_target)

    data = []
    for record, (x, y) in zip(records, coordinates, strict=True):
        try:
            image = record["image"].relative_to(output_path.parent)
        except ValueError:
            image = Path(__import__("os").path.relpath(record["image"], output_path.parent))
        data.append(
            {
                "glyph": record["glyph"],
                "codepoint": record["codepoint"],
                "source": record["source"],
                "image": image.as_posix() if record["image"].is_file() else "",
                "renderable": record["image"].is_file(),
                "x": float(x),
                "y": float(y),
            }
        )
    html = HTML_TEMPLATE.replace("__FONT__", font_target.name).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    )
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    font_path = args.font if args.font.is_absolute() else ROOT / args.font
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    records, vectors = load_data(input_path)
    print(f"Loaded {len(records):,} glyphs with {vectors.shape[1]}-dimensional codes.", flush=True)
    coordinates = get_coordinates(vectors, input_path, output_path, args.method, args.seed, not args.no_cache)
    write_html(records, coordinates, output_path, font_path)
    print(f"Interactive map: {output_path}", flush=True)
    print(f"Font copied to: {output_path.with_name(f'{output_path.stem}.HanaMinA.ttf')}", flush=True)


if __name__ == "__main__":
    main()