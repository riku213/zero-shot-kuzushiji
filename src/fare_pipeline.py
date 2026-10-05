from __future__ import annotations

import hashlib
import json
import random
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import torch


STRUCTURE_SYMBOLS = ("⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻")
STRUCTURE_TO_BITS = {symbol: format(index, "04b") for index, symbol in enumerate(STRUCTURE_SYMBOLS)}
STRUCTURE_SET = set(STRUCTURE_SYMBOLS)


@dataclass(slots=True)
class RadicalInventory:
    source_files: list[str]
    ids_entries: dict[str, str]
    radical_counts: dict[str, int]
    radicals: list[str]
    structures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RenderedRadical:
    radical: str
    codepoint: str
    image_path: Optional[str]
    renderable: bool
    font_path: Optional[str] = None
    status: str = "rendered"
    bbox: Optional[list[int]] = None
    ink_pixel_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RadicalFeatureRecord:
    radical: str
    codepoint: str
    image_path: str
    renderable: bool
    feature: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class FareCodeRecord:
    radical: str
    codepoint: str
    image_path: str
    renderable: bool
    feature: list[float]
    projected: list[float]
    fare_code: list[int]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_ids_entries(ids_files: Iterable[Path]) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for ids_file in ids_files:
        with ids_file.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith(";") or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                codepoint, character, expression = parts[0], parts[1], parts[2]
                entries.append((codepoint, character, normalize_ids_expression(expression)))

    return entries


def normalize_ids_expression(expression: str) -> str:
    cleaned = re.sub(r"\[.*?\]", "", expression)
    cleaned = cleaned.replace(" ", "")
    return cleaned


def merge_ids_entries(entries: Iterable[tuple[str, str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for _, character, expression in entries:
        if not character:
            continue

        existing = merged.get(character)
        if existing is None:
            merged[character] = expression
            continue

        if existing == character and expression != character:
            merged[character] = expression
            continue

        if len(expression) > len(existing) and expression != character:
            merged[character] = expression

    return merged


def extract_radicals_and_structures(ids_entries: dict[str, str]) -> tuple[list[str], list[str], Counter[str]]:
    radical_counts: Counter[str] = Counter()
    structure_symbols: set[str] = set()

    for expression in ids_entries.values():
        if not any(symbol in STRUCTURE_SET for symbol in expression):
            continue

        for symbol in expression:
            if symbol in STRUCTURE_SET:
                structure_symbols.add(symbol)
                continue
            if symbol.isspace():
                continue
            radical_counts[symbol] += 1

    radicals = sorted(radical_counts)
    structures = [symbol for symbol in STRUCTURE_SYMBOLS if symbol in structure_symbols]
    return radicals, structures, radical_counts


def build_inventory(ids_files: Iterable[Path]) -> RadicalInventory:
    source_files = [str(path) for path in ids_files]
    entries = read_ids_entries(ids_files)
    merged = merge_ids_entries(entries)
    radicals, structures, radical_counts = extract_radicals_and_structures(merged)

    return RadicalInventory(
        source_files=source_files,
        ids_entries=merged,
        radical_counts=dict(sorted(radical_counts.items(), key=lambda item: item[0])),
        radicals=radicals,
        structures=structures,
    )


def save_inventory(inventory: RadicalInventory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(inventory.to_dict(), handle)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory.to_dict(), handle, ensure_ascii=False, indent=2)


def load_inventory(inventory_path: Path) -> dict:
    with inventory_path.open("rb") as handle:
        return pickle.load(handle)


class FontSet:
    """Fonts in priority order; a text uses the first font whose cmap covers all its codepoints."""

    def __init__(self, font_paths: Iterable[Path]) -> None:
        self.paths = [Path(p) for p in font_paths]
        self.cmaps: list[set[int]] = []
        self.sha256: dict[Path, str] = {}
        for path in self.paths:
            font = TTFont(str(path), lazy=True)
            self.cmaps.append(set(font.getBestCmap().keys()))
            font.close()
            self.sha256[path] = hashlib.sha256(path.read_bytes()).hexdigest()

    def select(self, text: str) -> Optional[Path]:
        codepoints = [ord(char) for char in text]
        for path, cmap in zip(self.paths, self.cmaps):
            if all(cp in cmap for cp in codepoints):
                return path
        return None


def _best_font_size(text: str, font_path: Path, image_size: int, max_ratio: float = 0.78) -> ImageFont.FreeTypeFont:
    max_side = int(image_size * max_ratio)
    for size in range(int(image_size * 0.8), 10, -2):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = font.getbbox(text)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_side and height <= max_side:
            return font

    return ImageFont.truetype(str(font_path), size=18)


def is_renderable(text: str, font_path: Path) -> bool:
    font = ImageFont.truetype(str(font_path), size=48)
    try:
        bbox = font.getbbox(text)
    except Exception:
        return False

    return bbox is not None and (bbox[2] - bbox[0] > 0 or bbox[3] - bbox[1] > 0)


def render_radical_image(radical: str, font_path: Path, output_path: Path, image_size: int = 96) -> bool:
    renderable = is_renderable(radical, font_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(canvas)

    if renderable:
        font = _best_font_size(radical, font_path, image_size)
        bbox = draw.textbbox((0, 0), radical, font=font)
        if bbox is None:
            renderable = False
        else:
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (image_size - text_width) / 2 - bbox[0]
            y = (image_size - text_height) / 2 - bbox[1]
            draw.text((x, y), radical, font=font, fill=0)

    canvas.save(output_path)
    return renderable


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def render_radicals(radicals: Iterable[str], fonts: FontSet, output_dir: Path, image_size: int = 96) -> tuple[list[RenderedRadical], list[str]]:
    rendered: list[RenderedRadical] = []
    alien_radicals: list[str] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for radical in radicals:
        codepoint = "_".join(f"U+{ord(char):04X}" for char in radical)
        image_path = output_dir / f"{codepoint}.png"
        font_path = fonts.select(radical)
        item = RenderedRadical(
            radical=radical,
            codepoint=codepoint,
            image_path=None,
            renderable=False,
            font_path=_posix(font_path or fonts.paths[0]),
        )

        if font_path is None:
            item.status = "missing_glyph"
            item.error = "glyph_not_supported"
        else:
            try:
                canvas = _draw_radical(radical, font_path, image_size)
                mask = np.asarray(canvas) < 128
                item.ink_pixel_count = int(mask.sum())
                if item.ink_pixel_count == 0:
                    item.status = "blank"
                else:
                    ys, xs = np.nonzero(mask)
                    item.bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    output_dir.mkdir(parents=True, exist_ok=True)
                    canvas.save(image_path)
                    item.image_path = _posix(image_path)
                    item.renderable = True
                    item.status = "rendered"
            except Exception as exc:
                item.status = "error"
                item.error = f"{type(exc).__name__}: {exc}"

        rendered.append(item)
        if not item.renderable:
            alien_radicals.append(radical)

    return rendered, alien_radicals


def _draw_radical(radical: str, font_path: Path, image_size: int) -> Image.Image:
    canvas = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(canvas)
    font = _best_font_size(radical, font_path, image_size)
    bbox = draw.textbbox((0, 0), radical, font=font)
    x = (image_size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (image_size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), radical, font=font, fill=0)
    return canvas


def save_render_metadata(
    rendered: list[RenderedRadical],
    fonts: FontSet,
    ids_files: list[str],
    image_size: int,
    output_path: Path,
) -> None:
    statuses = ["rendered", "missing_glyph", "blank", "error"]
    counts = Counter(item.status for item in rendered)
    primary = fonts.paths[0]
    payload = {
        "schema_version": 1,
        "run": {
            "ids_files": [_posix(p) for p in ids_files],
            "font_path": _posix(primary),
            "font_sha256": fonts.sha256[primary],
            "image_size": image_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {"total": len(rendered), **{key: counts.get(key, 0) for key in statuses}},
        "entries": [
            {
                "text": item.radical,
                "codepoints": [f"U+{ord(char):04X}" for char in item.radical],
                "status": item.status,
                "renderable": item.renderable,
                "image_path": item.image_path,
                "font_path": item.font_path,
                "bbox": item.bbox,
                "ink_pixel_count": item.ink_pixel_count,
                "error": item.error,
            }
            for item in rendered
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_render_manifest(rendered: list[RenderedRadical], alien_radicals: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rendered": [item.to_dict() for item in rendered],
        "alien_radicals": alien_radicals,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_render_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("rb") as handle:
        return pickle.load(handle)


def load_rendered_radicals(manifest_path: Path) -> list[RenderedRadical]:
    manifest = load_render_manifest(manifest_path)
    rendered_items = manifest.get("rendered", [])
    return [RenderedRadical(**item) for item in rendered_items]


def image_to_tensor(image_path: Path, image_size: int = 96) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
        array = np.repeat(array, 3, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    normalize = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - normalize) / std


def save_pickle_and_json(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_binary_code(values: np.ndarray) -> list[int]:
    return [1 if value > 0 else -1 for value in values.tolist()]
