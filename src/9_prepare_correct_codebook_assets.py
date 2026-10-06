from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from fare_pipeline import (
    FontSet,
    load_inventory,
    merge_ids_entries,
    read_ids_entries,
    render_radicals,
    save_render_metadata,
    save_render_manifest,
)
from structured_codebook import STRUCTURE_SET, parse_ids_expression


CASIA_LABEL_ALIASES = {
    "asterisk": "*",
    "backslash": "\\",
    "colon": ":",
    "double_quote": '"',
    "forward_slash": "/",
    "full_stop": ".",
    "greater_than": ">",
    "less_than": "<",
    "question_mark": "?",
    "vertical_bar": "|",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the CASIA/Kuzushiji Unicode class union and missing glyph assets.")
    parser.add_argument("--casia-root", default="../kuzushiji-recognition/CASIA-HWDB")
    parser.add_argument("--kuzushiji-root", default="../kuzushiji-recognition/char_sep_datas")
    parser.add_argument("--ids-files", nargs="+", default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"])
    parser.add_argument("--radical-codes", default="outputs/261005_structured_codebook/radical_codes.pkl")
    parser.add_argument("--render-manifest", default="outputs/radical_render_manifest.pkl")
    parser.add_argument("--render-metadata", default="outputs/radical_render_metadata.json")
    parser.add_argument("--inventory", default="outputs/radical_inventory.pkl")
    parser.add_argument("--fonts", nargs="+", default=["dataset/fonts/HanaMinA.ttf", "dataset/fonts/HanaMinB.ttf"])
    parser.add_argument("--output-dir", default="outputs/261007_correct_dict")
    parser.add_argument("--image-size", type=int, default=96)
    return parser.parse_args()


def normalize_unicode_label(label: str) -> str:
    value = str(label).strip()
    if value.startswith(("U+", "u+")):
        try:
            return chr(int(value[2:], 16))
        except (ValueError, OverflowError):
            return value
    return value


def casia_label(folder_name: str) -> str | None:
    if folder_name in CASIA_LABEL_ALIASES:
        return CASIA_LABEL_ALIASES[folder_name]
    normalized = normalize_unicode_label(folder_name)
    return normalized if len(normalized) == 1 else None


def collect_casia_split(split_root: Path) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
    if not split_root.is_dir():
        raise FileNotFoundError(f"CASIA split folder not found: {split_root}")
    top_level_files = [path.name for path in split_root.iterdir() if path.is_file()]
    label_counts: dict[str, int] = {}
    label_sources: dict[str, str] = {}
    invalid_folders: list[str] = []
    invalid_files: list[dict[str, str]] = []
    class_folders = 0

    for class_dir in split_root.iterdir():
        if not class_dir.is_dir():
            continue
        class_folders += 1
        label = casia_label(class_dir.name)
        if label is None:
            invalid_folders.append(class_dir.name)
            continue
        samples = 0
        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                invalid_files.append({"path": str(image_path), "reason": "nested_directory"})
                continue
            if image_path.suffix.lower() != ".png" or not image_path.stem.isdigit():
                invalid_files.append({"path": str(image_path), "reason": "expected_numeric_png"})
                continue
            samples += 1
        if label in label_counts:
            raise ValueError(f"Multiple CASIA folders normalize to the same class label {label!r} in {split_root}.")
        label_counts[label] = samples
        label_sources[label] = class_dir.name

    if top_level_files:
        raise ValueError(f"Expected all CASIA images to be inside label directories; found files directly under {split_root}.")
    if invalid_folders:
        raise ValueError(f"CASIA has non-single-character label folders not covered by explicit aliases: {invalid_folders[:30]!r}")
    if invalid_files:
        raise ValueError(f"CASIA class folders contain unexpected files: {invalid_files[:30]!r}")
    return label_counts, label_sources, {"class_folders": class_folders, "top_level_files": top_level_files}


def collect_kuzushiji_classes(root: Path) -> tuple[set[str], dict[str, list[str]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Kuzushiji dataset root not found: {root}")
    classes: set[str] = set()
    source_folders: dict[str, list[str]] = {}
    for book in root.iterdir():
        character_root = book / "characters"
        if not book.is_dir() or not character_root.is_dir():
            continue
        for class_dir in character_root.iterdir():
            if not class_dir.is_dir():
                continue
            label = normalize_unicode_label(class_dir.name)
            if len(label) != 1:
                raise ValueError(f"Kuzushiji class folder is not one Unicode character: {class_dir}")
            classes.add(label)
            source_folders.setdefault(label, []).append(str(class_dir))
    return classes, source_folders


def load_radical_keys(path: Path) -> set[str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("radical_to_code"), dict):
        raise ValueError(f"Expected radical_to_code mapping in {path}.")
    return {str(key) for key in payload["radical_to_code"]}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    casia_root = Path(args.casia_root)
    train_counts, train_source_names, train_layout = collect_casia_split(casia_root / "CASIA-HWDB_Train" / "Train")
    test_counts, test_source_names, test_layout = collect_casia_split(casia_root / "CASIA-HWDB_Test" / "Test")
    train_labels, test_labels = set(train_counts), set(test_counts)
    if train_labels != test_labels:
        raise ValueError(
            f"CASIA Train/Test class label sets differ: train_only={len(train_labels - test_labels)}, "
            f"test_only={len(test_labels - train_labels)}."
        )

    kuzushiji_classes, kuzushiji_sources = collect_kuzushiji_classes(Path(args.kuzushiji_root))
    class_union = train_labels | kuzushiji_classes
    ids_entries = merge_ids_entries(read_ids_entries([Path(item) for item in args.ids_files]))
    ids_characters = set(ids_entries)
    radical_keys = load_radical_keys(Path(args.radical_codes))

    missing_ids = class_union - ids_characters
    # Render whole characters absent from IDS when their code is not already present.
    extra_render_candidates = sorted(
        char for char in missing_ids if char not in radical_keys and char not in STRUCTURE_SET
    )
    extra_manifest_path = output_dir / "union_character_render_manifest.pkl"
    extra_metadata_path = output_dir / "union_character_render_metadata.json"
    extra_image_root = output_dir / "union_character_images"
    fonts = FontSet(Path(item) for item in args.fonts)
    rendered_extra, alien_extra = render_radicals(
        extra_render_candidates,
        fonts,
        extra_image_root,
        image_size=args.image_size,
    )
    save_render_manifest(rendered_extra, alien_extra, extra_manifest_path)
    save_render_metadata(rendered_extra, fonts, [str(path) for path in args.ids_files], args.image_size, extra_metadata_path)

    with Path(args.render_manifest).open("rb") as handle:
        base_manifest = pickle.load(handle)
    base_metadata = json.loads(Path(args.render_metadata).read_text(encoding="utf-8"))
    extra_metadata = json.loads(extra_metadata_path.read_text(encoding="utf-8"))
    combined_manifest_entries = list(base_manifest.get("rendered", [])) + [item.to_dict() for item in rendered_extra]
    manifest_by_character = {str(item["radical"]): item for item in combined_manifest_entries}
    if len(manifest_by_character) != len(combined_manifest_entries):
        raise ValueError("Duplicate characters found while combining existing radicals and union glyphs.")
    combined_manifest = {
        "rendered": list(manifest_by_character.values()),
        "alien_radicals": sorted(set(base_manifest.get("alien_radicals", [])) | set(alien_extra)),
    }
    combined_metadata_entries = list(base_metadata.get("entries", [])) + list(extra_metadata.get("entries", []))
    combined_metadata = {
        "schema_version": 1,
        "run": {
            "base_metadata": base_metadata.get("run", {}),
            "extra_glyph_metadata": extra_metadata.get("run", {}),
        },
        "summary": {
            "total": len(combined_metadata_entries),
            "rendered": sum(entry.get("status") == "rendered" for entry in combined_metadata_entries),
            "missing_glyph": sum(entry.get("status") == "missing_glyph" for entry in combined_metadata_entries),
            "blank": sum(entry.get("status") == "blank" for entry in combined_metadata_entries),
            "error": sum(entry.get("status") == "error" for entry in combined_metadata_entries),
        },
        "entries": combined_metadata_entries,
    }
    combined_manifest_path = output_dir / "combined_render_manifest.pkl"
    combined_metadata_path = output_dir / "combined_render_metadata.json"
    with combined_manifest_path.open("wb") as handle:
        pickle.dump(combined_manifest, handle)
    combined_metadata_path.write_text(json.dumps(combined_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    classes: dict[str, dict[str, Any]] = {}
    for label in sorted(class_union):
        classes[f"U+{ord(label):04X}"] = {
            "character": label,
            "source": {
                "casia": label in train_labels,
                "kuzushiji": label in kuzushiji_classes,
            },
            "casia_train_samples": train_counts.get(label, 0),
            "casia_test_samples": test_counts.get(label, 0),
            "casia_folder_name": train_source_names.get(label),
            "kuzushiji_class_dirs": kuzushiji_sources.get(label, []),
            "ids_expression": ids_entries.get(label),
        }

    unresolved_missing_ids = [
        {
            "unicode": f"U+{ord(char):04X}",
            "character": char,
            "render_status": next(
                (entry.get("status") for entry in extra_metadata["entries"] if entry.get("text") == char),
                "already_has_radical_code" if char in radical_keys else "not_rendered",
            ),
        }
        for char in extra_render_candidates
        if not next((entry.get("renderable", False) for entry in extra_metadata["entries"] if entry.get("text") == char), False)
    ]
    audit = {
        "schema_version": 1,
        "casia_root": str(casia_root.resolve()),
        "kuzushiji_root": str(Path(args.kuzushiji_root).resolve()),
        "ids_files": [str(Path(item).resolve()) for item in args.ids_files],
        "casia_train": train_layout | {"labels": len(train_labels), "samples": sum(train_counts.values())},
        "casia_test": test_layout | {"labels": len(test_labels), "samples": sum(test_counts.values())},
        "kuzushiji_classes": len(kuzushiji_classes),
        "intersection_classes": len(train_labels & kuzushiji_classes),
        "union_classes": len(class_union),
        "ids_definitions": len(ids_entries),
        "union_classes_without_ids": len(missing_ids),
        "whole_character_render_candidates": len(extra_render_candidates),
        "whole_character_rendered": sum(item.renderable for item in rendered_extra),
        "whole_character_unrenderable": len(alien_extra),
        "unresolved_missing_ids": unresolved_missing_ids,
        "classes": classes,
    }
    audit_path = output_dir / "class_union_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Class union prepared: "
        f"CASIA={len(train_labels)} Kuzushiji={len(kuzushiji_classes)} "
        f"intersection={len(train_labels & kuzushiji_classes)} union={len(class_union)}"
    )
    print(
        "IDS/image coverage: "
        f"ids_missing={len(missing_ids)} extra_glyphs={len(extra_render_candidates)} "
        f"rendered={sum(item.renderable for item in rendered_extra)} unrenderable={len(alien_extra)}"
    )
    print(f"Saved class union audit to {audit_path.resolve()}")
    print(f"Saved combined render manifest to {combined_manifest_path.resolve()}")
    print(f"Saved combined render metadata to {combined_metadata_path.resolve()}")


if __name__ == "__main__":
    main()