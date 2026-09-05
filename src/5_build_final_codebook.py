from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from fare_pipeline import STRUCTURE_SET, merge_ids_entries, read_ids_entries
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final Unicode-to-codebook dictionary for training.")
    parser.add_argument(
        "--ids-files",
        nargs="+",
        default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"],
        help="IDS files used to enumerate Unicode classes.",
    )
    parser.add_argument(
        "--radical-codes",
        default="outputs/radical_codes.pkl",
        help="FaRE code file produced by the radical code generation step.",
    )
    parser.add_argument(
        "--dataset-root",
        default="../kuzushiji-recognition/char_sep_datas",
        help="Optional dataset root used to keep only Unicode classes that are actually present in the training data.",
    )
    parser.add_argument(
        "--pretrain-root",
        default=None,
        help="Optional pretraining dataset root (e.g. CASIA). Classes found here will also be kept in the final codebook.",
    )
    parser.add_argument(
        "--output",
        default="outputs/260828_codebook/final_codebook.pkl",
        help="Output path for the final Unicode codebook.",
    )
    return parser.parse_args()


def unicode_key_from_character(character: str) -> str:
    text = str(character).strip()
    if not text:
        return text
    if text.startswith("U+") or text.startswith("u+"):
        return text.upper()
    if len(text) == 1:
        return f"U+{ord(text):04X}".upper()
    return text


def load_radical_to_code(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        if "radical_to_code" in payload and isinstance(payload["radical_to_code"], dict):
            radical_to_code = payload["radical_to_code"]
        elif "records" in payload and isinstance(payload["records"], list):
            radical_to_code = {}
            for item in payload["records"]:
                radical = item.get("radical")
                code = item.get("fare_code")
                if radical is not None and code is not None:
                    radical_to_code[str(radical)] = [int(value) for value in code]
        else:
            raise ValueError(f"Unsupported radical code payload in {path}.")
    else:
        raise TypeError(f"Unsupported radical code file format: {type(payload)!r}")

    return {str(key): [int(value) for value in value_list] for key, value_list in radical_to_code.items()}


def stabilized_fallback_code(expression: str, dimension: int) -> np.ndarray:
    seed = 0
    for index, ch in enumerate(expression):
        seed += (index + 1) * ord(ch)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dimension).astype(np.float32)


def expression_to_vector(expression: str, radical_to_code: dict[str, list[int]], dimension: int) -> np.ndarray:
    chars = [char for char in expression if not char.isspace() and char not in STRUCTURE_SET]
    if not chars:
        return np.zeros(dimension, dtype=np.float32)

    vectors: list[np.ndarray] = []
    for char in chars:
        code = radical_to_code.get(char)
        if code is None:
            code = stabilized_fallback_code(char, dimension).tolist()
        vectors.append(np.asarray(code, dtype=np.float32))

    if len(vectors) == 1:
        return vectors[0].astype(np.float32)

    stacked = np.stack(vectors, axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def dataset_has_class(dataset_root: Path, unicode_key: str) -> bool:
    if not dataset_root.exists():
        return True
    for book_dir in dataset_root.iterdir():
        if not book_dir.is_dir():
            continue
        class_dir = book_dir / "characters" / unicode_key
        if class_dir.exists():
            return True
    return False


def collect_dataset_classes(dataset_root: Path) -> set[str]:
    available_classes: set[str] = set()
    if not dataset_root.exists():
        return available_classes

    for book_dir in dataset_root.iterdir():
        if not book_dir.is_dir():
            continue
        character_root = book_dir / "characters"
        if not character_root.exists():
            continue
        for class_dir in character_root.iterdir():
            if class_dir.is_dir():
                available_classes.add(unicode_key_from_character(class_dir.name))

    return available_classes


def collect_classes_from_arbitrary_dataset(dataset_root: Path) -> set[str]:
    """Scan any dataset tree for image files and extract candidate class tokens.

    This helps with datasets like CASIA where images are not organized under
    Book/characters/<U+XXXX> but use numeric or dataset-specific folder/filename labels.
    We extract filename stems and parent folder names, split tokens on common
    delimiters, and attempt to convert numeric/hex tokens to `U+XXXX` keys.
    """
    classes: set[str] = set()
    if not dataset_root.exists():
        return classes

    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in dataset_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in image_exts:
            continue

        tokens = [p.stem, p.parent.name]
        for parent in p.parents:
            if parent == dataset_root:
                break
            tokens.append(parent.name)

        for tok in tokens:
            if not tok:
                continue
            parts = re.split(r"[\.・_\- ]+", tok)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Direct character -> U+XXXX
                key = unicode_key_from_character(part)
                if key:
                    classes.add(key)
                    continue

                # decimal substring (e.g. CASIA sometimes uses numeric labels)
                m = re.search(r"(\d{3,6})", part)
                if m:
                    try:
                        cp = int(m.group(1), 10)
                        if 0 <= cp <= 0x10FFFF:
                            classes.add(f"U+{cp:04X}")
                            continue
                    except Exception:
                        pass

                # hex substring
                m2 = re.search(r"([0-9A-Fa-f]{4,6})", part)
                if m2:
                    try:
                        cp = int(m2.group(1), 16)
                        if 0 <= cp <= 0x10FFFF:
                            classes.add(f"U+{cp:04X}")
                            continue
                    except Exception:
                        pass

    return classes


def build_final_codebook(ids_files: list[str], radical_codes_path: Path, dataset_root: Path | None = None, pretrain_root: Path | None = None) -> dict[str, list[int]]:
    radical_to_code = load_radical_to_code(radical_codes_path)
    code_dim = max(len(code) for code in radical_to_code.values()) if radical_to_code else 64

    merged = merge_ids_entries(read_ids_entries([Path(path) for path in ids_files]))
    codebook: dict[str, list[int]] = {}
    # Collect classes from the main dataset and (optionally) a large pretrain dataset
    available_classes: set[str] = set()
    if dataset_root is not None:
        available_classes |= collect_dataset_classes(dataset_root)
    if pretrain_root is not None:
        # Try structured char_sep format first, fallback to arbitrary scan for datasets like CASIA
        pretrain_structured = collect_dataset_classes(pretrain_root)
        if pretrain_structured:
            available_classes |= pretrain_structured
        else:
            available_classes |= collect_classes_from_arbitrary_dataset(pretrain_root)

    # If a pretrain dataset root is provided, try to find additional candidate
    # Unicode keys from that dataset and add minimal entries for any classes
    # not present in the IDS merge. This allows CASIA-only characters to be
    # represented in the final codebook even when they are not listed in IDS.
    extra_candidates: set[str] = set()
    if pretrain_root is not None:
        extra_candidates = collect_classes_from_arbitrary_dataset(pretrain_root)
        # Convert U+XXXX strings to the single character form for use as "character"
        for cand in list(extra_candidates):
            if cand.startswith("U+"):
                try:
                    cp = int(cand[2:], 16)
                    ch = chr(cp)
                    # add the character form too
                    extra_candidates.add(ch)
                except Exception:
                    pass

    for character, expression in merged.items():
        key = unicode_key_from_character(character)
        if not key:
            continue

        # If we found any available classes from dataset(s), restrict to their union.
        if available_classes and key not in available_classes:
            continue

        vector = expression_to_vector(expression, radical_to_code, code_dim)
        codebook[key] = np.asarray(vector, dtype=np.float32).astype(float).tolist()

    # Add extra candidates discovered in pretrain data that were not
    # present in IDS merge. We build a fallback vector per-character.
    for cand in sorted(extra_candidates):
        key = unicode_key_from_character(cand)
        if not key or key in codebook:
            continue
        # Use the single-character expression which will fallback to a stable
        # randomized vector if radical codes are not found.
        expr = cand if len(cand) == 1 else ''
        if not expr:
            # try parse U+XXXX form
            if key.startswith("U+"):
                try:
                    cp = int(key[2:], 16)
                    expr = chr(cp)
                except Exception:
                    expr = ''
        vector = expression_to_vector(expr or cand, radical_to_code, code_dim)
        codebook[key] = np.asarray(vector, dtype=np.float32).astype(float).tolist()

    if not codebook:
        raise RuntimeError("No final codebook entries were generated. Check IDS files and radical code data.")

    return codebook


def main() -> None:
    args = parse_args()
    radical_codes_path = Path(args.radical_codes)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    pretrain_root = Path(args.pretrain_root) if args.pretrain_root else None

    final_codebook = build_final_codebook(args.ids_files, radical_codes_path, dataset_root, pretrain_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        pickle.dump(final_codebook, handle)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(final_codebook, handle, ensure_ascii=False, indent=2)

    print(f"Loaded IDS entries from {len(args.ids_files)} file(s).")
    print(f"Built final_codebook with {len(final_codebook)} Unicode classes.")
    print(f"Saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
