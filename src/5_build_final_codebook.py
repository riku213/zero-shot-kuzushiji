from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from fare_pipeline import STRUCTURE_SET, merge_ids_entries, read_ids_entries


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


def build_final_codebook(ids_files: list[str], radical_codes_path: Path, dataset_root: Path | None = None) -> dict[str, list[int]]:
    radical_to_code = load_radical_to_code(radical_codes_path)
    code_dim = max(len(code) for code in radical_to_code.values()) if radical_to_code else 64

    merged = merge_ids_entries(read_ids_entries([Path(path) for path in ids_files]))
    codebook: dict[str, list[int]] = {}
    available_classes = collect_dataset_classes(dataset_root) if dataset_root is not None else set()

    for character, expression in merged.items():
        key = unicode_key_from_character(character)
        if not key:
            continue

        if dataset_root is not None and available_classes and key not in available_classes:
            continue

        vector = expression_to_vector(expression, radical_to_code, code_dim)
        codebook[key] = np.asarray(vector, dtype=np.float32).astype(float).tolist()

    if not codebook:
        raise RuntimeError("No final codebook entries were generated. Check IDS files and radical code data.")

    return codebook


def main() -> None:
    args = parse_args()
    radical_codes_path = Path(args.radical_codes)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    final_codebook = build_final_codebook(args.ids_files, radical_codes_path, dataset_root)
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
