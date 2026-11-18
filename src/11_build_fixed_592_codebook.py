from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fare_pipeline import merge_ids_entries, read_ids_entries, STRUCTURE_SYMBOLS
from inspect_ids_depth import (
    character_from_key,
    collect_dataset_characters,
    effective_tree,
    fits_592_layout,
    load_eligible_radicals,
    unicode_key,
)
from structured_codebook import CODEBOOK_FORMAT, IDSNode, canonical_ids, ids_to_polish, parse_ids_expression

CODEBOOK_DIM = 592
ROOT_SLOT = 0
BRANCH_START = 4
BRANCH_WIDTH = 4 + 3 * 64
LEAF_START = 8
SINGLE_CODE = 1
OPERATOR_CODES = {symbol: index + 2 for index, symbol in enumerate(STRUCTURE_SYMBOLS)}
SYMBOL_CODES = {"NONE": 0, "SINGLE": SINGLE_CODE, **OPERATOR_CODES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="592-bit固定木構造のクラスcodebookを作成します。")
    parser.add_argument("--ids-files", nargs="+", default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"])
    parser.add_argument("--dataset-root", default="../kuzushiji-recognition/char_sep_datas")
    parser.add_argument("--class-union-manifest", default="outputs/261007_correct_dict/class_union_audit.json")
    parser.add_argument("--radical-codes", default="outputs/261018_cos_median_codebook/all_glyph_codes.pkl")
    parser.add_argument("--render-metadata", default="outputs/261007_correct_dict/combined_render_metadata.json")
    parser.add_argument("--image-root", default="outputs/261007_correct_dict")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output", default="outputs/261119_fixed_bit_ViT/fixed_592_codebook.pkl")
    return parser.parse_args()


def load_glyph_codes(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as handle:
        payload: Any = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported glyph-code payload: {path}")
    raw_codes = payload.get("radical_to_code")
    if not isinstance(raw_codes, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"No radical_to_code mapping or records found in {path}.")
        raw_codes = {
            str(record["radical"]): record["fare_code"]
            for record in records
            if isinstance(record, dict)
            and record.get("radical") is not None
            and record.get("fare_code") is not None
        }
    result: dict[str, list[int]] = {}
    for glyph, raw_vector in raw_codes.items():
        vector = [int(value) for value in raw_vector]
        if len(vector) != 64 or any(value not in {-1, 1} for value in vector):
            raise ValueError(f"Expected a 64-bit +/-1 code for {glyph!r} in {path}.")
        result[str(glyph)] = vector
    return result


def signed_bits(value: int) -> list[int]:
    return [1 if bit == "1" else -1 for bit in f"{value:04b}"]


def set_symbol(code: np.ndarray, mask: np.ndarray, offset: int, symbol: str) -> None:
    code[offset : offset + 4] = signed_bits(SYMBOL_CODES[symbol])
    mask[offset : offset + 4] = True


def subtree_code(
    node: IDSNode,
    glyph_codes: dict[str, list[int]],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> list[int] | None:
    if node.symbol not in STRUCTURE_SYMBOLS and node.symbol in glyph_codes:
        return glyph_codes[node.symbol]
    candidates = reverse_subtrees.get(canonical_ids(node), [])
    return glyph_codes[candidates[0]] if candidates else None


def encode_fixed_tree(
    node: IDSNode,
    glyph_codes: dict[str, list[int]],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> tuple[list[int] | None, list[bool] | None, str | None]:
    code = np.zeros(CODEBOOK_DIM, dtype=np.int8)
    mask = np.zeros(CODEBOOK_DIM, dtype=np.bool_)
    if node.symbol not in STRUCTURE_SYMBOLS:
        glyph = subtree_code(node, glyph_codes, reverse_subtrees)
        if glyph is None:
            return None, None, f"SINGLEに必要な直接64-bitコードがありません: {node.symbol!r}"
        set_symbol(code, mask, ROOT_SLOT, "SINGLE")
        for branch_index in range(3):
            set_symbol(code, mask, BRANCH_START + branch_index * BRANCH_WIDTH, "NONE")
        code[LEAF_START : LEAF_START + 64] = glyph
        mask[LEAF_START : LEAF_START + 64] = True
        return code.tolist(), mask.tolist(), None

    set_symbol(code, mask, ROOT_SLOT, node.symbol)
    for branch_index in range(3):
        branch_offset = BRANCH_START + branch_index * BRANCH_WIDTH
        leaf_offset = branch_offset + 4
        if branch_index >= len(node.children):
            set_symbol(code, mask, branch_offset, "NONE")
            continue

        branch = node.children[branch_index]
        leaf_codes: list[list[int]] | None = None
        if branch.symbol in STRUCTURE_SYMBOLS:
            candidate_leaves = [subtree_code(child, glyph_codes, reverse_subtrees) for child in branch.children]
            if all(candidate is not None for candidate in candidate_leaves):
                set_symbol(code, mask, branch_offset, branch.symbol)
                leaf_codes = [candidate for candidate in candidate_leaves if candidate is not None]
            else:
                replacement = subtree_code(branch, glyph_codes, reverse_subtrees)
                if replacement is not None:
                    set_symbol(code, mask, branch_offset, "NONE")
                    leaf_codes = [replacement]
                else:
                    return None, None, (
                        f"branch {branch_index} ({ids_to_polish(branch)})は深さ1の演算子枠に収まらず、"
                        "子部分木を葉コードへ置換する登録glyph codeもありません"
                    )
        else:
            replacement = subtree_code(branch, glyph_codes, reverse_subtrees)
            if replacement is None:
                return None, None, f"branch {branch_index} の末端に64-bitコードがありません: {branch.symbol!r}"
            set_symbol(code, mask, branch_offset, "NONE")
            leaf_codes = [replacement]

        for leaf_index, glyph in enumerate(leaf_codes):
            start = leaf_offset + leaf_index * 64
            code[start : start + 64] = glyph
            mask[start : start + 64] = True

    if int(mask.sum()) == 0 or len(code) != CODEBOOK_DIM:
        raise AssertionError("Malformed fixed 592-bit code.")
    return code.tolist(), mask.tolist(), None


def build_codebook(args: argparse.Namespace) -> dict[str, Any]:
    glyph_codes_all = load_glyph_codes(Path(args.radical_codes))
    eligible = load_eligible_radicals(
        Path(args.radical_codes), Path(args.render_metadata), Path(args.image_root)
    )
    glyph_codes = {glyph: vector for glyph, vector in glyph_codes_all.items() if glyph in eligible}

    union_payload = json.loads(Path(args.class_union_manifest).read_text(encoding="utf-8"))
    union_classes = union_payload.get("classes")
    if not isinstance(union_classes, dict):
        raise ValueError("Class union manifest must contain a 'classes' mapping.")

    dataset_characters = collect_dataset_characters(Path(args.dataset_root))
    dataset_keys = {unicode_key(character) for character in dataset_characters}
    ids_files = [Path(path) for path in args.ids_files]
    merged = merge_ids_entries(read_ids_entries(ids_files))
    parsed_ids: dict[str, IDSNode] = {}
    parse_errors: dict[str, str] = {}
    for character, expression in merged.items():
        try:
            parsed_ids[character] = parse_ids_expression(expression)
        except ValueError as exc:
            parse_errors[character] = str(exc)

    reverse_subtrees: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for character, tree in parsed_ids.items():
        if character in glyph_codes:
            reverse_subtrees[canonical_ids(tree)].append(character)
    for candidates in reverse_subtrees.values():
        candidates.sort(key=lambda item: (len(item), ord(item[0]) if item else 0, item))

    excluded: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    effective_cache: dict[str, tuple[IDSNode, bool]] = {}
    dataset_failure_keys: set[str] = set()

    for character in sorted(dataset_characters, key=ord):
        tree = parsed_ids.get(character, IDSNode(character))
        effective, resolved = effective_tree(
            tree, 1, args.max_depth, set(glyph_codes), parsed_ids, reverse_subtrees
        )
        effective_cache[character] = (effective, resolved)
        if not (resolved and fits_592_layout(effective, set(glyph_codes), reverse_subtrees)):
            dataset_failure_keys.add(unicode_key(character))

    missing_dataset_union = dataset_keys - set(union_classes)
    if missing_dataset_union:
        examples = ", ".join(sorted(missing_dataset_union)[:10])
        raise ValueError(f"Class union manifest omits {len(missing_dataset_union)} Kuzushiji classes: {examples}")

    for unicode_class, provenance in sorted(union_classes.items()):
        character = str(provenance.get("character", "")) if isinstance(provenance, dict) else ""
        character = character if len(character) == 1 else character_from_key(str(unicode_class))
        if character is None or len(character) != 1:
            excluded.append({"unicode": unicode_class, "reason": "invalid_unicode_class_key"})
            continue

        is_kuzushiji = unicode_class in dataset_keys
        if is_kuzushiji and unicode_class in dataset_failure_keys:
            original = parsed_ids.get(character, IDSNode(character))
            effective, resolved = effective_cache[character]
            excluded.append(
                {
                    "unicode": unicode_class,
                    "character": character,
                    "ids": ids_to_polish(original),
                    "effective_ids": ids_to_polish(effective),
                    "reason": "kuzushiji_class_fails_strict_592_layout",
                    "terminals_resolved": resolved,
                }
            )
            continue

        original = parsed_ids.get(character, IDSNode(character))
        effective, resolved = effective_cache.get(character, (None, False))
        if effective is None:
            effective, resolved = effective_tree(
                original, 1, args.max_depth, set(glyph_codes), parsed_ids, reverse_subtrees
            )
        if resolved and fits_592_layout(effective, set(glyph_codes), reverse_subtrees):
            code, mask, error = encode_fixed_tree(effective, glyph_codes, reverse_subtrees)
        else:
            code, mask = None, None
            error = "IDS展開後の木が固定592-bit配置に適合しません"

        if code is None or mask is None:
            excluded.append(
                {
                    "unicode": unicode_class,
                    "character": character,
                    "ids": ids_to_polish(original),
                    "effective_ids": ids_to_polish(effective),
                    "reason": error or "unknown_encoding_failure",
                    "source": provenance.get("source") if isinstance(provenance, dict) else None,
                }
            )
            continue

        source = provenance.get("source", {}) if isinstance(provenance, dict) else {}
        entries[unicode_class] = {
            "code": code,
            "mask": mask,
            "metadata": {
                "character": character,
                "ids": merged.get(character),
                "polish": ids_to_polish(original),
                "effective_ids": ids_to_polish(effective),
                "source": source,
                "casia_train_samples": provenance.get("casia_train_samples") if isinstance(provenance, dict) else None,
                "casia_test_samples": provenance.get("casia_test_samples") if isinstance(provenance, dict) else None,
                "valid_bit_count": int(sum(mask)),
            },
        }

    if not entries:
        raise RuntimeError("592-bit codebook contains no encodable classes.")

    codes = {key: item["code"] for key, item in entries.items()}
    masks = {key: item["mask"] for key, item in entries.items()}
    class_metadata = {key: item["metadata"] for key, item in entries.items()}
    kuzushiji_included = sum(bool(meta["source"].get("kuzushiji")) for meta in class_metadata.values())
    casia_included = sum(bool(meta["source"].get("casia")) for meta in class_metadata.values())
    casia_only_included = sum(
        bool(meta["source"].get("casia")) and not bool(meta["source"].get("kuzushiji"))
        for meta in class_metadata.values()
    )
    payload = {
        "format": CODEBOOK_FORMAT,
        "schema_version": 1,
        "layout_format": "fixed-592-tree-v1",
        "codebook_dim": CODEBOOK_DIM,
        "ids_files": [path.as_posix() for path in ids_files],
        "radical_codes_path": Path(args.radical_codes).as_posix(),
        "render_metadata_path": Path(args.render_metadata).as_posix(),
        "class_union_manifest": Path(args.class_union_manifest).as_posix(),
        "structure_symbols_order": list(STRUCTURE_SYMBOLS),
        "symbol_codes_unsigned": SYMBOL_CODES,
        "symbol_codes_signed_bits": {name: signed_bits(value) for name, value in SYMBOL_CODES.items()},
        "slot_layout": {
            "root_symbol_bits": [0, 4],
            "branch_symbol_bits": [[4 + index * BRANCH_WIDTH, 8 + index * BRANCH_WIDTH] for index in range(3)],
            "leaf_code_ranges": [
                [LEAF_START + branch * BRANCH_WIDTH + leaf * 64,
                 LEAF_START + branch * BRANCH_WIDTH + (leaf + 1) * 64]
                for branch in range(3)
                for leaf in range(3)
            ],
            "branch_width": BRANCH_WIDTH,
            "leaf_code_width": 64,
            "unused_leaf_value": 0,
            "mask_semantics": "true_for_symbol_bits_and_present_leaf_code_bits",
        },
        "padding_value": 0,
        "padding_mask_semantics": "true_is_valid_code_bit",
        "score_normalization": "masked_dot_product_scaled_to_64_valid_bits",
        "max_code_length": CODEBOOK_DIM,
        "codes": codes,
        "masks": masks,
        "lengths": {key: item["metadata"]["valid_bit_count"] for key, item in entries.items()},
        "entries": class_metadata,
        "diagnostics": {
            "excluded_classes": excluded,
            "ids_parse_errors": parse_errors,
        },
        "build": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_union_classes": len(union_classes),
            "kuzushiji_dataset_classes": len(dataset_keys),
            "kuzushiji_strict_592_excluded": len(dataset_failure_keys),
            "included_classes": len(entries),
            "included_kuzushiji_classes": kuzushiji_included,
            "included_casia_classes": casia_included,
            "included_casia_only_classes": casia_only_included,
            "excluded_classes": len(excluded),
            "registered_rendered_glyph_codes": len(glyph_codes),
            "unresolved_ids_classes": sum(not item["terminals_resolved"] for item in excluded if "terminals_resolved" in item),
            "excluded_kuzushiji_unicode_keys": sorted(dataset_failure_keys),
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = build_codebook(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)
    audit_payload = {
        key: value for key, value in payload.items() if key not in {"codes", "masks"}
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_path = output_path.with_name("codebook_exclusions.json")
    audit_path.write_text(
        json.dumps(payload["diagnostics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "固定592-bit codebook作成完了: "
        f"全体={payload['build']['included_classes']}, "
        f"Kuzushiji={payload['build']['included_kuzushiji_classes']}, "
        f"CASIAのみ={payload['build']['included_casia_only_classes']}, "
        f"除外={payload['build']['excluded_classes']}"
    )
    print(f"保存先: {output_path.resolve()}")
    print(f"除外一覧: {audit_path.resolve()}")


if __name__ == "__main__":
    main()
