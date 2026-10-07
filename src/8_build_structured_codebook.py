from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from fare_pipeline import merge_ids_entries, read_ids_entries
from structured_codebook import (
    CODEBOOK_FORMAT,
    STRUCTURE_BITS,
    STRUCTURE_SET,
    IDSNode,
    canonical_ids,
    ids_to_polish,
    parse_ids_expression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a masked, fixed-width FaRE codebook from IDS preorder and rendered radical codes."
    )
    parser.add_argument(
        "--ids-files",
        nargs="+",
        default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"],
        help="IDS source files.",
    )
    parser.add_argument("--radical-codes", default="outputs/radical_codes.pkl")
    parser.add_argument("--render-metadata", default="outputs/radical_render_metadata.json")
    parser.add_argument("--render-manifest", default="outputs/radical_render_manifest.pkl")
    parser.add_argument("--image-root", default="outputs/radical_images")
    parser.add_argument(
        "--class-union-manifest",
        default=None,
        help="Audit JSON produced by 9_prepare_correct_codebook_assets.py; its classes define the complete source union.",
    )
    parser.add_argument(
        "--class-source-codebook",
        default=None,
        help="Legacy optional source of Unicode classes. Prefer --class-union-manifest.",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional structured dataset root to include classes from.")
    parser.add_argument("--pretrain-root", default=None, help="Optional pretraining dataset root to include classes from.")
    parser.add_argument(
        "--output",
        default="outputs/261005_structured_codebook/final_structured_codebook.pkl",
        help="Output pickle path; a JSON companion is written beside it.",
    )
    parser.add_argument(
        "--log-path",
        default="outputs/261005_structured_codebook/codebook_build.log",
        help="Append stdout diagnostics to this log file.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum IDS tree depth, with the root at depth 1.",
    )
    parser.add_argument(
        "--codebook-dim",
        type=int,
        default=1212,
        help="Fixed padded output dimension. Any class exceeding this length aborts generation and is listed.",
    )
    parser.add_argument(
        "--unresolved-policy",
        choices=("exclude", "random"),
        default="exclude",
        help="Exclude unresolved classes, or explicitly assign deterministic random terminal codes.",
    )
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify that metadata-marked rendered image files exist and can be opened.",
    )
    return parser.parse_args()


class Tee:
    def __init__(self, stream: Any, handle: Any) -> None:
        self.stream = stream
        self.handle = handle

    def write(self, text: str) -> int:
        self.handle.write(text)
        self.handle.flush()
        try:
            return self.stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            printable = text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
            return self.stream.write(printable)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


def emit(message: str) -> None:
    print(message, flush=True)


def unicode_key(character: str) -> str:
    value = str(character).strip()
    if value.startswith(("U+", "u+")):
        return value.upper()
    if len(value) == 1:
        return f"U+{ord(value):04X}"
    return value


def character_from_key(key: str) -> str | None:
    normalized = str(key).strip()
    if normalized.startswith("U+"):
        try:
            return chr(int(normalized[2:], 16))
        except (ValueError, OverflowError):
            return None
    return normalized if len(normalized) == 1 else None


def load_radical_codes(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported radical code payload in {path}: {type(payload)!r}")

    raw_codes = payload.get("radical_to_code")
    if not isinstance(raw_codes, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"No radical_to_code or records found in {path}.")
        raw_codes = {
            str(record["radical"]): record["fare_code"]
            for record in records
            if isinstance(record, dict) and record.get("radical") is not None and record.get("fare_code") is not None
        }

    codes: dict[str, list[int]] = {}
    for radical, vector in raw_codes.items():
        values = [int(value) for value in vector]
        if len(values) != 64 or any(value not in {-1, 1} for value in values):
            raise ValueError(f"Expected a 64-bit +/-1 FaRE code for {radical!r} in {path}.")
        codes[str(radical)] = values
    return codes


def load_render_status(metadata_path: Path, manifest_path: Path, image_root: Path, verify_images: bool) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported render metadata schema in {metadata_path}.")
    with manifest_path.open("rb") as handle:
        manifest = pickle.load(handle)

    manifest_entries = manifest.get("rendered", []) if isinstance(manifest, dict) else []
    manifest_by_text = {str(item.get("radical")): item for item in manifest_entries if isinstance(item, dict)}
    status_by_text: dict[str, dict[str, Any]] = {}
    for entry in metadata.get("entries", []):
        text = str(entry.get("text", ""))
        if not text:
            continue
        status = dict(entry)
        manifest_entry = manifest_by_text.get(text)
        if manifest_entry is None:
            status["status"] = "missing_manifest_entry"
            status["renderable"] = False
        else:
            for field in ("status", "renderable", "image_path"):
                if field in manifest_entry and manifest_entry[field] != status.get(field):
                    raise ValueError(f"Render metadata/manifest disagree for {text!r} on {field}.")

        image_path = status.get("image_path")
        if status.get("status") == "rendered" and status.get("renderable") and image_path:
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            if not resolved.is_file():
                status["status"] = "image_missing"
                status["renderable"] = False
            elif verify_images:
                try:
                    with Image.open(resolved) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError):
                    status["status"] = "image_unreadable"
                    status["renderable"] = False
        status_by_text[text] = status

    if len(status_by_text) != len(metadata.get("entries", [])):
        raise ValueError("Render metadata has duplicate or empty text entries.")
    emit(
        "Render input check: "
        f"metadata_entries={len(status_by_text)} manifest_entries={len(manifest_by_text)} "
        f"reported_summary={metadata.get('summary', {})} image_root={image_root.resolve()}"
    )
    return status_by_text, metadata


def read_class_source(path: Path) -> set[str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and payload.get("format") == CODEBOOK_FORMAT:
        values = payload.get("codes", {})
    elif isinstance(payload, dict) and isinstance(payload.get("codebook"), dict):
        values = payload["codebook"]
    elif isinstance(payload, dict):
        values = payload
    else:
        raise TypeError(f"Unsupported class source codebook in {path}.")
    return {str(key) for key in values if character_from_key(str(key)) is not None}


def collect_structured_dataset_classes(root: Path) -> set[str]:
    classes: set[str] = set()
    if not root.exists():
        return classes
    for book_dir in root.iterdir():
        character_root = book_dir / "characters"
        if not book_dir.is_dir() or not character_root.is_dir():
            continue
        classes.update(unicode_key(class_dir.name) for class_dir in character_root.iterdir() if class_dir.is_dir())
    return classes


def collect_pretrain_classes(root: Path) -> set[str]:
    classes: set[str] = set()
    if not root.exists():
        return classes
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        candidates = [path.parent.name, *[parent.name for parent in list(path.parents)[:5]]]
        candidates.append(path.stem)
        for candidate in candidates:
            for token in re.split(r"[\.・_\- ]+", candidate):
                token = token.strip()
                if len(token) == 1 and not token.isdigit():
                    classes.add(unicode_key(token))
                    continue
                decimal = re.search(r"(?<!\d)(\d{4,6})(?!\d)", token)
                if decimal:
                    codepoint = int(decimal.group(1), 10)
                    if 0 <= codepoint <= 0x10FFFF:
                        classes.add(f"U+{codepoint:04X}")
                hexadecimal = re.search(r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{4,6})(?![0-9A-Fa-f])", token)
                if hexadecimal:
                    codepoint = int(hexadecimal.group(1), 16)
                    if 0 <= codepoint <= 0x10FFFF:
                        classes.add(f"U+{codepoint:04X}")
    return classes


def deterministic_random_code(token: str) -> list[int]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:64]
    return [1 if bit else -1 for bit in bits.tolist()]


def build_codebook(args: argparse.Namespace) -> dict[str, Any]:
    radical_codes = load_radical_codes(Path(args.radical_codes))
    render_status, render_metadata = load_render_status(
        Path(args.render_metadata), Path(args.render_manifest), Path(args.image_root), args.verify_images
    )
    eligible_codes: dict[str, list[int]] = {}
    for text, vector in radical_codes.items():
        status = render_status.get(text)
        if status and status.get("status") == "rendered" and status.get("renderable"):
            eligible_codes[text] = vector

    if not eligible_codes:
        raise ValueError("No 64-bit radical codes have a matching successfully rendered image.")
    code_status_counts = Counter(
        render_status.get(text, {}).get("status", "not_in_metadata") for text in radical_codes
    )
    emit(f"Usable image-derived radical codes: {len(eligible_codes)} / {len(radical_codes)}; states={dict(code_status_counts)}")

    ids_files = [Path(path) for path in args.ids_files]
    merged = merge_ids_entries(read_ids_entries(ids_files))
    selected_classes: set[str] = set()
    class_provenance: dict[str, dict[str, Any]] = {}
    if args.class_union_manifest:
        union_audit = json.loads(Path(args.class_union_manifest).read_text(encoding="utf-8"))
        class_provenance = {str(key): value for key, value in union_audit.get("classes", {}).items()}
        selected_classes = set(class_provenance)
    elif args.class_source_codebook:
        selected_classes |= read_class_source(Path(args.class_source_codebook))
    if args.dataset_root:
        selected_classes |= collect_structured_dataset_classes(Path(args.dataset_root))
    if args.pretrain_root:
        selected_classes |= collect_pretrain_classes(Path(args.pretrain_root))

    if not selected_classes:
        selected_classes = {unicode_key(character) for character in merged}
    class_entries = {
        unicode_key(character): (character, expression)
        for character, expression in merged.items()
        if unicode_key(character) in selected_classes
    }
    if class_provenance:
        for key in sorted(selected_classes - set(class_entries)):
            item = class_provenance[key]
            character = str(item.get("character", ""))
            if len(character) == 1:
                class_entries[key] = (character, character)

    parsed_ids: dict[str, IDSNode] = {}
    parse_errors: list[dict[str, str]] = []
    for character, expression in merged.items():
        try:
            parsed_ids[character] = parse_ids_expression(expression)
        except ValueError as exc:
            issue = {"character": character, "expression": expression, "error": str(exc)}
            parse_errors.append(issue)
            emit(f"[ids_parse_error] {json.dumps(issue, ensure_ascii=True, sort_keys=True)}")

    reverse_subtrees: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for character, tree in parsed_ids.items():
        if character in eligible_codes:
            reverse_subtrees[canonical_ids(tree)].append(character)
    for candidates in reverse_subtrees.values():
        candidates.sort(key=lambda item: (len(item), ord(item[0]) if item else 0, item))

    diagnostic_counts: Counter[str] = Counter()
    diagnostic_events: list[dict[str, Any]] = []
    resolved_vectors: dict[str, list[int]] = {}
    class_metadata: dict[str, dict[str, Any]] = {}

    def event(kind: str, owner: str, node: IDSNode, detail: dict[str, Any]) -> None:
        item = {
            "kind": kind,
            "owner": owner,
            "owner_unicode": unicode_key(owner),
            "subtree": ids_to_polish(node),
            **detail,
        }
        diagnostic_events.append(item)
        diagnostic_counts[kind] += 1
        if kind in {
            "unregistered_component",
            "render_failure",
            "component_decomposition",
            "subtree_replacement",
            "random_code",
        }:
            emit(
                f"[{kind}] owner={owner} ({unicode_key(owner)}) subtree={item['subtree']} "
                f"details={json.dumps(detail, ensure_ascii=True, sort_keys=True)}"
            )

    def replacement_for(node: IDSNode) -> str | None:
        candidates = reverse_subtrees.get(canonical_ids(node), [])
        return candidates[0] if candidates else None

    def encode_node(node: IDSNode, owner: str, depth: int) -> tuple[list[int] | None, str | None]:
        if node.symbol not in STRUCTURE_SET:
            vector = eligible_codes.get(node.symbol)
            if vector is not None:
                return vector, None
            status = render_status.get(node.symbol, {})
            failure_kind = "render_failure" if status and status.get("status") != "rendered" else "unregistered_component"
            event(
                failure_kind,
                owner,
                node,
                {
                    "component": node.symbol,
                    "component_unicode": unicode_key(node.symbol) if len(node.symbol) == 1 else None,
                    "render_status": status.get("status", "not_in_metadata"),
                    "font_path": status.get("font_path"),
                    "image_path": status.get("image_path"),
                },
            )
            if depth < args.max_depth and node.symbol in parsed_ids:
                expansion = parsed_ids[node.symbol]
                if expansion.symbol != node.symbol:
                    expanded, _ = encode_node(expansion, owner, depth + 1)
                    if expanded is not None:
                        event("component_decomposition", owner, node, {"component": node.symbol, "expanded_as": ids_to_polish(expansion)})
                        return expanded, None
            return None, f"no registered image-derived code for token {node.symbol!r}"

        if depth >= args.max_depth:
            candidate = replacement_for(node)
            if candidate is not None:
                event("subtree_replacement", owner, node, {"replacement": candidate, "replacement_unicode": unicode_key(candidate), "reason": "max_depth"})
                return eligible_codes[candidate], None
            return None, f"depth {depth} reached and no registered rendered character matches subtree"

        child_vectors: list[int] = []
        for child in node.children:
            encoded_child, error = encode_node(child, owner, depth + 1)
            if encoded_child is None:
                candidate = replacement_for(node)
                if candidate is not None:
                    event(
                        "subtree_replacement",
                        owner,
                        node,
                        {
                            "replacement": candidate,
                            "replacement_unicode": unicode_key(candidate),
                            "reason": error,
                            "replaced_components": [ids_to_polish(node)],
                        },
                    )
                    return eligible_codes[candidate], None
                return None, error
            child_vectors.extend(encoded_child)
        return list(STRUCTURE_BITS[node.symbol]) + child_vectors, None

    unresolved: list[dict[str, str]] = []
    unresolved_excluded = 0
    all_class_entries = dict(class_entries)
    for unicode_class in sorted(selected_classes - set(class_entries)):
        character = character_from_key(unicode_class)
        if character is not None:
            all_class_entries[unicode_class] = (character, character)

    for unicode_class, (character, expression) in all_class_entries.items():
        tree = IDSNode(character)
        try:
            tree = parsed_ids.get(character)
            if tree is None:
                if character in eligible_codes:
                    tree = IDSNode(character)
                else:
                    tree = parse_ids_expression(expression)
            vector, error = encode_node(tree, character, 1)
        except ValueError as exc:
            vector, error = None, str(exc)
        if vector is None:
            unresolved_item = {
                "character": character,
                "unicode": unicode_class,
                "ids": expression,
                "reason": str(error),
                "disposition": "random_fallback" if args.unresolved_policy == "random" else "excluded",
            }
            unresolved.append(unresolved_item)
            if args.unresolved_policy == "random":
                diagnostic_counts["random_fallback_class"] += 1
            else:
                diagnostic_counts["unresolved_class"] += 1
                unresolved_excluded += 1
            emit(f"[unresolved_class] {character} ({unicode_class}) IDS={expression!r}: {error}")
            if args.unresolved_policy == "random":
                vector = deterministic_random_code(f"unresolved-class|{unicode_class}|{expression}")
                event(
                    "random_code",
                    character,
                    tree,
                    {"unicode": unicode_class, "reason": str(error), "policy": "explicit_unresolved_policy_random"},
                )
                resolved_vectors[unicode_class] = vector
                class_metadata[unicode_class] = {
                    "character": character,
                    "ids": expression,
                    "polish": expression,
                    "bit_length": len(vector),
                    "source": "explicit_random_fallback",
                }
            continue
        resolved_vectors[unicode_class] = vector
        class_metadata[unicode_class] = {
            "character": character,
            "ids": expression,
            "polish": ids_to_polish(tree),
            "bit_length": len(vector),
            "source": class_provenance.get(unicode_class, {}).get("source"),
            "casia_train_samples": class_provenance.get(unicode_class, {}).get("casia_train_samples"),
            "casia_test_samples": class_provenance.get(unicode_class, {}).get("casia_test_samples"),
        }

    if not resolved_vectors:
        raise RuntimeError("No classes could be encoded. Review IDS, render metadata, and radical codes.")

    observed_max_code_length = max(map(len, resolved_vectors.values()))
    overlength = [
        {"unicode": key, "character": class_metadata[key]["character"], "bit_length": len(vector), "ids": class_metadata[key]["ids"]}
        for key, vector in resolved_vectors.items()
        if len(vector) > args.codebook_dim
    ]
    if overlength:
        overlength_path = Path(args.output).with_name("codebook_overlength_classes.json")
        overlength_path.parent.mkdir(parents=True, exist_ok=True)
        overlength_path.write_text(json.dumps(overlength, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(
            f"{len(overlength)} classes exceed the fixed {args.codebook_dim}-bit limit "
            f"(observed maximum {observed_max_code_length}); details saved to {overlength_path}."
        )
    max_code_length = args.codebook_dim
    codes: dict[str, list[int]] = {}
    masks: dict[str, list[bool]] = {}
    for key, vector in resolved_vectors.items():
        pad_length = max_code_length - len(vector)
        codes[key] = vector + [0] * pad_length
        masks[key] = [True] * len(vector) + [False] * pad_length

    payload = {
        "format": CODEBOOK_FORMAT,
        "schema_version": 1,
        "ids_files": [path.as_posix() for path in ids_files],
        "radical_codes_path": Path(args.radical_codes).as_posix(),
        "render_metadata_path": Path(args.render_metadata).as_posix(),
        "render_manifest_path": Path(args.render_manifest).as_posix(),
        "render_font_info": render_metadata.get("run", {}),
        "structure_bits": {symbol: list(bits) for symbol, bits in STRUCTURE_BITS.items()},
        "max_depth": args.max_depth,
        "depth_convention": "root_is_depth_1; nodes_at_max_depth_are_encoded_as_registered_terminal_subtrees",
        "radical_code_dim": 64,
        "padding_value": 0,
        "padding_mask_semantics": "true_is_valid_code_bit",
        "score_normalization": "masked_dot_product_scaled_to_64_valid_bits",
        "max_code_length": max_code_length,
        "observed_max_code_length": observed_max_code_length,
        "codes": codes,
        "masks": masks,
        "lengths": {key: len(value) for key, value in resolved_vectors.items()},
        "entries": class_metadata,
        "diagnostics": {
            "counts": dict(diagnostic_counts),
            "events": diagnostic_events,
            "unresolved_classes": unresolved,
            "ids_parse_errors": parse_errors,
        },
        "build": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_classes": len(selected_classes),
            "class_union_manifest": str(Path(args.class_union_manifest).as_posix()) if args.class_union_manifest else None,
            "codebook_dimension": args.codebook_dim,
            "observed_max_code_length": observed_max_code_length,
            "resolved_classes": len(codes),
            "unresolved_classes": unresolved_excluded,
            "random_fallback_classes": diagnostic_counts.get("random_fallback_class", 0),
            "registered_rendered_radicals": len(eligible_codes),
            "unresolved_policy": args.unresolved_policy,
        },
    }
    emit(
        "CodeBook summary: "
        f"requested_classes={len(selected_classes)} resolved={len(codes)} unresolved={len(unresolved)} "
        f"max_code_length={max_code_length} diagnostics={dict(diagnostic_counts)}"
    )
    return payload


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_stdout, handle), Tee(original_stderr, handle)
        try:
            emit(f"Structured CodeBook build started; log={log_path.resolve()}")
            payload = build_codebook(args)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as output_handle:
                pickle.dump(payload, output_handle)
            json_path = output_path.with_suffix(".json")
            with json_path.open("w", encoding="utf-8") as output_handle:
                json.dump(payload, output_handle, ensure_ascii=False, indent=2)
            emit(f"Saved structured CodeBook pickle: {output_path.resolve()}")
            emit(f"Saved structured CodeBook JSON: {json_path.resolve()}")
        except Exception as exc:
            emit(f"CodeBook build failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()