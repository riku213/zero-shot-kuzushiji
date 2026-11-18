from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fare_pipeline import merge_ids_entries, read_ids_entries
from structured_codebook import (
    IDSNode,
    STRUCTURE_ARITY,
    STRUCTURE_SET,
    canonical_ids,
    ids_to_polish,
    parse_ids_expression,
    tree_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count effective IDS tree depths for Kuzushiji dataset classes."
    )
    parser.add_argument(
        "--ids-files",
        nargs="+",
        default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"],
        help="IDS source files used by the structured codebook builder.",
    )
    parser.add_argument(
        "--dataset-root",
        default="../kuzushiji-recognition/char_sep_datas",
        help="Kuzushiji dataset root containing book/characters/U+XXXX directories.",
    )
    parser.add_argument(
        "--radical-codes",
        default="outputs/261018_cos_median_codebook/all_glyph_codes.pkl",
        help="64-bit glyph-code dictionary used as terminal codes.",
    )
    parser.add_argument(
        "--render-metadata",
        default="outputs/261007_correct_dict/combined_render_metadata.json",
    )
    parser.add_argument("--image-root", default="outputs/261007_correct_dict")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Match the structured codebook builder's recursion limit (root depth is 1).",
    )
    parser.add_argument("--samples", type=int, default=20, help="Maximum depth-4+ examples to print.")
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional Markdown path for a per-class fixed 592-bit layout failure report.",
    )
    return parser.parse_args()


def unicode_key(value: str) -> str:
    text = str(value).strip()
    if text.startswith(("U+", "u+")):
        return text.upper()
    if len(text) == 1:
        return f"U+{ord(text):04X}"
    return text


def character_from_key(key: str) -> str | None:
    text = key.strip()
    if text.startswith("U+"):
        try:
            return chr(int(text[2:], 16))
        except (ValueError, OverflowError):
            return None
    return text if len(text) == 1 else None


def collect_dataset_characters(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Kuzushiji dataset root not found: {root}")
    characters: set[str] = set()
    for book_dir in root.iterdir():
        character_root = book_dir / "characters"
        if not book_dir.is_dir() or not character_root.is_dir():
            continue
        for class_dir in character_root.iterdir():
            if class_dir.is_dir():
                character = character_from_key(unicode_key(class_dir.name))
                if character is not None:
                    characters.add(character)
    return characters


def load_eligible_radicals(codes_path: Path, metadata_path: Path, image_root: Path) -> set[str]:
    with codes_path.open("rb") as handle:
        payload: Any = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported radical-code payload in {codes_path}.")
    raw_codes = payload.get("radical_to_code")
    if not isinstance(raw_codes, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"No radical_to_code mapping or records found in {codes_path}.")
        raw_codes = {
            str(record["radical"]): record["fare_code"]
            for record in records
            if isinstance(record, dict)
            and record.get("radical") is not None
            and record.get("fare_code") is not None
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported render metadata schema in {metadata_path}.")
    statuses = {
        str(entry.get("text", "")): entry
        for entry in metadata.get("entries", [])
        if isinstance(entry, dict) and entry.get("text")
    }

    eligible: set[str] = set()
    for radical, vector in raw_codes.items():
        values = [int(value) for value in vector]
        if len(values) != 64 or any(value not in {-1, 1} for value in values):
            continue
        status = statuses.get(str(radical), {})
        image_path = status.get("image_path")
        if status.get("status") != "rendered" or not status.get("renderable") or not image_path:
            continue
        resolved_image = Path(image_path)
        if not resolved_image.is_absolute():
            resolved_image = Path.cwd() / resolved_image
            if not resolved_image.is_file():
                resolved_image = image_root / Path(image_path).name
        if resolved_image.is_file():
            eligible.add(str(radical))
    return eligible


def count_split_nodes(node: IDSNode) -> Counter[int]:
    counts: Counter[int] = Counter()
    if node.symbol in STRUCTURE_SET:
        counts[STRUCTURE_ARITY[node.symbol]] += 1
    for child in node.children:
        counts.update(count_split_nodes(child))
    return counts


def code_length(node: IDSNode) -> int:
    if node.symbol in STRUCTURE_SET:
        return 4 + sum(code_length(child) for child in node.children)
    return 64


def count_node_types(node: IDSNode) -> tuple[int, int]:
    if node.symbol not in STRUCTURE_SET:
        return 0, 1
    structure_count = 1
    leaf_count = 0
    for child in node.children:
        child_structures, child_leaves = count_node_types(child)
        structure_count += child_structures
        leaf_count += child_leaves
    return structure_count, leaf_count


def max_structure_depth(node: IDSNode, depth: int = 0) -> int:
    if node.symbol not in STRUCTURE_SET:
        return -1
    return max([depth, *(max_structure_depth(child, depth + 1) for child in node.children)])


def fits_592_layout(
    node: IDSNode,
    eligible: set[str],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> bool:
    def branch_fits(branch: IDSNode) -> bool:
        if branch.symbol not in STRUCTURE_SET:
            return branch.symbol in eligible
        return all(
            (child.symbol not in STRUCTURE_SET and child.symbol in eligible)
            or bool(reverse_subtrees.get(canonical_ids(child)))
            for child in branch.children
        ) or bool(reverse_subtrees.get(canonical_ids(branch)))

    if node.symbol not in STRUCTURE_SET:
        return node.symbol in eligible
    return all(branch_fits(child) for child in node.children)


def character_category(character: str) -> str:
    codepoint = ord(character)
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana_and_related"
    if 0x30A0 <= codepoint <= 0x30FF:
        return "katakana_and_related"
    if 0x3200 <= codepoint <= 0x32FF:
        return "enclosed_ideographs_and_units"
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    ):
        return "cjk_ideographs"
    if codepoint < 0x3000 or 0x25A0 <= codepoint <= 0x25FF:
        return "punctuation_geometric_and_other_symbols"
    if 0x3000 <= codepoint <= 0x303F:
        return "cjk_punctuation_and_iteration_marks"
    return "other_unicode"


def effective_tree(
    node: IDSNode,
    depth: int,
    max_depth: int,
    eligible: set[str],
    parsed_ids: dict[str, IDSNode],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> tuple[IDSNode, bool]:
    def replacement() -> str | None:
        candidates = reverse_subtrees.get(canonical_ids(node), [])
        return candidates[0] if candidates else None

    if node.symbol not in STRUCTURE_SET:
        if node.symbol in eligible:
            return IDSNode(node.symbol), True
        expansion = parsed_ids.get(node.symbol)
        if depth < max_depth and expansion is not None and expansion.symbol != node.symbol:
            return effective_tree(
                expansion, depth + 1, max_depth, eligible, parsed_ids, reverse_subtrees
            )
        return IDSNode(node.symbol), False

    if depth >= max_depth:
        candidate = replacement()
        return (IDSNode(candidate), True) if candidate is not None else (node, False)

    expanded_children: list[IDSNode] = []
    all_resolved = True
    for child in node.children:
        expanded_child, resolved = effective_tree(
            child, depth + 1, max_depth, eligible, parsed_ids, reverse_subtrees
        )
        expanded_children.append(expanded_child)
        all_resolved = all_resolved and resolved
        if not resolved:
            candidate = replacement()
            if candidate is not None:
                return IDSNode(candidate), True

    return IDSNode(node.symbol, tuple(expanded_children)), all_resolved


def render_ids_tree(node: IDSNode) -> str:
    lines = [node.symbol]
    for index, child in enumerate(node.children):
        is_last = index == len(node.children) - 1
        child_lines = render_ids_tree(child).splitlines()
        lines.append(("└─ " if is_last else "├─ ") + child_lines[0])
        continuation = "   " if is_last else "│  "
        lines.extend(continuation + line for line in child_lines[1:])
    return "\n".join(lines)


def glyph_label(character: str) -> str:
    return f"`{character}` ({unicode_key(character)})"


def leaf_code_source(
    node: IDSNode,
    eligible: set[str],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> str | None:
    if node.symbol not in STRUCTURE_SET and node.symbol in eligible:
        return f"direct glyph code for {glyph_label(node.symbol)}"
    candidates = reverse_subtrees.get(canonical_ids(node), [])
    if candidates:
        return f"exact-subtree glyph code for {glyph_label(candidates[0])}"
    return None


def describe_fixed_layout(
    node: IDSNode,
    eligible: set[str],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> str:
    lines = ["Root structure slot: `NONE` (tree root is a terminal)"]
    if node.symbol not in STRUCTURE_SET:
        source = leaf_code_source(node, eligible, reverse_subtrees)
        lines[0] = "Root mode: `SINGLE`"
        lines.append(f"Leaf slot L0: {source}" if source else f"Leaf slot L0: NO CODE for `{node.symbol}`")
        return "\n".join(lines)

    lines[0] = f"Root structure slot: `{node.symbol}` (requires its assigned nonzero 4-bit symbol)"
    for branch_index in range(3):
        if branch_index >= len(node.children):
            lines.append(f"Branch B{branch_index}: `NONE` (unused); its three leaf slots are zero")
            continue

        branch = node.children[branch_index]
        lines.append(f"Branch B{branch_index} for IDS child `{branch.symbol}`:")
        if branch.symbol not in STRUCTURE_SET:
            source = leaf_code_source(branch, eligible, reverse_subtrees)
            if source:
                lines.append(f"  branch structure: `NONE`; leaf L{branch_index}.0: {source}")
            else:
                lines.append(
                    f"  NOT REPRESENTABLE: leaf L{branch_index}.0 needs a 64-bit code for "
                    f"`{branch.symbol}`, but none is available"
                )
            continue

        child_sources = [
            leaf_code_source(child, eligible, reverse_subtrees) for child in branch.children
        ]
        if all(child_sources):
            lines.append(f"  branch structure slot: `{branch.symbol}`")
            for leaf_index, source in enumerate(child_sources):
                lines.append(f"  leaf L{branch_index}.{leaf_index}: {source}")
            for leaf_index in range(len(child_sources), 3):
                lines.append(f"  leaf L{branch_index}.{leaf_index}: zero (unused)")
            continue

        branch_source = leaf_code_source(branch, eligible, reverse_subtrees)
        if branch_source:
            lines.append(f"  branch structure: `NONE`; leaf L{branch_index}.0: {branch_source}")
            continue

        lines.append(
            f"  NOT REPRESENTABLE: `{branch.symbol}` needs a structure slot here, but one or more "
            "children cannot be stored as 64-bit leaves and the whole branch has no exact-subtree code"
        )
        for leaf_index, (child, source) in enumerate(zip(branch.children, child_sources)):
            if source:
                lines.append(f"  required leaf L{branch_index}.{leaf_index}: {source}")
            elif child.symbol in STRUCTURE_SET:
                lines.append(
                    f"  blocked leaf L{branch_index}.{leaf_index}: subtree `"
                    f"{ids_to_polish(child)}` has no exact registered glyph code; "
                    "it contains an operator below the allowed depth"
                )
            else:
                lines.append(
                    f"  blocked leaf L{branch_index}.{leaf_index}: no direct code for "
                    f"`{child.symbol}`"
                )
    return "\n".join(lines)


def write_fixed_layout_report(
    output_path: Path,
    failures: list[dict[str, Any]],
    total: int,
    eligible: set[str],
    parsed_ids: dict[str, IDSNode],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
) -> None:
    single_fallbacks = sum(item["character"] in eligible for item in failures)
    unresolved = sum(not item["resolved"] for item in failures)
    resolved_without_class_code = sum(
        item["resolved"] and item["character"] not in eligible for item in failures
    )
    lines = [
        "# Fixed 592-Bit Layout Validation: Non-Fitting Classes",
        "",
        f"- Dataset classes: {total}",
        f"- Strict fixed-layout failures: {len(failures)}",
        f"- Failures rescuable by whole-class `SINGLE` using the class's own 64-bit code: {single_fallbacks}",
        f"- Unresolved IDS terminal failures: {unresolved}",
        f"- Resolved trees with no class-level code and an uncompressible deep subtree: {resolved_without_class_code}",
        "- Inputs: `dataset/ids_text/ids.txt`, `dataset/ids_text/ids-cdp.txt`, and "
        "`outputs/261018_cos_median_codebook/all_glyph_codes.pkl` with rendered status from "
        "`outputs/261007_correct_dict/combined_render_metadata.json`.",
        "",
        "## Interpretation",
        "",
        "The ordinary tree below is the merged source IDS definition (or a terminal-only placeholder "
        "when the class has no IDS entry). The effective tree is the result after the existing "
        "component expansion and exact-subtree replacement pass. The 592-bit view shows the "
        "root slot, three branch-operator slots, and three 64-bit leaf positions per branch.",
        "",
        "A subtree code means an existing rendered glyph code whose IDS tree exactly matches that "
        "subtree. It does not mean merely that every component has an IDS definition. `SINGLE` "
        "fallback at the class root discards the internal IDS decomposition and is not counted as "
        "strict structural fit.",
        "",
        "`&CDP-...;` is a CDP component identifier in the IDS data, not a Unicode character glyph "
        "code. For the unresolved cases listed here, the IDS entry for that identifier is only a "
        "self-reference and the current rendered 64-bit dictionary has no direct code for it.",
        "",
        "The report names structure symbols rather than assigning concrete 4-bit values. The "
        "symbol-to-bit mapping must reserve `0000` for `NONE` and assign distinct nonzero values "
        "to all 12 IDS operators and `SINGLE`.",
        "",
    ]

    for index, item in enumerate(failures, start=1):
        character = item["character"]
        raw_tree = item["raw_tree"]
        effective = item["effective_tree"]
        lines.extend(
            [
                f"## {index}. {glyph_label(character)}",
                "",
                f"- Unicode: `{unicode_key(character)}`",
                f"- IDS dictionary entry: {'yes' if item['ids_expression'] is not None else 'no'}",
                f"- Effective terminals resolved to usable 64-bit codes: {'yes' if item['resolved'] else 'no'}",
                f"- Direct 64-bit code for this whole class: {'yes' if character in eligible else 'no'}",
                f"- Strict-layout failure: `{item['reason']}`",
                "",
                "### Ordinary IDS Tree",
                "",
                "```text",
                render_ids_tree(raw_tree),
                "```",
                "",
                f"IDS prefix form: `{ids_to_polish(raw_tree)}`",
                "",
                "### After Component Resolution",
                "",
                "```text",
                render_ids_tree(effective),
                "```",
                "",
                f"Effective prefix form: `{ids_to_polish(effective)}`",
                "",
                "### Fixed 592-Bit Layout Tree",
                "",
                "```text",
                describe_fixed_layout(effective, eligible, reverse_subtrees),
                "```",
                "",
            ]
        )

        missing_terminals: set[str] = set()

        def collect_missing(node: IDSNode) -> None:
            if node.symbol not in STRUCTURE_SET and node.symbol not in eligible:
                missing_terminals.add(node.symbol)
            for child in node.children:
                collect_missing(child)

        collect_missing(effective)
        if missing_terminals:
            lines.extend(["### Unresolved Terminal Details", ""])
            for token in sorted(missing_terminals):
                component_tree = parsed_ids.get(token)
                if component_tree is None:
                    ids_status = "no IDS entry"
                elif ids_to_polish(component_tree) == token:
                    ids_status = "IDS entry exists but is a self-reference, not a decomposition"
                else:
                    ids_status = f"IDS entry exists: `{ids_to_polish(component_tree)}`"
                lines.append(
                    f"- `{token}`: no usable direct 64-bit code; {ids_status}."
                )
            lines.append("")

        if character in eligible:
            lines.extend(
                [
                    "### Optional Whole-Class Fallback",
                    "",
                    f"Root mode `SINGLE`, with the class's own 64-bit code for {glyph_label(character)} "
                    "in the first leaf slot. This represents the class identity but not the IDS tree above.",
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_characters = collect_dataset_characters(Path(args.dataset_root))
    ids_entries = merge_ids_entries(read_ids_entries([Path(path) for path in args.ids_files]))
    eligible = load_eligible_radicals(
        Path(args.radical_codes), Path(args.render_metadata), Path(args.image_root)
    )

    parsed_ids: dict[str, IDSNode] = {}
    parse_errors: dict[str, str] = {}
    for character, expression in ids_entries.items():
        try:
            parsed_ids[character] = parse_ids_expression(expression)
        except ValueError as exc:
            parse_errors[character] = str(exc)

    reverse_subtrees: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for character, tree in parsed_ids.items():
        if character in eligible:
            reverse_subtrees[canonical_ids(tree)].append(character)
    for candidates in reverse_subtrees.values():
        candidates.sort(key=lambda item: (len(item), ord(item[0]) if item else 0, item))

    depth_counts: Counter[int] = Counter()
    split_characters: Counter[int] = Counter()
    split_nodes: Counter[int] = Counter()
    missing_ids_count = 0
    missing_ids_with_terminal_code = 0
    ids_defined_unresolved_count = 0
    resolved_count = 0
    maximum_length = 0
    maximum_character = ""
    maximum_expression = ""
    maximum_resolved_length = 0
    maximum_resolved_character = ""
    samples: list[tuple[str, str]] = []
    capacity_fit_count = 0
    overflow_reasons: Counter[str] = Counter()
    overflow_samples: list[tuple[str, int, int, str]] = []
    unresolved_categories: Counter[str] = Counter()
    unresolved_category_samples: defaultdict[str, list[str]] = defaultdict(list)
    layout_fit_count = 0
    layout_fit_with_class_fallback_count = 0
    class_fallback_count = 0
    layout_failure_reasons: Counter[str] = Counter()
    layout_failure_samples: list[tuple[str, int, int, int, bool, str]] = []
    layout_failures: list[dict[str, Any]] = []

    for character in sorted(dataset_characters, key=ord):
        tree = parsed_ids.get(character, IDSNode(character))
        expanded, resolved = effective_tree(
            tree, 1, args.max_depth, eligible, parsed_ids, reverse_subtrees
        )
        depth = tree_depth(expanded)
        depth_counts[depth if depth <= 3 else 4] += 1
        if character not in ids_entries:
            missing_ids_count += 1
            if resolved:
                missing_ids_with_terminal_code += 1
        elif not resolved:
            ids_defined_unresolved_count += 1
        if resolved:
            resolved_count += 1

        if character not in ids_entries or not resolved:
            category = character_category(character)
            issue = "missing_ids" if character not in ids_entries else "missing_component_code"
            unresolved_categories[f"{issue}:{category}"] += 1
            if len(unresolved_category_samples[f"{issue}:{category}"]) < 10:
                unresolved_category_samples[f"{issue}:{category}"].append(unicode_key(character))

        length = code_length(expanded)
        structure_count, leaf_count = count_node_types(expanded)
        structure_depth = max_structure_depth(expanded)
        exceeds_structures = structure_count > 4
        exceeds_leaves = leaf_count > 9
        layout_fits = resolved and fits_592_layout(expanded, eligible, reverse_subtrees)
        if layout_fits:
            layout_fit_count += 1
            layout_fit_with_class_fallback_count += 1
        else:
            reasons = []
            if not resolved:
                reasons.append("unresolved_terminal")
            if structure_depth > 1:
                reasons.append("operator_below_depth_1")
            if exceeds_structures:
                reasons.append("more_than_4_structure_nodes")
            if exceeds_leaves:
                reasons.append("more_than_9_leaves")
            reason = ",".join(reasons)
            layout_failures.append(
                {
                    "character": character,
                    "ids_expression": ids_entries.get(character),
                    "raw_tree": tree,
                    "effective_tree": expanded,
                    "resolved": resolved,
                    "reason": reason,
                }
            )
            layout_failure_reasons[reason] += 1
            if len(layout_failure_samples) < args.samples:
                layout_failure_samples.append(
                    (character, structure_depth, structure_count, leaf_count, resolved, ids_to_polish(expanded))
                )
            if character in eligible:
                layout_fit_with_class_fallback_count += 1
                class_fallback_count += 1

        if length > maximum_length:
            maximum_length = length
            maximum_character = character
            maximum_expression = ids_to_polish(expanded)
        if resolved and length > maximum_resolved_length:
            maximum_resolved_length = length
            maximum_resolved_character = character

        splits = count_split_nodes(expanded)
        for arity in (2, 3):
            if splits[arity]:
                split_characters[arity] += 1
                split_nodes[arity] += splits[arity]
        if depth >= 4 and len(samples) < args.samples:
            samples.append((character, ids_to_polish(expanded)))

    total = len(dataset_characters)
    print(f"Total analyzed characters: {total}")
    for label, bucket in (("Depth 1", 1), ("Depth 2", 2), ("Depth 3", 3), ("Depth 4+ (Fatal!)", 4)):
        count = depth_counts[bucket]
        percentage = count * 100.0 / total if total else 0.0
        print(f"{label}: {count} ({percentage:.2f}%)")

    print(
        "Dataset classes without an IDS dictionary entry (treated as one terminal leaf): "
        f"{missing_ids_count}; direct terminal code available: {missing_ids_with_terminal_code}; "
        f"not available: {missing_ids_count - missing_ids_with_terminal_code}"
    )
    print(
        "IDS-defined classes not resolvable with current rendered radical codes: "
        f"{ids_defined_unresolved_count}"
    )
    print(
        "Classes resolved by both IDS and current rendered radical codes: "
        f"{resolved_count}/{total}"
    )
    if unresolved_categories:
        print("Classes needing separate handling (issue/category: count; sample Unicode keys):")
        for category, count in sorted(unresolved_categories.items()):
            examples = ", ".join(unresolved_category_samples[category])
            print(f"  {category}: {count}; {examples}")
    print(f"IDS parse errors in analyzed classes: {sum(ch in parse_errors for ch in dataset_characters)}")
    print(
        "Fits strict 592-dimension layout (operators only at depth 0/1; 4 structure slots; "
        f"9 leaf slots; all terminals resolved): {layout_fit_count}/{total}; "
        f"does not fit: {total - layout_fit_count}"
    )
    print(
        "Fits if a non-fitting class may use its own 64-bit glyph code as SINGLE: "
        f"{layout_fit_with_class_fallback_count}/{total}; "
        f"rescued by class-level SINGLE: {class_fallback_count}"
    )
    print(f"592-layout failure reasons: {dict(layout_failure_reasons)}")
    if layout_failure_samples:
        print("\n--- Samples that do not fit the strict 592-dimension layout ---")
        for character, depth, structures, leaves, resolved, expression in layout_failure_samples:
            print(
                f"[{character}] max_operator_depth={depth}, structures={structures}, "
                f"leaves={leaves}, terminals_resolved={resolved}: {expression}"
            )
    print(
        "Maximum code length across all effective IDS trees (assuming 64-bit terminal codes): "
        f"{maximum_length} dimensions, character={maximum_character!r}, IDS={maximum_expression}"
    )
    print(
        "Maximum code length among currently encodable classes: "
        f"{maximum_resolved_length} dimensions, character={maximum_resolved_character!r}"
    )
    print(
        "2-way split: "
        f"{split_characters[2]} characters, {split_nodes[2]} nodes; "
        "3-way split: "
        f"{split_characters[3]} characters, {split_nodes[3]} nodes"
    )
    if samples:
        print("\n--- Samples of Depth 4+ ---")
        for character, expression in samples:
            print(f"[{character}]: {expression}")
    if args.report_output:
        report_path = Path(args.report_output)
        write_fixed_layout_report(
            report_path, layout_failures, total, eligible, parsed_ids, reverse_subtrees
        )
        print(f"Fixed-layout failure report written: {report_path.resolve()}")


if __name__ == "__main__":
    main()