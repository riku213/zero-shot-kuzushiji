from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.inspect_ids_depth import (
    collect_dataset_characters,
    effective_tree,
    load_eligible_radicals,
)
from src.fare_pipeline import merge_ids_entries, read_ids_entries
from structured_codebook import (
    IDSNode,
    canonical_ids,
    ids_to_polish,
    parse_ids_expression,
)
from src.two_stage.pipeline.reconstruct import (
    ReverseIDSIndex,
    reconstruct,
    restore_ids,
    tree_to_skeleton_components,
)


ROOT = Path(__file__).resolve().parents[1]


class TwoStageReconstructionTests(unittest.TestCase):
    def test明_and_森_round_trip(self) -> None:
        cases = {
            "明": ("⿰日月", ["⿰", "□", "□"], ["日", "月"]),
            "森": ("⿱木⿰木木", ["⿱", "□", "⿰", "□", "□"], ["木", "木", "木"]),
        }
        trees = {character: parse_ids_expression(ids) for character, (ids, _, _) in cases.items()}
        reverse_index = ReverseIDSIndex.from_trees(trees)

        for character, (expected_ids, expected_skeleton, expected_components) in cases.items():
            with self.subTest(character=character):
                skeleton, components = tree_to_skeleton_components(trees[character])
                self.assertEqual(skeleton, expected_skeleton)
                self.assertEqual(components, expected_components)
                self.assertEqual(restore_ids(skeleton, components), expected_ids)
                result = reconstruct(skeleton, components, reverse_index)
                self.assertEqual(result.ids, expected_ids)
                self.assertEqual(result.character, character)

    def test_leaf_tree_stays_atomic(self) -> None:
        parsed_ids = {"化": parse_ids_expression("⿰亻匕")}
        leaf_tree, resolved = effective_tree(
            parsed_ids["化"],
            depth=1,
            max_depth=6,
            eligible={"化"},
            parsed_ids=parsed_ids,
            reverse_subtrees={canonical_ids(parsed_ids["化"]): ["化"]},
        )

        self.assertTrue(resolved)
        self.assertEqual(leaf_tree, IDSNode("化"))
        self.assertEqual(tree_to_skeleton_components(leaf_tree), (["□"], ["化"]))
        self.assertEqual(restore_ids(["□"], ["化"]), "化")

    def test_depth_zero_and_invalid_slot_counts(self) -> None:
        tree = IDSNode("木")
        skeleton, components = tree_to_skeleton_components(tree)
        self.assertEqual((skeleton, components), (["□"], ["木"]))
        self.assertEqual(restore_ids(skeleton, components), "木")
        result = reconstruct(skeleton, components, ReverseIDSIndex.from_trees({"木": tree}))
        self.assertEqual(result.character, "木")
        self.assertTrue(result.dictionary_hit)
        with self.assertRaisesRegex(ValueError, "slot"):
            restore_ids(["⿰", "□", "□"], ["日"])

    def test_reverse_lookup_miss_collision_and_unresolved_classes(self) -> None:
        tree = parse_ids_expression("⿰日月")
        index = ReverseIDSIndex.from_trees(
            {"明": tree, "昍": tree, "未解決クラス": None}
        )

        self.assertEqual(index.lookup("⿰日月"), ["昍", "明"])
        self.assertEqual(index.lookup("⿱木木"), "⿱木木")
        self.assertEqual(index.unresolved_characters, ("未解決クラス",))

    def test_all_dataset_classes_round_trip(self) -> None:
        ids_files = [ROOT / "dataset/ids_text/ids.txt", ROOT / "dataset/ids_text/ids-cdp.txt"]
        codes_path = ROOT / "outputs/261018_cos_median_codebook/all_glyph_codes.pkl"
        metadata_path = ROOT / "outputs/261007_correct_dict/combined_render_metadata.json"
        image_root = ROOT / "outputs/261007_correct_dict"
        dataset_root = ROOT.parent / "kuzushiji-recognition" / "char_sep_datas"
        required_paths = [*ids_files, codes_path, metadata_path, dataset_root]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            self.skipTest(f"Full-dataset inputs unavailable: {', '.join(missing)}")

        characters = collect_dataset_characters(dataset_root)
        ids_by_character = merge_ids_entries(read_ids_entries(ids_files))
        eligible = load_eligible_radicals(codes_path, metadata_path, image_root)
        parsed_ids: dict[str, IDSNode] = {}
        for character, expression in ids_by_character.items():
            try:
                parsed_ids[character] = parse_ids_expression(expression)
            except ValueError:
                continue

        reverse_subtrees: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
        for character, tree in parsed_ids.items():
            if character in eligible:
                reverse_subtrees[canonical_ids(tree)].append(character)
        for candidates in reverse_subtrees.values():
            candidates.sort(key=lambda item: (len(item), ord(item[0]) if item else 0, item))

        class_trees: dict[str, IDSNode | None] = {}
        expected_depth_counts: Counter[int] = Counter()
        for character in characters:
            source_tree = parsed_ids.get(character, IDSNode(character))
            resolved_tree, resolved = effective_tree(
                source_tree,
                depth=1,
                max_depth=6,
                eligible=eligible,
                parsed_ids=parsed_ids,
                reverse_subtrees=reverse_subtrees,
            )
            class_trees[character] = resolved_tree if resolved else None
            depth = max(0, _max_operator_depth(resolved_tree) + 1)
            expected_depth_counts[min(depth, 3)] += 1

        reverse_index = ReverseIDSIndex.from_trees(class_trees)
        self.assertEqual(len(characters), 4328)
        self.assertEqual(len(reverse_index.unresolved_characters), 5)
        self.assertEqual(
            expected_depth_counts,
            Counter({0: 372, 1: 3748, 2: 170, 3: 38}),
        )

        for character, tree in class_trees.items():
            if tree is None:
                self.assertIn(character, reverse_index.unresolved_characters)
                continue
            skeleton, components = tree_to_skeleton_components(tree)
            result = reconstruct(skeleton, components, reverse_index)
            self.assertEqual(result.ids, ids_to_polish(tree), character)
            if isinstance(result.character, list):
                self.assertIn(character, result.character)
            else:
                self.assertEqual(result.character, character)


def _max_operator_depth(node: IDSNode, depth: int = -1) -> int:
    if not node.children:
        return depth
    return max(_max_operator_depth(child, depth + 1) for child in node.children)


if __name__ == "__main__":
    unittest.main()