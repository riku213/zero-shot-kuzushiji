from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("fixed_592_builder", SRC / "11_build_fixed_592_codebook.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load fixed 592-bit codebook builder.")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)

from structured_codebook import IDSNode, STRUCTURE_SYMBOLS, canonical_ids


def glyph(value: int) -> list[int]:
    return [1 if (value >> (index % 8)) & 1 else -1 for index in range(64)]


class Fixed592CodebookTests(unittest.TestCase):
    def test_symbol_codes_reserve_none_and_follow_existing_operator_order(self) -> None:
        self.assertEqual(fixed.SYMBOL_CODES["NONE"], 0)
        self.assertEqual(fixed.SYMBOL_CODES["SINGLE"], 1)
        self.assertEqual(
            [fixed.SYMBOL_CODES[symbol] for symbol in STRUCTURE_SYMBOLS],
            list(range(2, 2 + len(STRUCTURE_SYMBOLS))),
        )
        self.assertEqual(len(set(fixed.SYMBOL_CODES.values())), 14)
        self.assertTrue(all(len(fixed.signed_bits(value)) == 4 for value in fixed.SYMBOL_CODES.values()))

    def test_binary_root_places_two_terminal_children_and_zeroes_unused_leaves(self) -> None:
        codes = {"口": glyph(1), "十": glyph(2)}
        node = IDSNode("⿰", (IDSNode("口"), IDSNode("十")))
        vector, mask, error = fixed.encode_fixed_tree(node, codes, {})
        self.assertIsNone(error)
        self.assertIsNotNone(vector)
        self.assertIsNotNone(mask)
        assert vector is not None and mask is not None
        self.assertEqual(len(vector), 592)
        self.assertEqual(len(mask), 592)
        np.testing.assert_array_equal(vector[8:72], codes["口"])
        np.testing.assert_array_equal(vector[204:268], codes["十"])
        self.assertEqual(vector[136:200], [0] * 64)
        self.assertEqual(sum(mask), 16 + 2 * 64)
        self.assertTrue(all(mask[index] for index in range(16)))

    def test_nested_branch_uses_operator_slot_and_leaf_slots(self) -> None:
        codes = {"口": glyph(1), "十": glyph(2), "木": glyph(3)}
        node = IDSNode("⿰", (IDSNode("⿱", (IDSNode("口"), IDSNode("十"))), IDSNode("木")))
        vector, mask, error = fixed.encode_fixed_tree(node, codes, {})
        self.assertIsNone(error)
        assert vector is not None and mask is not None
        np.testing.assert_array_equal(vector[8:72], codes["口"])
        np.testing.assert_array_equal(vector[72:136], codes["十"])
        np.testing.assert_array_equal(vector[204:268], codes["木"])
        self.assertEqual(sum(mask), 16 + 3 * 64)

    def test_exact_registered_subtree_can_be_collapsed_to_one_leaf(self) -> None:
        codes = {"口": glyph(1), "木": glyph(2), "X": glyph(3)}
        subtree = IDSNode("⿱", (IDSNode("⿰", (IDSNode("口"), IDSNode("木"))), IDSNode("口")))
        root = IDSNode("⿰", (subtree, IDSNode("木")))
        reverse = {canonical_ids(subtree): ["X"]}
        vector, mask, error = fixed.encode_fixed_tree(root, codes, reverse)
        self.assertIsNone(error)
        assert vector is not None and mask is not None
        self.assertEqual(vector[4:8], fixed.signed_bits(fixed.SYMBOL_CODES["NONE"]))
        np.testing.assert_array_equal(vector[8:72], codes["X"])
        np.testing.assert_array_equal(vector[204:268], codes["木"])
        self.assertEqual(sum(mask), 16 + 2 * 64)

    def test_single_root_uses_first_leaf_and_masks_all_operator_slots(self) -> None:
        codes = {"一": glyph(5)}
        vector, mask, error = fixed.encode_fixed_tree(IDSNode("一"), codes, {})
        self.assertIsNone(error)
        assert vector is not None and mask is not None
        self.assertEqual(vector[0:4], fixed.signed_bits(fixed.SYMBOL_CODES["SINGLE"]))
        np.testing.assert_array_equal(vector[8:72], codes["一"])
        self.assertEqual(sum(mask), 16 + 64)

    def test_missing_leaf_code_is_rejected(self) -> None:
        node = IDSNode("⿰", (IDSNode("口"), IDSNode("未登録")))
        vector, mask, error = fixed.encode_fixed_tree(node, {"口": glyph(1)}, {})
        self.assertIsNone(vector)
        self.assertIsNone(mask)
        self.assertIn("64-bit", error or "")


if __name__ == "__main__":
    unittest.main()
