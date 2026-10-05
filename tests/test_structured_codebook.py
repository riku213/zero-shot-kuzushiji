from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.structured_codebook import (
    CODEBOOK_FORMAT,
    STRUCTURE_BITS,
    codebook_logits,
    parse_codebook_payload,
    parse_ids_expression,
    ids_to_polish,
)


class StructuredCodebookTests(unittest.TestCase):
    def test_polish_parser_preserves_operator_and_child_order(self) -> None:
        tree = parse_ids_expression("⿰口十")
        self.assertEqual(ids_to_polish(tree), "⿰口十")
        self.assertNotEqual(
            ids_to_polish(parse_ids_expression("⿰口十")),
            ids_to_polish(parse_ids_expression("⿰十口")),
        )

    def test_cdp_entity_is_one_atomic_ids_token(self) -> None:
        tree = parse_ids_expression("⿰口&CDP-8B6C;")
        self.assertEqual(ids_to_polish(tree), "⿰口&CDP-8B6C;")
        self.assertEqual(tree.children[1].symbol, "&CDP-8B6C;")

    def test_structure_codes_are_four_signed_bits_and_distinct(self) -> None:
        self.assertEqual(len(STRUCTURE_BITS["⿰"]), 4)
        self.assertTrue(set(STRUCTURE_BITS["⿰"]) <= {-1, 1})
        self.assertNotEqual(STRUCTURE_BITS["⿰"], STRUCTURE_BITS["⿱"])

    def test_structured_loader_validates_padding_mask(self) -> None:
        payload = {
            "format": CODEBOOK_FORMAT,
            "max_code_length": 4,
            "codes": {"U+4E00": [1, -1, 0, 0]},
            "masks": {"U+4E00": [True, True, False, False]},
        }
        loaded = parse_codebook_payload(payload, Path("fixture.pkl"))
        self.assertTrue(loaded.structured)
        np.testing.assert_array_equal(loaded["U+4E00"], np.array([1, -1, 0, 0], dtype=np.float32))

    def test_masked_logits_ignore_padding_and_rescale_to_64_bits(self) -> None:
        outputs = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
        codebook = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        logits = codebook_logits(outputs, codebook, structured=True)
        self.assertAlmostEqual(float(logits[0, 0]), 64.0)

    def test_legacy_mapping_stays_unscaled(self) -> None:
        outputs = torch.tensor([[1.0, -1.0]])
        codebook = torch.tensor([[1.0, -1.0]])
        logits = codebook_logits(outputs, codebook, structured=False)
        self.assertAlmostEqual(float(logits[0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()