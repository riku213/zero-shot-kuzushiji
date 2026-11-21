from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("train_model_vit", SRC / "10_train_model_vit.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not import src/10_train_model_vit.py.")
vit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vit)


class ComponentHeadTransferTests(unittest.TestCase):
    def test_shared_rows_transfer_by_token_and_output_matches_target_vocab(self) -> None:
        source = nn.Linear(3, 3)
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
            source.bias.copy_(torch.tensor([10.0, 20.0, 30.0]))

        target, copied = vit.transfer_component_head(
            source,
            ["木", "口", "水"],
            ["水", "新部品"],
        )

        self.assertEqual(copied, 1)
        self.assertEqual(target.out_features, 2)
        self.assertTrue(torch.equal(target.weight[0], source.weight[2]))
        self.assertEqual(target.bias[0].item(), source.bias[2].item())
        self.assertFalse(torch.equal(target.weight[1], source.weight[0]))

    def test_without_source_head_creates_target_width(self) -> None:
        target, copied = vit.transfer_component_head(None, None, ["甲", "乙"], input_dim=4)
        self.assertEqual(target.in_features, 4)
        self.assertEqual(target.out_features, 2)
        self.assertEqual(copied, 0)


if __name__ == "__main__":
    unittest.main()
