from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("finetune_supcon", SRC / "3a_finetune_resnet_supcon.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not import src/3a_finetune_resnet_supcon.py.")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

FONT = ROOT / "dataset" / "fonts" / "HanaMinA.ttf"


class LossTests(unittest.TestCase):
    def test_supcon_prefers_aligned_positives(self) -> None:
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        base = F.normalize(torch.randn(4, 16), dim=1)
        aligned = F.normalize(torch.cat([base, base + 0.01 * torch.randn(4, 16)]), dim=1)
        random_z = F.normalize(torch.randn(8, 16), dim=1)
        self.assertLess(ft.supcon_loss(aligned, labels).item(), ft.supcon_loss(random_z, labels).item())

    def test_supcon_without_positives_is_zero_with_grad(self) -> None:
        z = F.normalize(torch.randn(4, 8), dim=1).requires_grad_()
        loss = ft.supcon_loss(z, torch.arange(4))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)

    def test_uniformity_lower_when_spread(self) -> None:
        collapsed = F.normalize(torch.ones(16, 8) + 0.001 * torch.randn(16, 8), dim=1)
        spread = F.normalize(torch.randn(16, 8), dim=1)
        self.assertLess(ft.uniformity_loss(spread).item(), ft.uniformity_loss(collapsed).item())


@unittest.skipUnless(FONT.exists(), "font not available")
class DatasetTests(unittest.TestCase):
    def test_views_and_labels(self) -> None:
        fonts = ft.FontSet([FONT])
        dataset = ft.RadicalViewDataset(["木", "水", "口"], fonts, image_size=64, show_progress=False)
        view1, view2, label = dataset[1]
        self.assertEqual(tuple(view1.shape), (3, 64, 64))
        self.assertEqual(tuple(view2.shape), (3, 64, 64))
        self.assertEqual(label, 1)

    def test_one_training_step(self) -> None:
        fonts = ft.FontSet([FONT])
        dataset = ft.RadicalViewDataset(["木", "水", "口", "日"], fonts, image_size=64, show_progress=False)
        view1, view2, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
        backbone, head = ft.build_backbone(), ft.build_head()
        z = F.normalize(head(backbone(torch.cat([view1, view2])).flatten(1)), dim=1)
        loss = ft.supcon_loss(z, torch.cat([labels, labels])) + ft.uniformity_loss(z)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
