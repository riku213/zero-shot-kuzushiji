from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("fixed_592_training_entry", SRC / "12_train_fixed_592_vit.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load fixed 592-bit training entry point.")
fixed_training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixed_training
spec.loader.exec_module(fixed_training)


class Fixed592TrainingSpawnTests(unittest.TestCase):
    def test_vit_dataset_and_collate_are_importable_in_spawn_workers(self) -> None:
        vit = fixed_training.load_vit_training_module()
        self.assertEqual(vit.ViTCharacterDataset.__module__, "10_train_model_vit")

        loader = DataLoader(
            vit.ViTCharacterDataset([], augment=False),
            batch_size=1,
            num_workers=4,
            collate_fn=vit.base.safe_collate,
        )
        self.assertEqual(list(loader), [])


if __name__ == "__main__":
    unittest.main()