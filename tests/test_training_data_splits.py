from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("fare_train_model", SRC / "6_train_model.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not import src/6_train_model.py for split tests.")
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)


class TrainingDataSplitTests(unittest.TestCase):
    def test_casia_numeric_and_named_symbol_folder_labels_resolve(self) -> None:
        codebook = {"U+0030": [], "U+002A": [], "U+005C": [], "U+003C": []}
        label_map = {
            "0": "U+0030",
            "asterisk": "U+002A",
            "backslash": "U+005C",
            "less_than": "U+003C",
        }
        for folder, expected in label_map.items():
            with self.subTest(folder=folder):
                path = Path("CASIA-HWDB_Train/Train") / folder / "18.png"
                self.assertEqual(
                    trainmod.resolve_pretrain_true_label(path, codebook, set(codebook), label_map),
                    expected,
                )

    def test_official_test_images_never_enter_train_or_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CASIA-HWDB"
            train_root = root / "CASIA-HWDB_Train" / "Train"
            test_root = root / "CASIA-HWDB_Test" / "Test"
            labels = ("A", "B", "0", "asterisk")
            for split_root in (train_root, test_root):
                for label in labels:
                    class_dir = split_root / label
                    class_dir.mkdir(parents=True)
                    for index in range(1, 5):
                        (class_dir / f"{index}.png").write_bytes(b"png-placeholder")

            codebook = {f"U+{ord(char):04X}": [] for char in ("A", "B", "0", "*")}
            label_map = {
                "A": "U+0041",
                "B": "U+0042",
                "0": "U+0030",
                "asterisk": "U+002A",
            }
            class_to_index, train, validation, test, summary = trainmod.collect_pretrain_entries_with_redefined_split(
                root,
                codebook,
                seed=17,
                train_class_ratio=0.5,
                seen_train_ratio=0.75,
                folder_label_map=label_map,
            )

            self.assertEqual(len(class_to_index), 4)
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertTrue(test)
            self.assertTrue(all("CASIA-HWDB_Train" in entry["image_path"] for entry in train + validation))
            self.assertTrue(all("CASIA-HWDB_Test" in entry["image_path"] for entry in test))
            self.assertTrue(any(entry["split"] == "val_seen" for entry in validation))
            self.assertTrue(any(entry["split"] == "val_unseen" for entry in validation))
            self.assertTrue(any(entry["split"] == "test_seen" for entry in test))
            self.assertTrue(any(entry["split"] == "test_unseen" for entry in test))
            self.assertEqual(summary["official_test"]["total_images"], 16)

    def test_finetuning_split_has_seen_and_unseen_validation_and_test(self) -> None:
        entries = []
        for class_index in range(8):
            unicode_key = f"U+{0x4E00 + class_index:04X}"
            for sample_index in range(10):
                entries.append(
                    {
                        "unicode": unicode_key,
                        "label": class_index,
                        "image_path": f"book/characters/{unicode_key}/{sample_index}.jpg",
                    }
                )

        train, validation, test, summary = trainmod.build_seen_unseen_sample_split(
            entries,
            train_class_ratio=0.5,
            train_sample_ratio=0.6,
            validation_sample_ratio=0.5,
            seed=73,
        )
        self.assertTrue(train)
        self.assertTrue(any(item["split"] == "val_seen" for item in validation))
        self.assertTrue(any(item["split"] == "val_unseen" for item in validation))
        self.assertTrue(any(item["split"] == "test_seen" for item in test))
        self.assertTrue(any(item["split"] == "test_unseen" for item in test))
        train_classes = {item["unicode"] for item in train}
        val_unseen_classes = {item["unicode"] for item in validation if item["split"] == "val_unseen"}
        test_unseen_classes = {item["unicode"] for item in test if item["split"] == "test_unseen"}
        self.assertTrue(train_classes.isdisjoint(val_unseen_classes))
        self.assertTrue(train_classes.isdisjoint(test_unseen_classes))
        self.assertEqual(summary["classes"], 8)

    def test_manifest_from_official_test_is_rejected_for_train_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CASIA-HWDB"
            train_class = root / "CASIA-HWDB_Train" / "Train" / "A"
            test_class = root / "CASIA-HWDB_Test" / "Test" / "A"
            train_class.mkdir(parents=True)
            test_class.mkdir(parents=True)
            (train_class / "1.png").write_bytes(b"png-placeholder")
            test_image = test_class / "1.png"
            test_image.write_bytes(b"png-placeholder")
            stale_train_manifest = Path(temporary) / "train_manifest.txt"
            stale_train_manifest.write_text(str(test_image) + "\n", encoding="utf-8")
            codebook = {"U+0041": []}
            with self.assertRaisesRegex(ValueError, "outside official_train"):
                trainmod.collect_pretrain_entries_with_redefined_split(
                    root,
                    codebook,
                    seed=1,
                    train_class_ratio=0.8,
                    seen_train_ratio=0.8,
                    manifest_path=stale_train_manifest,
                    folder_label_map={"A": "U+0041"},
                )


if __name__ == "__main__":
    unittest.main()