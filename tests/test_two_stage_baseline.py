from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.two_stage.data.pipeline import (
    ImageRecordDataset,
    expand_filled_marker_instances,
    resolve_effective_tree,
    split_records,
)
from src.two_stage.evaluate import build_labnote_payload, evaluate_oracle
from src.two_stage.experiment import skeleton_vocabulary, stage2_vocabularies
from src.two_stage.experiment import slot_loader
from src.two_stage.train_stage1 import _run_epoch as run_stage1_epoch
from src.two_stage.train_stage2 import _run_epoch as run_stage2_epoch
from src.structured_codebook import STRUCTURE_SYMBOLS
from src.two_stage.models.stage1_classifier import Stage1SkeletonClassifier
from src.two_stage.models.stage2_filler import (
    Stage2SlotFiller,
    VisionTransformerEncoder,
    slot_only_cross_entropy,
)
from src.two_stage.pipeline.reconstruct import ReverseIDSIndex
from structured_codebook import IDSNode, parse_ids_expression


class TwoStageBaselineTests(unittest.TestCase):
    def test_filled_marker_instances_expand_by_slot_without_component_inputs(self) -> None:
        record = {
            "image_path": "unused.png",
            "char": "森",
            "depth": 2,
            "skeleton": ["⿱", "□", "⿰", "□", "□"],
            "components": ["木", "木", "木"],
        }
        input_vocab = {"<PAD>": 0, "□": 1, "<FILLED>": 2, "⿱": 3, "⿰": 4}
        component_vocab = {"木": 0}
        instances = expand_filled_marker_instances(record, input_vocab, component_vocab, 6)

        self.assertEqual(len(instances), 3)
        self.assertEqual(instances[0]["input_ids"], [3, 1, 4, 1, 1, 0])
        self.assertEqual(instances[1]["input_ids"], [3, 2, 4, 1, 1, 0])
        self.assertEqual(instances[2]["input_ids"], [3, 2, 4, 2, 1, 0])
        self.assertEqual([item["target_position"] for item in instances], [1, 3, 4])
        self.assertEqual([item["target_id"] for item in instances], [0, 0, 0])

    def test_model_output_vocabularies_are_separate_and_slot_loss_is_masked(self) -> None:
        stage1 = Stage1SkeletonClassifier(num_skeletons=3, channels=(8,), dropout=0)
        stage1_logits = stage1(torch.randn(2, 3, 32, 32))
        self.assertEqual(stage1_logits.shape, (2, 3))

        stage2 = Stage2SlotFiller(
            input_vocabulary_size=5,
            component_vocabulary_size=7,
            max_skeleton_length=3,
            pad_id=0,
            image_size=32,
            patch_size=16,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            attention_heads=4,
            feedforward_dim=32,
            dropout=0,
        )
        logits = stage2(torch.randn(2, 3, 32, 32), torch.tensor([[3, 1, 1], [3, 1, 1]]))
        self.assertEqual(logits.shape, (2, 3, 7))
        loss = slot_only_cross_entropy(logits, torch.tensor([1, 2]), torch.tensor([2, 3]))
        expected = torch.nn.functional.cross_entropy(logits[torch.arange(2), torch.tensor([1, 2])], torch.tensor([2, 3]))
        self.assertTrue(torch.allclose(loss, expected))

    def test_pretrained_vit_resizes_positions_for_configured_image_size(self) -> None:
        class FakeEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pos_embedding = torch.nn.Parameter(torch.zeros(1, 197, 768))
                self.dropout = torch.nn.Identity()
                self.layers = torch.nn.Identity()
                self.ln = torch.nn.Identity()

        class FakeBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_proj = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)
                self.class_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
                self.encoder = FakeEncoder()
                self.heads = torch.nn.Identity()

        with patch("torchvision.models.vit_b_16", return_value=FakeBackbone()):
            encoder = VisionTransformerEncoder(
                image_size=128,
                patch_size=16,
                d_model=16,
                pretrained=True,
            )

        features = encoder(torch.randn(2, 3, 128, 128))

        self.assertEqual(features.shape, (2, 64, 16))

    def test_stage2_input_operator_vocabulary_is_fixed_across_splits(self) -> None:
        first_input, first_components = stage2_vocabularies(
            {"木", "日"}, [{"components": ["木"], "skeleton": ["⿰", "□", "□"]}]
        )
        second_input, second_components = stage2_vocabularies(
            {"木", "日"}, [{"components": ["日"], "skeleton": ["⿱", "□", "□"]}]
        )

        self.assertEqual(first_input, second_input)
        self.assertEqual(first_input["<PAD>"], 0)
        self.assertTrue(set(STRUCTURE_SYMBOLS).issubset(first_input))
        self.assertEqual(first_components, second_components)

    def test_stage2_loader_batches_input_ids_as_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (32, 32), color="white").save(image_path)
            record = {
                "image_path": str(image_path),
                "char": "明",
                "depth": 1,
                "skeleton": ["⿰", "□", "□"],
                "components": ["日", "月"],
            }
            loader = slot_loader(
                [record],
                {"<PAD>": 0, "□": 1, "<FILLED>": 2, "⿰": 3},
                {"日": 0, "月": 1},
                image_size=32,
                batch_size=2,
                workers=0,
                shuffle=False,
                max_skeleton_length=3,
            )

            batch = next(iter(loader))

            self.assertIsInstance(batch["input_ids"], torch.Tensor)
            self.assertEqual(batch["input_ids"].shape, (2, 3))

    def test_depth_limited_unresolved_leaf_is_returned_without_indexing_missing_fallback(self) -> None:
        node = IDSNode("木")

        effective, resolved = resolve_effective_tree(
            node,
            depth=1,
            max_depth=1,
            eligible=set(),
            parsed_ids={},
            reverse_subtrees={},
        )

        self.assertEqual(effective.symbol, node.symbol)
        self.assertEqual(effective.children, node.children)
        self.assertFalse(resolved)

    def test_split_is_deterministic_and_keeps_each_class_in_training(self) -> None:
        records = [
            {"char": char, "sample": index}
            for char in ("木", "日")
            for index in range(10)
        ]
        first = split_records(records, 0.6, 0.2, 7)
        second = split_records(records, 0.6, 0.2, 7)
        self.assertEqual(first, second)
        self.assertEqual({item["char"] for item in first["train"]}, {"木", "日"})
        self.assertEqual((len(first["train"]), len(first["validation"]), len(first["test"])), (12, 4, 4))

    def test_unseen_character_split_keeps_all_images_and_disjoins_classes(self) -> None:
        records = [
            {"char": chr(0x4E00 + class_index), "sample": sample_index}
            for class_index in range(10)
            for sample_index in range(class_index + 1)
        ]
        splits = split_records(records, 0.8, 0.1, 21, "unseen_character")
        class_sets = {name: {record["char"] for record in rows} for name, rows in splits.items()}

        self.assertFalse(class_sets["train"] & class_sets["validation"])
        self.assertFalse(class_sets["train"] & class_sets["test"])
        self.assertFalse(class_sets["validation"] & class_sets["test"])
        self.assertEqual(sum(map(len, splits.values())), len(records))
        self.assertEqual({record["sample"] for rows in splits.values() for record in rows},
                         {record["sample"] for record in records})
        self.assertEqual({name: len(chars) for name, chars in class_sets.items()},
                         {"train": 8, "validation": 1, "test": 1})

    def test_stage1_validation_counts_unseen_skeletons_as_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            records = []
            for index, skeleton in enumerate((["□"], ["⿰", "□", "□"])):
                image_path = Path(temp_dir) / f"{index}.png"
                Image.new("RGB", (32, 32), color="white").save(image_path)
                records.append(
                    {
                        "image_path": str(image_path),
                        "char": chr(0x4E00 + index),
                        "skeleton": skeleton,
                    }
                )
            loader = DataLoader(
                ImageRecordDataset(records, 32), batch_size=2, collate_fn=lambda rows: {
                    "image": torch.stack([row["image"] for row in rows]),
                    "skeleton": [row["skeleton"] for row in rows],
                    "char": [row["char"] for row in rows],
                }
            )
            model = Stage1SkeletonClassifier(1, channels=(8,), dropout=0)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()

            _, accuracy = run_stage1_epoch(
                model,
                loader,
                None,
                torch.device("cpu"),
                {'["□"]': 0},
            )

            self.assertEqual(accuracy, 0.5)

    def test_stage2_epoch_returns_slot_accuracy_and_image_exact_match(self) -> None:
        class FixedModel:
            def train(self, mode: bool) -> None:
                self.training = mode

            def __call__(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((2, 2, 2))
                logits[0, 0, 0] = 1
                logits[1, 1, 1] = 1
                return logits

        batch = {
            "image": torch.zeros((2, 3, 32, 32)),
            "input_ids": torch.zeros((2, 2), dtype=torch.long),
            "target_position": torch.tensor([0, 1]),
            "target_id": torch.tensor([0, 1]),
            "image_path": ["same-image.png", "same-image.png"],
        }

        loss, slot_accuracy, exact_match = run_stage2_epoch(
            FixedModel(), [batch], None, torch.device("cpu")
        )

        self.assertGreater(loss, 0.0)
        self.assertEqual(slot_accuracy, 1.0)
        self.assertEqual(exact_match, 1.0)

    def test_oracle_evaluation_metrics_and_labnote_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (32, 32), color="white").save(image_path)
            record = {
                "image_path": str(image_path),
                "char": "明",
                "depth": 1,
                "skeleton": ["⿰", "□", "□"],
                "components": ["日", "月"],
            }
            input_vocab = {"<PAD>": 0, "□": 1, "<FILLED>": 2, "⿰": 3}
            component_vocab = {"日": 0, "月": 1}
            model = _FixedSlotModel([0, 1])
            reverse_index = ReverseIDSIndex.from_trees({"明": parse_ids_expression("⿰日月")})
            metrics, rows = evaluate_oracle(
                model,
                [record],
                reverse_index,
                input_vocab,
                component_vocab,
                image_size=32,
                device=torch.device("cpu"),
            )
            self.assertEqual(metrics["overall"]["stage2.slot_accuracy"], 100.0)
            self.assertEqual(metrics["overall"]["stage2.slot_exact_match"], 100.0)
            self.assertEqual(metrics["overall"]["char.accuracy_oracle"], 100.0)
            self.assertEqual(metrics["overall"]["dict.hit_rate"], 100.0)
            self.assertEqual(len(rows), 1)

            config = {
                "experiment": {"split": "random", "title": "test", "train_ratio": 0.6, "validation_ratio": 0.2},
                "stage1": {"epochs": 1, "learning_rate": 0.001},
                "stage2": {
                    "input_mode": "filled_marker",
                    "encoder": "vit",
                    "pretrained": False,
                    "glyph_code_mode": "none",
                    "guidance_loss_weight": 0.0,
                    "triplet_loss_weight": 0.0,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "d_model": 16,
                    "encoder_layers": 1,
                    "decoder_layers": 1,
                    "patch_size": 16,
                },
            }
            data_summary = {"split_classes": {"train": 2, "test": 1}, "split_samples": {"train": 8, "test": 2}}
            payload = build_labnote_payload(config, metrics, "baseline-test", Path("best.pt"), data_summary)
            self.assertEqual(payload["experiment"]["code"], "baseline-test")
            self.assertEqual(payload["conditions"]["train_classes"], 2)
            self.assertEqual(payload["conditions"]["test_classes"], 1)
            self.assertTrue(all("key" in item and isinstance(item["value"], float) for item in payload["metrics"]))


class _FixedSlotModel:
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def eval(self):
        return self

    def __call__(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, input_ids.shape[1], 2), -10.0)
        for index, prediction in enumerate(self.predictions):
            logits[0, index + 1, prediction] = 10.0
        return logits


if __name__ == "__main__":
    unittest.main()