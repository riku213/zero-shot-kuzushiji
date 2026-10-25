from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("train_model_6", SRC_DIR / "6_train_model.py")
base = importlib.util.module_from_spec(spec)
sys.modules["train_model_6"] = base
spec.loader.exec_module(base)

IMAGE_SIZE = 224
SETTINGS = argparse.Namespace(
    lr=5e-5,
    head_lr=1e-3,
    weight_decay=0.05,
    patience=10,
    min_delta=0.01,
    aux_loss=True,
    aux_weight=0.5,
    augmentation=True,
    num_workers=4,
)


def is_kana_label(label: str) -> bool:
    try:
        text = label.strip()
        codepoint = int(text[2:], 16) if text.upper().startswith("U+") else ord(text)
    except (ValueError, TypeError):
        return False
    name = unicodedata.name(chr(codepoint), "")
    return "HIRAGANA" in name or "KATAKANA" in name


original_stable_value = base.stable_value


def stable_value_with_kana_seen(seed: int, *parts: str) -> float:
    if len(parts) >= 2 and parts[0] == "finetune-class" and is_kana_label(str(parts[1])):
        return 0.0
    return original_stable_value(seed, *parts)


base.stable_value = stable_value_with_kana_seen
original_seen_unseen_split = base.build_seen_unseen_sample_split


def build_split_with_kana_seen(
    entries: list[dict[str, Any]],
    train_class_ratio: float,
    train_sample_ratio: float,
    validation_sample_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_entries, val_entries, test_entries, _ = original_seen_unseen_split(
        entries, train_class_ratio, train_sample_ratio, validation_sample_ratio, seed
    )
    moved_kana_classes: set[str] = set()
    for group in (val_entries, test_entries):
        remaining = []
        for entry in group:
            if entry.get("split") in {"val_unseen", "test_unseen"} and is_kana_label(str(entry["unicode"])):
                moved = dict(entry)
                moved["split"] = "train"
                train_entries.append(moved)
                moved_kana_classes.add(str(entry["unicode"]))
            else:
                remaining.append(entry)
        group[:] = remaining

    if moved_kana_classes:
        print(
            f"Moved low-sample kana classes into training to keep them out of unseen splits: "
            f"{len(moved_kana_classes)} classes, {sum(is_kana_label(str(entry['unicode'])) for entry in train_entries)} train samples."
        )

    def class_count(group: list[dict[str, Any]], split: str) -> int:
        return len({str(entry["unicode"]) for entry in group if entry.get("split") == split})

    unseen_classes = {
        str(entry["unicode"])
        for entry in val_entries + test_entries
        if entry.get("split") in {"val_unseen", "test_unseen"}
    }
    summary = {
        "classes": len({str(entry["unicode"]) for entry in entries}),
        "train_classes": len({str(entry["unicode"]) for entry in train_entries}),
        "unseen_classes": len(unseen_classes),
        "train_samples": len(train_entries),
        "val_seen_classes": class_count(val_entries, "val_seen"),
        "val_seen_samples": sum(entry.get("split") == "val_seen" for entry in val_entries),
        "val_unseen_classes": class_count(val_entries, "val_unseen"),
        "val_unseen_samples": sum(entry.get("split") == "val_unseen" for entry in val_entries),
        "test_seen_classes": class_count(test_entries, "test_seen"),
        "test_seen_samples": sum(entry.get("split") == "test_seen" for entry in test_entries),
        "test_unseen_classes": class_count(test_entries, "test_unseen"),
        "test_unseen_samples": sum(entry.get("split") == "test_unseen" for entry in test_entries),
    }
    return train_entries, val_entries, test_entries, summary


base.build_seen_unseen_sample_split = build_split_with_kana_seen


class ViTCharacterDataset(base.CharacterImageDataset):
    def __init__(self, entries: list[dict[str, Any]], image_size: int = IMAGE_SIZE, augment: bool = False) -> None:
        super().__init__(entries, image_size=IMAGE_SIZE)
        self.augment = augment
        self.spatial_augmentation = transforms.RandomAffine(
            degrees=8,
            translate=(0.06, 0.06),
            scale=(0.9, 1.1),
            fill=2.0,
        )
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        item = super().__getitem__(index)
        if item is None or not self.augment:
            return item
        image = self.spatial_augmentation(item["image"])
        if random.random() < 0.15:
            image = self.blur(image)
        if random.random() < 0.2:
            image = (image + torch.randn_like(image) * 0.025).clamp(-2.2, 2.6)
        item["image"] = image
        return item


class ViTFareRecognitionModel(nn.Module):
    def __init__(self, codebook_dim: int = 400, **_: Any) -> None:
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.projection = nn.Linear(768, codebook_dim)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            features = self.backbone(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.projection(features.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return base.STEBinarize.apply(self.forward_logits(x))


def train_vit_on_dataset(
    model: nn.Module,
    train_entries: list[dict[str, Any]],
    val_entries: list[dict[str, Any]],
    codebook_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    checkpoint_path: Path,
    state_path: Path,
    class_to_index: dict[str, int],
    seed: int,
    structured_codebook: bool = False,
    verbose: bool = True,
) -> tuple[float, float]:
    workers = SETTINGS.num_workers
    loader_options = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": base.safe_collate,
    }
    train_loader = DataLoader(
        ViTCharacterDataset(train_entries, augment=SETTINGS.augmentation),
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    seen_validation = [entry for entry in val_entries if entry.get("split") in {"val_seen", ""}]
    if not seen_validation:
        seen_validation = val_entries
    val_loader = DataLoader(
        ViTCharacterDataset(seen_validation),
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    backbone_params = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": SETTINGS.lr},
            {"params": head_params, "lr": SETTINGS.head_lr},
        ],
        weight_decay=SETTINGS.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=SETTINGS.min_delta,
        threshold_mode="abs", cooldown=1, min_lr=[1e-6, 1e-5],
    )
    criterion = nn.CrossEntropyLoss()
    best_seen_loss = float("inf")
    best_accuracy = -1.0
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start_epoch = 1

    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if isinstance(state.get("model_state_dict"), dict):
                model.load_state_dict(state["model_state_dict"])
            if isinstance(state.get("optimizer_state_dict"), dict):
                optimizer.load_state_dict(state["optimizer_state_dict"])
                base.move_optimizer_state_to_device(optimizer, device)
            if isinstance(state.get("scheduler_state_dict"), dict):
                scheduler.load_state_dict(state["scheduler_state_dict"])
            best_seen_loss = float(state.get("best_seen_loss", best_seen_loss))
            best_accuracy = float(state.get("best_accuracy", best_accuracy))
            best_epoch = int(state.get("best_epoch", best_epoch))
            epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            start_epoch = int(state.get("epoch", 0)) + 1
            if verbose:
                print(f"Resumed training from {state_path} at epoch {start_epoch}")

    if start_epoch > epochs:
        if checkpoint_path.exists():
            best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_state_dict = best_state.get("state_dict") if isinstance(best_state, dict) else None
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        return best_accuracy, best_epoch

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"Train E{epoch}", leave=False):
            if batch is None:
                continue
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            raw_logits = model.forward_logits(images)
            encoded = base.STEBinarize.apply(raw_logits)
            class_scores = base.codebook_logits(encoded, codebook_matrix, structured_codebook)
            classification_loss = criterion(class_scores, labels)
            if SETTINGS.aux_loss:
                targets = codebook_matrix.index_select(0, labels)
                valid_bits = targets.ne(0)
                binary_targets = targets.gt(0).to(raw_logits.dtype)
                bit_loss = F.binary_cross_entropy_with_logits(raw_logits, binary_targets, reduction="none")
                per_sample_aux = (bit_loss * valid_bits).sum(dim=1) / valid_bits.sum(dim=1).clamp_min(1)
                auxiliary_loss = per_sample_aux.mean()
                loss = classification_loss + SETTINGS.aux_weight * auxiliary_loss
            else:
                auxiliary_loss = raw_logits.new_zeros(())
                loss = classification_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_accuracy = base.evaluate(model, val_loader, codebook_matrix, device, structured_codebook)
        scheduler.step(val_loss)
        learning_rates = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        if verbose:
            aug_label = "on" if SETTINGS.augmentation else "off"
            aux_label = "on" if SETTINGS.aux_loss else "off"
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | "
                f"val_seen_loss={val_loss:.4f} | val_seen_acc={val_accuracy:.4f} | "
                f"lr={learning_rates} | aug={aug_label} | aux={aux_label}"
            )

        if val_loss < best_seen_loss - SETTINGS.min_delta:
            best_seen_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "val_seen_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_to_index": class_to_index,
                    "codebook_dim": codebook_matrix.shape[1],
                    "state_dict": best_state_dict,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "best_seen_loss": best_seen_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "class_to_index": class_to_index,
                "codebook_dim": codebook_matrix.shape[1],
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "seed": seed,
            },
            state_path,
        )
        if epochs_without_improvement >= SETTINGS.patience:
            print(f"Early stopping: val_seen_loss did not improve for {SETTINGS.patience} epochs.")
            break

    if best_state_dict is None:
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_state_dict = saved.get("state_dict") if isinstance(saved, dict) else None
    if best_state_dict is None:
        raise RuntimeError("No best ViT checkpoint was saved.")
    model.load_state_dict(best_state_dict)
    return best_accuracy, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT-B/16 with the existing FaRE data splits and evaluation.", allow_abbrev=False)
    parser.add_argument("--vit-lr", type=float, default=SETTINGS.lr)
    parser.add_argument("--vit-head-lr", type=float, default=SETTINGS.head_lr)
    parser.add_argument("--vit-weight-decay", type=float, default=SETTINGS.weight_decay)
    parser.add_argument("--early-stopping-patience", type=int, default=SETTINGS.patience)
    parser.add_argument("--early-stopping-min-delta", type=float, default=SETTINGS.min_delta)
    parser.add_argument("--vit-aux-loss", action=argparse.BooleanOptionalAction, default=SETTINGS.aux_loss)
    parser.add_argument("--vit-aux-weight", type=float, default=SETTINGS.aux_weight)
    parser.add_argument("--vit-augmentation", action=argparse.BooleanOptionalAction, default=SETTINGS.augmentation)
    parser.add_argument("--num-workers", type=int, default=SETTINGS.num_workers)
    options, remaining = parser.parse_known_args()
    SETTINGS.lr = options.vit_lr
    SETTINGS.head_lr = options.vit_head_lr
    SETTINGS.weight_decay = options.vit_weight_decay
    SETTINGS.patience = options.early_stopping_patience
    SETTINGS.min_delta = options.early_stopping_min_delta
    SETTINGS.aux_loss = options.vit_aux_loss
    SETTINGS.aux_weight = options.vit_aux_weight
    SETTINGS.augmentation = options.vit_augmentation
    SETTINGS.num_workers = options.num_workers
    sys.argv = [sys.argv[0], *remaining]

    original_parse_args = base.parse_args
    original_save_json = base.save_json

    def parse_args_with_early_stopping() -> argparse.Namespace:
        args = original_parse_args()
        if "--epochs" not in sys.argv:
            args.epochs = 100
        if args.pretrain_root and "--pretrain-epochs" not in sys.argv:
            args.pretrain_epochs = 100
        return args

    def save_json_with_vit_settings(path: Path, payload: dict[str, Any]) -> None:
        metadata = dict(payload)
        metadata["vit_settings"] = {
            "architecture": "torchvision ViT-B/16",
            "pretrained_weights": "IMAGENET1K_V1",
            "image_size": IMAGE_SIZE,
            "output_code_bits": 400,
            "kuzushiji_kana_forced_seen": True,
            "auxiliary_bce_enabled": SETTINGS.aux_loss,
            "auxiliary_bce_weight": SETTINGS.aux_weight,
            "augmentation_enabled": SETTINGS.augmentation,
            "backbone_learning_rate": SETTINGS.lr,
            "projection_learning_rate": SETTINGS.head_lr,
            "weight_decay": SETTINGS.weight_decay,
            "early_stopping_monitor": "seen_validation_loss",
            "early_stopping_patience": SETTINGS.patience,
            "early_stopping_min_delta": SETTINGS.min_delta,
            "scheduler": "ReduceLROnPlateau",
        }
        original_save_json(path, metadata)

    base.parse_args = parse_args_with_early_stopping
    base.save_json = save_json_with_vit_settings
    base.FareRecognitionModel = ViTFareRecognitionModel
    base.CharacterImageDataset = ViTCharacterDataset
    base.train_model_on_dataset = train_vit_on_dataset
    torch.set_float32_matmul_precision("high")
    base.main()


if __name__ == "__main__":
    main()
