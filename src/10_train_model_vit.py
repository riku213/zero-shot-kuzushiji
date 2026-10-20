from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent

# The module name starts with a digit, so it cannot be imported normally.
_spec = importlib.util.spec_from_file_location("train_model_6", SRC_DIR / "6_train_model.py")
base = importlib.util.module_from_spec(_spec)
sys.modules["train_model_6"] = base
_spec.loader.exec_module(base)

IMAGE_SIZE = 224
SETTINGS = argparse.Namespace(lr=5e-5, head_lr=1e-3, weight_decay=0.05, warmup_epochs=1.0, num_workers=4)


class ViTCharacterDataset(base.CharacterImageDataset):
    """Loads images directly at the ViT input size, whatever size the caller asks for."""

    def __init__(self, entries: list[dict[str, Any]], image_size: int = IMAGE_SIZE) -> None:
        super().__init__(entries, image_size=IMAGE_SIZE)


class ViTFareRecognitionModel(nn.Module):
    def __init__(self, codebook_dim: int = 1212, **_: Any) -> None:
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.projection = nn.Linear(768, codebook_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            features = self.backbone(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.projection(features.float())
        return base.STEBinarize.apply(logits)


def learning_rate_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


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
    train_loader = DataLoader(ViTCharacterDataset(train_entries), batch_size=batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(ViTCharacterDataset(val_entries), batch_size=batch_size, shuffle=False, **loader_options)

    backbone_params = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": SETTINGS.lr, "base_lr": SETTINGS.lr},
            {"params": head_params, "lr": SETTINGS.head_lr, "base_lr": SETTINGS.head_lr},
        ],
        weight_decay=SETTINGS.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(SETTINGS.warmup_epochs * steps_per_epoch)

    best_accuracy = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    start_epoch = 1

    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if isinstance(state.get("model_state_dict"), dict):
                model.load_state_dict(state["model_state_dict"])
            if isinstance(state.get("optimizer_state_dict"), dict):
                optimizer.load_state_dict(state["optimizer_state_dict"])
                base.move_optimizer_state_to_device(optimizer, device)
            best_accuracy = float(state.get("best_accuracy", best_accuracy))
            best_epoch = int(state.get("best_epoch", best_epoch))
            start_epoch = int(state.get("epoch", 0)) + 1
            if verbose:
                print(f"Resumed training from {state_path} at epoch {start_epoch}")

    if start_epoch > epochs:
        if verbose:
            print(f"Checkpoint at {state_path} already covers {epochs} epochs; skipping training loop.")
        if checkpoint_path.exists():
            best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(best_state, dict) and isinstance(best_state.get("state_dict"), dict):
                best_state_dict = best_state["state_dict"]
        if best_state_dict is None:
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state_dict)
        return best_accuracy, best_epoch

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch}", leave=False)):
            if batch is None:
                continue
            factor = learning_rate_factor((epoch - 1) * steps_per_epoch + batch_index, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            encoded = model(images)
            logits = base.codebook_logits(encoded, codebook_matrix, structured_codebook)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_accuracy = base.evaluate(model, val_loader, codebook_matrix, device, structured_codebook)

        if verbose:
            print(f"Epoch {epoch:02d}/{epochs:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "class_to_index": class_to_index,
                    "codebook_dim": codebook_matrix.shape[1],
                    "state_dict": best_state_dict,
                },
                checkpoint_path,
            )
            if verbose:
                print(f"Saved best model checkpoint to {checkpoint_path}")

        torch.save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "class_to_index": class_to_index,
                "codebook_dim": codebook_matrix.shape[1],
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": seed,
            },
            state_path,
        )
        if verbose:
            print(f"Saved resume state to {state_path}")

    if best_state_dict is None:
        raise RuntimeError("No model checkpoint was saved during training.")

    model.load_state_dict(best_state_dict)
    return best_accuracy, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT-B/16 options; all other arguments are passed to 6_train_model.py.", allow_abbrev=False)
    parser.add_argument("--vit-lr", type=float, default=SETTINGS.lr, help="Learning rate for the pretrained ViT backbone.")
    parser.add_argument("--vit-head-lr", type=float, default=SETTINGS.head_lr, help="Learning rate for the new projection layer.")
    parser.add_argument("--vit-weight-decay", type=float, default=SETTINGS.weight_decay)
    parser.add_argument("--vit-warmup-epochs", type=float, default=SETTINGS.warmup_epochs)
    parser.add_argument("--num-workers", type=int, default=SETTINGS.num_workers, help="DataLoader workers.")
    options, remaining = parser.parse_known_args()
    SETTINGS.lr = options.vit_lr
    SETTINGS.head_lr = options.vit_head_lr
    SETTINGS.weight_decay = options.vit_weight_decay
    SETTINGS.warmup_epochs = options.vit_warmup_epochs
    SETTINGS.num_workers = options.num_workers
    sys.argv = [sys.argv[0], *remaining]

    torch.set_float32_matmul_precision("high")
    base.FareRecognitionModel = ViTFareRecognitionModel
    base.CharacterImageDataset = ViTCharacterDataset
    base.train_model_on_dataset = train_vit_on_dataset
    base.main()


if __name__ == "__main__":
    main()
