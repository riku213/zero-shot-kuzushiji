from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FaRE recognition model from the separated kana/character dataset.")
    parser.add_argument(
        "--data-root",
        default="../kuzushiji-recognition/char_sep_datas",
        help="Root folder containing Book_ID/characters/<Unicode>/image.jpg files.",
    )
    parser.add_argument(
        "--codebook",
        default="outputs/final_codebook.pkl",
        help="Pickle file containing the final CodeBook mapping Unicode class -> code vector.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for checkpoints and reports.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/best_fare_model.pth",
        help="Path to save the best model checkpoint. Default: outputs/best_fare_model.pth",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for training and validation.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio for each class.")
    parser.add_argument("--image-size", type=int, default=96, help="Input image size (aligned with the FaRE render size).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-classes", type=int, default=None, help="Optional limit on the number of classes used for debugging.")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Optional cap per class for quick validation.")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_unicode_key(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.startswith(("U+", "u+")):
        return text.upper()
    return text


def unicode_aliases(value: Any) -> set[str]:
    aliases: set[str] = set()
    text = normalize_unicode_key(value)
    if not text:
        return aliases
    aliases.add(text)
    if text.startswith("U+"):
        raw = text[2:]
        try:
            codepoint = int(raw, 16)
            aliases.add(chr(codepoint))
        except ValueError:
            pass
    else:
        try:
            aliases.add(f"U+{ord(text):04X}".upper())
        except TypeError:
            pass
        if len(text) == 1:
            aliases.add(f"U+{ord(text):04X}".upper())
    return aliases


def load_codebook(codebook_path: Path) -> dict[str, np.ndarray]:
    with codebook_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        if "codebook" in payload and isinstance(payload["codebook"], dict):
            payload = payload["codebook"]
        if "entries" in payload and isinstance(payload["entries"], list):
            entries = payload["entries"]
            converted: dict[str, np.ndarray] = {}
            for item in entries:
                if not isinstance(item, dict):
                    continue
                key = item.get("unicode") or item.get("class") or item.get("label")
                vector = item.get("code") or item.get("vector") or item.get("embedding")
                if key is None or vector is None:
                    continue
                converted[str(key)] = np.asarray(vector, dtype=np.float32)
            if converted:
                return converted

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported codebook format in {codebook_path}: {type(payload)!r}")

    converted: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            converted[str(key)] = np.asarray(value, dtype=np.float32)
    if not converted:
        raise ValueError(f"No vector-based entries were found in {codebook_path}.")
    return converted


def collect_class_samples(data_root: Path, codebook: dict[str, np.ndarray], max_classes: int | None = None, max_samples_per_class: int | None = None) -> tuple[dict[str, int], list[dict[str, Any]]]:
    allowed_aliases: set[str] = set()
    for key in codebook.keys():
        allowed_aliases.update(unicode_aliases(key))

    class_to_index: dict[str, int] = {}
    sorted_codebook_keys = sorted(codebook.keys(), key=lambda value: str(value))
    for index, key in enumerate(sorted_codebook_keys):
        class_to_index[key] = index

    sample_entries: list[dict[str, Any]] = []
    class_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    for book_dir in sorted(data_root.iterdir(), key=lambda item: item.name):
        if not book_dir.is_dir():
            continue
        characters_dir = book_dir / "characters"
        if not characters_dir.is_dir():
            continue

        for unicode_dir in sorted(characters_dir.iterdir(), key=lambda item: item.name):
            if not unicode_dir.is_dir():
                continue
            directory_name = unicode_dir.name
            for class_key in codebook.keys():
                if directory_name in unicode_aliases(class_key):
                    matched_key = str(class_key)
                    break
            else:
                continue

            files = sorted(
                list(unicode_dir.glob("*.jpg"))
                + list(unicode_dir.glob("*.jpeg"))
                + list(unicode_dir.glob("*.png"))
                + list(unicode_dir.glob("*.bmp"))
            )
            if not files:
                continue

            for image_path in files[: max_samples_per_class] if max_samples_per_class is not None else files:
                sample_entries.append(
                    {
                        "book_id": book_dir.name,
                        "unicode": matched_key,
                        "label": class_to_index[matched_key],
                        "image_path": str(image_path),
                    }
                )
                class_samples[matched_key].append(sample_entries[-1])

    if not sample_entries:
        raise RuntimeError(f"No dataset samples were found under {data_root} for any CodeBook class.")

    filtered_classes = sorted(class_samples.keys(), key=lambda item: str(item))
    if max_classes is not None:
        filtered_classes = filtered_classes[:max_classes]

    final_class_to_index: dict[str, int] = {}
    final_entries: list[dict[str, Any]] = []
    for index, class_name in enumerate(filtered_classes):
        final_class_to_index[class_name] = index
        for entry in class_samples[class_name]:
            final_entry = dict(entry)
            final_entry["label"] = final_class_to_index[class_name]
            final_entries.append(final_entry)

    return final_class_to_index, final_entries


class CharacterImageDataset(Dataset):
    def __init__(self, entries: list[dict[str, Any]], image_size: int = 96) -> None:
        self.entries = entries
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image_path = Path(entry["image_path"])

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return {
            "image": tensor,
            "label": int(entry["label"]),
            "unicode": entry["unicode"],
            "book_id": entry["book_id"],
            "image_path": str(image_path),
        }


class STEBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(x)
        return torch.where(values > 0, torch.ones_like(values), -torch.ones_like(values))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class FareRecognitionModel(nn.Module):
    def __init__(self, codebook_dim: int = 1212, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        try:
            weights = models.ResNet34_Weights.DEFAULT
            self.backbone = models.resnet34(weights=weights)
        except AttributeError:
            self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_size * 2, codebook_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        sequence = features.unsqueeze(1)
        _, (hidden, _) = self.lstm(sequence)
        last_hidden = hidden[-2:].transpose(0, 1).contiguous().view(sequence.shape[0], -1)
        logits = self.projection(last_hidden)
        return STEBinarize.apply(logits)


def build_split(labels: list[int], train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    class_to_indices: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        class_to_indices[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(class_to_indices):
        indices = class_to_indices[label][:]
        rng.shuffle(indices)
        split_point = max(1, int(len(indices) * train_ratio)) if len(indices) > 1 else len(indices)
        train_indices.extend(indices[:split_point])
        if len(indices) > split_point:
            val_indices.extend(indices[split_point:])
        elif len(indices) > 1:
            val_indices.append(indices[-1])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def evaluate(model: nn.Module, dataloader: DataLoader, codebook: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            binary_code = model(images)
            logits = binary_code.mm(codebook.t())
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_seen += images.size(0)

    accuracy = total_correct / max(1, total_seen)
    mean_loss = total_loss / max(1, total_seen)
    return mean_loss, accuracy


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training started: epochs={args.epochs}, batch_size={args.batch_size}, device={args.device}")
    print(f"Checkpoint path: {checkpoint_path}")

    codebook = load_codebook(Path(args.codebook))
    if not codebook:
        raise ValueError(f"No entries were loaded from {args.codebook}.")

    class_to_index, entries = collect_class_samples(
        Path(args.data_root),
        codebook,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
    )

    # The class names are the Unicode identifiers, and the index space is zero-based in the same order as the CodeBook.
    if len(class_to_index) == 0:
        raise ValueError("No valid CodeBook classes were found in the dataset.")

    labels = [entry["label"] for entry in entries]
    train_indices, val_indices = build_split(labels, args.train_ratio, args.seed)
    if not train_indices or not val_indices:
        raise ValueError("Training or validation split is empty; use a larger dataset or lower train_ratio.")

    train_entries = [entries[idx] for idx in train_indices]
    val_entries = [entries[idx] for idx in val_indices]

    train_dataset = CharacterImageDataset(train_entries, image_size=args.image_size)
    val_dataset = CharacterImageDataset(val_entries, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    codebook_matrix = torch.stack([torch.tensor(codebook[unicode_key], dtype=torch.float32) for unicode_key in sorted(class_to_index.keys())])
    codebook_matrix = codebook_matrix.to(device)

    model = FareRecognitionModel(codebook_dim=codebook_matrix.shape[1], hidden_size=256, num_layers=2).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            encoded = model(images)
            logits = encoded.mm(codebook_matrix.t())
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_accuracy = evaluate(model, val_loader, codebook_matrix, device)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}")

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
            print(f"Saved best model checkpoint to {checkpoint_path}")

    if best_state_dict is None:
        raise RuntimeError("No model checkpoint was saved during training.")

    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}.")
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
