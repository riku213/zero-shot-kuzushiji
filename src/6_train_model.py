from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import os
import re

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FaRE recognition model from the separated kana/character dataset.")
    parser.add_argument(
        "--data-root",
        default="../kuzushiji-recognition/char_sep_datas",
        help="Root folder containing Book_ID/characters/<Unicode>/image.jpg files.",
    )
    parser.add_argument(
        "--codebook",
        default="outputs/260828_codebook/final_codebook.pkl",
        help="Pickle file containing the final CodeBook mapping Unicode class -> code vector.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/260913_redefine_train_data",
        help="Directory for checkpoints, resume state, and run metadata.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/260913_redefine_train_data/best_fare_model.pth",
        help="Path to save the best fine-tuning checkpoint. Default: outputs/best_fare_model.pth",
    )
    parser.add_argument(
        "--pretrain-checkpoint-path",
        default="outputs/260913_redefine_train_data/pretrain_best_fare_model.pth",
        help="Path to save the best pretraining checkpoint. Default: outputs/pretrain_best_fare_model.pth",
    )
    parser.add_argument(
        "--state-path",
        default="outputs/260913_redefine_train_data/training_state.pth",
        help="Path to save the fine-tuning resume state after each epoch.",
    )
    parser.add_argument(
        "--pretrain-state-path",
        default="outputs/260913_redefine_train_data/pretrain_training_state.pth",
        help="Path to save the pretraining resume state after each epoch.",
    )
    parser.add_argument(
        "--metadata-path",
        default="outputs/260913_redefine_train_data/run_metadata.json",
        help="Path to save the run parameters and split summaries.",
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/manifests/main_manifest.txt",
        help="Optional path to a manifest file (one image path per line) for the main dataset.",
    )
    parser.add_argument(
        "--pretrain-manifest-path",
        default="outputs/manifests/pretrain_manifest.txt",
        help="Optional path to a manifest file (one image path per line) for the pretraining dataset.",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="If set, build manifest files (if manifest paths supplied) before scanning datasets.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for training and validation.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio for each class.")
    parser.add_argument("--image-size", type=int, default=96, help="Input image size (aligned with the FaRE render size).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-classes", type=int, default=None, help="Optional limit on the number of classes used for debugging.")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Optional cap per class for quick validation.")
    parser.add_argument(
        "--pretrain-root",
        default=None,
        help="Optional root directory for a large handwritten-character pretraining dataset. The loader accepts flexible folder structures and only keeps classes present in the target CodeBook.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=1,
        help="Number of pretraining epochs to run before fine-tuning on the kuzushiji dataset.",
    )
    parser.add_argument(
        "--pretrain-max-classes",
        type=int,
        default=None,
        help="Optional cap on pretraining classes for quick validation.",
    )
    parser.add_argument(
        "--pretrain-max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap on pretraining samples per class.",
    )
    parser.add_argument(
        "--pretrain-train-class-ratio",
        type=float,
        default=0.8,
        help="Ratio of pretraining classes to use as training classes (remaining classes become unseen test classes).",
    )
    parser.add_argument(
        "--pretrain-seen-train-ratio",
        type=float,
        default=0.8,
        help="Within training classes, ratio of samples used for training (remaining are seen test samples).",
    )
    parser.add_argument(
        "--pretrain-split-manifest-train",
        default="outputs/260913_redefine_train_data/pretrain_train_manifest.txt",
        help="Output manifest path for pretraining train samples.",
    )
    parser.add_argument(
        "--pretrain-split-manifest-seen-test",
        default="outputs/260913_redefine_train_data/pretrain_seen_test_manifest.txt",
        help="Output manifest path for pretraining seen test samples.",
    )
    parser.add_argument(
        "--pretrain-split-manifest-unseen-test",
        default="outputs/260913_redefine_train_data/pretrain_unseen_test_manifest.txt",
        help="Output manifest path for pretraining unseen test samples.",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def count_unique_classes(entries: list[dict[str, Any]]) -> int:
    return len({str(entry["unicode"]) for entry in entries})


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "classes": count_unique_classes(entries),
        "samples": len(entries),
    }


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def stable_value(seed: int, *parts: str) -> float:
    digest = hashlib.sha1("|".join([str(seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_valid_unicode_codebook_key(key: str) -> bool:
    normalized = normalize_unicode_key(key)
    if re.fullmatch(r"U\+[0-9A-F]{4,6}", normalized):
        return True
    return len(normalized) == 1 and not normalized.isdigit()


def is_noise_candidate_token(token: str) -> bool:
    text = token.strip()
    if not text:
        return True
    low = text.lower()
    if low in {"train", "test", "characters", "casia-hwdb", "casia-hwdb_train", "casia-hwdb_test"}:
        return True
    if text.isdigit():
        return True
    return False


def extract_pretrain_candidate_names(image_path: Path) -> list[str]:
    candidates: list[str] = []

    parent_name = image_path.parent.name.strip()
    if parent_name and not is_noise_candidate_token(parent_name):
        candidates.append(parent_name)

    stem = image_path.stem.strip()
    if stem and not stem.isdigit():
        parts = re.split(r"[\.・_\- ]+", stem)
        for part in parts:
            part = part.strip()
            if not part or is_noise_candidate_token(part):
                continue
            candidates.append(part)

    return candidates


def resolve_pretrain_true_label(image_path: Path, codebook: dict[str, np.ndarray], allowed_keys: set[str]) -> str | None:
    for candidate in extract_pretrain_candidate_names(image_path):
        resolved = resolve_codebook_label(candidate, codebook)
        if resolved is not None and resolved in allowed_keys:
            return resolved
    return None


def write_manifest(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry['image_path']}\n")


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


def resolve_codebook_label(candidate: str, codebook: dict[str, np.ndarray]) -> str | None:
    normalized = normalize_unicode_key(candidate)
    if not normalized:
        return None

    # direct match
    if normalized in codebook:
        return normalized

    # class-key aliases (U+XXXX or character)
    for class_key in codebook:
        aliases = unicode_aliases(class_key)
        if normalized in aliases:
            return str(class_key)

    # Heuristics for CASIA-like numeric or hex labels:
    # - decimal codepoint (e.g. '19968')
    # - hex codepoint present in names (e.g. '4E00')
    # - mixed prefixes/suffixes containing digits/hex
    try:
        if normalized.isdigit():
            cp = int(normalized, 10)
            if 0 <= cp <= 0x10FFFF:
                candidate_char = chr(cp)
                # check direct char and U+XXXX form
                if candidate_char in codebook:
                    return candidate_char
                ukey = f"U+{cp:04X}".upper()
                if ukey in codebook:
                    return ukey
    except Exception:
        pass

    # search for a decimal substring
    m = re.search(r"(\d{3,6})", normalized)
    if m:
        try:
            cp = int(m.group(1), 10)
            if 0 <= cp <= 0x10FFFF:
                candidate_char = chr(cp)
                if candidate_char in codebook:
                    return candidate_char
                ukey = f"U+{cp:04X}".upper()
                if ukey in codebook:
                    return ukey
        except Exception:
            pass

    # search for a hex substring
    m2 = re.search(r"([0-9A-Fa-f]{4,6})", normalized)
    if m2:
        try:
            cp = int(m2.group(1), 16)
            if 0 <= cp <= 0x10FFFF:
                candidate_char = chr(cp)
                if candidate_char in codebook:
                    return candidate_char
                ukey = f"U+{cp:04X}".upper()
                if ukey in codebook:
                    return ukey
        except Exception:
            pass

    return None


def build_manifest(root: Path, manifest_path: Path, resume: bool = True) -> int:
    """Build a manifest file listing all image files under root.

    If resume is True and manifest_path exists, existing entries are respected and new
    files are appended. Returns total number of entries in the final manifest.
    """
    image_ext = (".png", ".jpg", ".jpeg", ".bmp")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if resume and manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    existing.add(line.strip())
        except Exception:
            existing = set()

    count = len(existing)

    # Open in append mode to add missing entries
    with manifest_path.open("a", encoding="utf-8") as fh:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith(image_ext):
                    continue
                full = os.path.join(dp, fn)
                if full in existing:
                    continue
                fh.write(full + "\n")
                existing.add(full)
                count += 1

    return count


def collect_class_samples(data_root: Path, codebook: dict[str, np.ndarray], max_classes: int | None = None, max_samples_per_class: int | None = None, manifest_path: Path | None = None, allow_empty: bool = False) -> tuple[dict[str, int], list[dict[str, Any]]]:
    class_to_index: dict[str, int] = {}
    sorted_codebook_keys = sorted(codebook.keys(), key=lambda value: str(value))
    for index, key in enumerate(sorted_codebook_keys):
        class_to_index[key] = index

    sample_entries: list[dict[str, Any]] = []
    class_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def iter_paths_from_manifest(path: Path):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                p = Path(line.strip())
                if p and p.suffix.lower() in image_extensions:
                    yield p

    if manifest_path is not None and manifest_path.exists():
        iterator = iter_paths_from_manifest(manifest_path)
        iterator = tqdm(iterator, desc=f"Reading manifest {manifest_path.name}", unit="path", leave=False)
    else:
        # Iterate recursively with a progress indicator; large datasets (CASIA) can take long to scan.
        iterator = data_root.rglob("*")
        iterator = tqdm(iterator, desc=f"Scanning {data_root}", unit="path", leave=False)

    for image_path in iterator:
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue

        candidate_names: list[str] = []
        candidate_names.extend([image_path.stem, image_path.parent.name])
        for parent in image_path.parents:
            if parent == data_root:
                break
            candidate_names.append(parent.name)

        # Expand candidate tokens to handle CASIA-style filenames with numbers and delimiters.
        expanded_candidates: list[str] = []
        for cand in candidate_names:
            if not cand:
                continue
            # strip trailing numeric suffixes like '_123' or ' 123'
            no_num = re.sub(r"[ _\-]*\d+$", "", cand)
            # split on common delimiters (middle dot, underscores, hyphens, spaces, dots)
            parts = re.split(r"[\.・_\- ]+", no_num)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                expanded_candidates.append(part)
                # also try individual characters (common for CJK labels)
                for ch in part:
                    expanded_candidates.append(ch)

        # ensure original candidates are considered first
        candidate_names = candidate_names + expanded_candidates

        matched_key: str | None = None
        for candidate in candidate_names:
            resolved = resolve_codebook_label(candidate, codebook)
            if resolved is not None:
                matched_key = resolved
                break

        if matched_key is None:
            continue

        if max_samples_per_class is not None and len(class_samples[matched_key]) >= max_samples_per_class:
            continue

        entry = {
            "book_id": data_root.name,
            "unicode": matched_key,
            "label": class_to_index[matched_key],
            "image_path": str(image_path),
        }
        sample_entries.append(entry)
        class_samples[matched_key].append(entry)

    if not sample_entries:
        if allow_empty:
            return {}, []
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


def collect_pretrain_entries_with_redefined_split(
    data_root: Path,
    codebook: dict[str, np.ndarray],
    seed: int,
    train_class_ratio: float,
    seen_train_ratio: float,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not 0.0 < train_class_ratio < 1.0:
        raise ValueError("pretrain_train_class_ratio must be between 0 and 1.")
    if not 0.0 < seen_train_ratio < 1.0:
        raise ValueError("pretrain_seen_train_ratio must be between 0 and 1.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    allowed_keys = {str(key) for key in codebook if is_valid_unicode_codebook_key(str(key))}
    class_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def iter_paths_from_manifest(path: Path):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                p = Path(line.strip())
                if p and p.suffix.lower() in image_extensions:
                    yield p

    iterator = None
    if manifest_path is not None and manifest_path.exists():
        iterator = iter_paths_from_manifest(manifest_path)
        iterator = tqdm(iterator, desc=f"Reading manifest {manifest_path.name}", unit="img")
    else:
        iterator = data_root.rglob("*")
        iterator = tqdm(iterator, desc=f"Scanning {data_root}", unit="path")

    total_images = 0
    unresolved_images = 0
    invalid_label_images = 0

    for image_path in iterator:
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue
        total_images += 1

        resolved = resolve_pretrain_true_label(image_path, codebook, allowed_keys)
        if resolved is None:
            unresolved_images += 1
            continue

        if resolved not in allowed_keys:
            invalid_label_images += 1
            continue

        if max_samples_per_class is not None and len(class_samples[resolved]) >= max_samples_per_class:
            continue

        class_samples[resolved].append(
            {
                "book_id": data_root.name,
                "unicode": resolved,
                "image_path": str(image_path),
            }
        )

    classes = sorted(class_samples.keys(), key=lambda item: str(item))
    if max_classes is not None:
        classes = classes[:max_classes]

    if not classes:
        return {}, [], [], {
            "total_images": total_images,
            "resolved_images": 0,
            "unresolved_images": unresolved_images,
            "invalid_label_images": invalid_label_images,
            "train_classes": 0,
            "unseen_classes": 0,
            "seen_test_classes": 0,
            "train_samples": 0,
            "seen_test_samples": 0,
            "unseen_test_samples": 0,
        }

    eligible_seen_classes = [cls for cls in classes if len(class_samples[cls]) >= 2]
    forced_unseen_classes = [cls for cls in classes if len(class_samples[cls]) < 2]

    train_class_set: set[str] = set()
    unseen_class_set: set[str] = set(forced_unseen_classes)
    for cls in eligible_seen_classes:
        if stable_value(seed, "pretrain-class", cls) < train_class_ratio:
            train_class_set.add(cls)
        else:
            unseen_class_set.add(cls)

    if eligible_seen_classes and not train_class_set:
        train_class_set.add(eligible_seen_classes[0])
        unseen_class_set.discard(eligible_seen_classes[0])
    if eligible_seen_classes and not (unseen_class_set - set(forced_unseen_classes)) and len(eligible_seen_classes) > 1:
        moved = sorted(train_class_set)[-1]
        train_class_set.discard(moved)
        unseen_class_set.add(moved)

    all_split_classes = sorted(train_class_set | unseen_class_set, key=lambda item: str(item))
    class_to_index = {name: idx for idx, name in enumerate(all_split_classes)}

    train_entries: list[dict[str, Any]] = []
    seen_test_entries: list[dict[str, Any]] = []
    unseen_test_entries: list[dict[str, Any]] = []

    for cls in tqdm(all_split_classes, desc="Building split", unit="class"):
        cls_entries = class_samples[cls]
        if cls in unseen_class_set:
            for entry in cls_entries:
                item = dict(entry)
                item["label"] = class_to_index[cls]
                item["split"] = "unseen_test"
                unseen_test_entries.append(item)
            continue

        local_train: list[dict[str, Any]] = []
        local_seen_test: list[dict[str, Any]] = []
        for entry in cls_entries:
            if stable_value(seed, "pretrain-sample", cls, entry["image_path"]) < seen_train_ratio:
                local_train.append(entry)
            else:
                local_seen_test.append(entry)

        if not local_seen_test and len(local_train) > 1:
            local_seen_test.append(local_train.pop())
        if not local_train and local_seen_test:
            local_train.append(local_seen_test.pop())

        for entry in local_train:
            item = dict(entry)
            item["label"] = class_to_index[cls]
            item["split"] = "train"
            train_entries.append(item)
        for entry in local_seen_test:
            item = dict(entry)
            item["label"] = class_to_index[cls]
            item["split"] = "seen_test"
            seen_test_entries.append(item)

    val_entries = seen_test_entries + unseen_test_entries
    split_summary = {
        "total_images": total_images,
        "resolved_images": sum(len(class_samples[c]) for c in all_split_classes),
        "unresolved_images": unresolved_images,
        "invalid_label_images": invalid_label_images,
        "train_classes": len(train_class_set),
        "seen_test_classes": len({entry["unicode"] for entry in seen_test_entries}),
        "unseen_classes": len(unseen_class_set),
        "train_samples": len(train_entries),
        "seen_test_samples": len(seen_test_entries),
        "unseen_test_samples": len(unseen_test_entries),
    }

    return class_to_index, train_entries, val_entries, split_summary


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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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


def train_model_on_dataset(
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
    verbose: bool = True,
) -> tuple[float, float]:
    train_dataset = CharacterImageDataset(train_entries, image_size=96)
    val_dataset = CharacterImageDataset(val_entries, image_size=96)
    num_workers = 0
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    start_epoch = 1

    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            model_state_dict = state.get("model_state_dict")
            optimizer_state_dict = state.get("optimizer_state_dict")
            if isinstance(model_state_dict, dict):
                model.load_state_dict(model_state_dict)
            if isinstance(optimizer_state_dict, dict):
                optimizer.load_state_dict(optimizer_state_dict)
                move_optimizer_state_to_device(optimizer, device)
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
        return best_accuracy, best_epoch

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"Train E{epoch}", leave=False):
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

    return best_accuracy, best_epoch


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pretrain_checkpoint_path = Path(args.pretrain_checkpoint_path)
    pretrain_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pretrain_state_path = Path(args.pretrain_state_path)
    pretrain_state_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path) if getattr(args, "manifest_path", None) else None
    pretrain_manifest_path = Path(args.pretrain_manifest_path) if getattr(args, "pretrain_manifest_path", None) else None

    # Optionally build manifest files before scanning (can be faster than repeated rglob)
    if args.build_manifest:
        if pretrain_manifest_path and args.pretrain_root:
            print(f"Building pretrain manifest {pretrain_manifest_path} from {args.pretrain_root}...")
            n = build_manifest(Path(args.pretrain_root), pretrain_manifest_path)
            print(f"Wrote {n} entries to {pretrain_manifest_path}")
        if manifest_path:
            print(f"Building main manifest {manifest_path} from {args.data_root}...")
            n = build_manifest(Path(args.data_root), manifest_path)
            print(f"Wrote {n} entries to {manifest_path}")

    print(f"Training started: epochs={args.epochs}, batch_size={args.batch_size}, device={args.device}")
    if args.pretrain_root:
        print(f"Pretraining dataset: {args.pretrain_root}")
    print(f"Fine-tuning checkpoint path: {checkpoint_path}")
    print(f"Fine-tuning resume state path: {state_path}")
    if args.pretrain_root:
        print(f"Pretraining checkpoint path: {pretrain_checkpoint_path}")
        print(f"Pretraining resume state path: {pretrain_state_path}")

    codebook = load_codebook(Path(args.codebook))
    if not codebook:
        raise ValueError(f"No entries were loaded from {args.codebook}.")

    device = torch.device(args.device)
    if args.pretrain_root:
        pretrain_root = Path(args.pretrain_root)
        pretrain_class_to_index, pretrain_train_entries, pretrain_val_entries, pretrain_split_summary = collect_pretrain_entries_with_redefined_split(
            pretrain_root,
            codebook,
            seed=args.seed,
            train_class_ratio=args.pretrain_train_class_ratio,
            seen_train_ratio=args.pretrain_seen_train_ratio,
            max_classes=args.pretrain_max_classes,
            max_samples_per_class=args.pretrain_max_samples_per_class,
            manifest_path=pretrain_manifest_path,
        )
        if pretrain_train_entries or pretrain_val_entries:
            write_manifest(pretrain_train_entries, Path(args.pretrain_split_manifest_train))
            write_manifest([entry for entry in pretrain_val_entries if entry.get("split") == "seen_test"], Path(args.pretrain_split_manifest_seen_test))
            write_manifest([entry for entry in pretrain_val_entries if entry.get("split") == "unseen_test"], Path(args.pretrain_split_manifest_unseen_test))

            print(f"Using {len(pretrain_class_to_index)} classes from pretraining dataset {pretrain_root}")
            print(
                "Pretrain redefined split summary: "
                f"total_images={pretrain_split_summary['total_images']} resolved={pretrain_split_summary['resolved_images']} unresolved={pretrain_split_summary['unresolved_images']} | "
                f"train_classes={pretrain_split_summary['train_classes']} train_samples={pretrain_split_summary['train_samples']} | "
                f"seen_test_classes={pretrain_split_summary['seen_test_classes']} seen_test_samples={pretrain_split_summary['seen_test_samples']} | "
                f"unseen_classes={pretrain_split_summary['unseen_classes']} unseen_test_samples={pretrain_split_summary['unseen_test_samples']}"
            )

            pretrain_codebook_vectors = [torch.tensor(codebook[unicode_key], dtype=torch.float32) for unicode_key in sorted(pretrain_class_to_index.keys())]
            pretrain_codebook_matrix = torch.stack(pretrain_codebook_vectors).to(device)
            pretrain_model = FareRecognitionModel(codebook_dim=pretrain_codebook_matrix.shape[1], hidden_size=256, num_layers=2).to(device)
            best_pre_acc, best_pre_epoch = train_model_on_dataset(
                model=pretrain_model,
                train_entries=pretrain_train_entries,
                val_entries=pretrain_val_entries,
                codebook_matrix=pretrain_codebook_matrix,
                device=device,
                batch_size=args.batch_size,
                epochs=args.pretrain_epochs,
                checkpoint_path=pretrain_checkpoint_path,
                state_path=pretrain_state_path,
                class_to_index=pretrain_class_to_index,
                seed=args.seed,
                verbose=True,
            )
            print(f"Pretraining complete. Best validation accuracy on pretraining data: {best_pre_acc:.4f} at epoch {best_pre_epoch}.")
        else:
            print(f"No pretraining samples found under {pretrain_root}. Continuing without pretraining.")

    class_to_index, entries = collect_class_samples(
        Path(args.data_root),
        codebook,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
        manifest_path=manifest_path,
    )

    if len(class_to_index) == 0:
        raise ValueError("No valid CodeBook classes were found in the dataset.")

    labels = [entry["label"] for entry in entries]
    train_indices, val_indices = build_split(labels, args.train_ratio, args.seed)
    if not train_indices or not val_indices:
        raise ValueError("Training or validation split is empty; use a larger dataset or lower train_ratio.")

    train_entries = [entries[idx] for idx in train_indices]
    val_entries = [entries[idx] for idx in val_indices]
    print(
        "Fine-tuning split summary: "
        f"classes={len(class_to_index)} samples={len(entries)} | "
        f"train_classes={count_unique_classes(train_entries)} train_samples={len(train_entries)} | "
        f"val_classes={count_unique_classes(val_entries)} val_samples={len(val_entries)}"
    )

    codebook_matrix = torch.stack([torch.tensor(codebook[unicode_key], dtype=torch.float32) for unicode_key in sorted(class_to_index.keys())])
    codebook_matrix = codebook_matrix.to(device)

    model = FareRecognitionModel(codebook_dim=codebook_matrix.shape[1], hidden_size=256, num_layers=2).to(device)
    if args.pretrain_root and 'pretrain_model' in locals():
        model.load_state_dict(pretrain_model.state_dict())

    best_accuracy, best_epoch = train_model_on_dataset(
        model=model,
        train_entries=train_entries,
        val_entries=val_entries,
        codebook_matrix=codebook_matrix,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        state_path=state_path,
        class_to_index=class_to_index,
        seed=args.seed,
        verbose=True,
    )

    metadata = {
        "output_dir": str(output_dir),
        "data_root": str(args.data_root),
        "pretrain_root": str(args.pretrain_root) if args.pretrain_root else None,
        "codebook": str(args.codebook),
        "train_ratio": args.train_ratio,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "checkpoint_path": str(checkpoint_path),
        "state_path": str(state_path),
        "pretrain_checkpoint_path": str(pretrain_checkpoint_path),
        "pretrain_state_path": str(pretrain_state_path),
        "pretrain_split_summary": {
            "classes": len(pretrain_class_to_index) if args.pretrain_root else 0,
            "train": summarize_entries(pretrain_train_entries) if args.pretrain_root and (pretrain_train_entries or pretrain_val_entries) else {"classes": 0, "samples": 0},
            "val": summarize_entries(pretrain_val_entries) if args.pretrain_root and (pretrain_train_entries or pretrain_val_entries) else {"classes": 0, "samples": 0},
            "redefined": pretrain_split_summary if args.pretrain_root and (pretrain_train_entries or pretrain_val_entries) else None,
        } if args.pretrain_root else None,
        "fine_tuning_split_summary": {
            "classes": len(class_to_index),
            "train": summarize_entries(train_entries),
            "val": summarize_entries(val_entries),
        },
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
    }
    save_json(metadata_path, metadata)
    print(f"Saved run metadata to {metadata_path}")
    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}.")
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
