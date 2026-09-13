from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pickle
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainmod = load_module("trainmod", "src/6_train_model.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CASIA pretraining predictions and seen/unseen evaluation.")
    parser.add_argument("--pretrain-root", required=True, help="CASIA-HWDB root directory.")
    parser.add_argument(
        "--manifest-path",
        default="outputs/manifests/pretrain_manifest.txt",
        help="Manifest file for the pretraining dataset.",
    )
    parser.add_argument(
        "--reference-codebook",
        default="outputs/260828_codebook/final_codebook.pkl",
        help="Reference codebook used to define seen classes (IDS + Hanazono baseline).",
    )
    parser.add_argument(
        "--prediction-codebook",
        default="outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl",
        help="Expanded codebook used for prediction.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/pretrain_best_fare_model.pth",
        help="Pretraining checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/260901_codebook_CASIA/results",
        help="Directory for the output text report.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="Number of sample predictions to export.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Deterministic holdout ratio within seen classes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling and deterministic splits.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of selected test samples to buffer before running inference.",
    )
    return parser.parse_args()


def unicode_to_char(label: str) -> str:
    text = str(label).strip()
    if text.startswith("U+"):
        try:
            return chr(int(text[2:], 16))
        except ValueError:
            return text
    return text


def count_manifest_lines(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def extract_candidate_names(image_path: Path, data_root: Path) -> list[str]:
    candidate_names: list[str] = [image_path.stem, image_path.parent.name]
    for parent in image_path.parents:
        if parent == data_root:
            break
        candidate_names.append(parent.name)

    expanded_candidates: list[str] = []
    for cand in candidate_names:
        if not cand:
            continue
        no_num = re.sub(r"[ _\-]*\d+$", "", cand)
        parts = re.split(r"[\.・_\- ]+", no_num)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            expanded_candidates.append(part)
            expanded_candidates.extend(list(part))

    return candidate_names + expanded_candidates


def stable_value(seed: int, class_key: str, image_path: str) -> float:
    digest = hashlib.sha1(f"{seed}|{class_key}|{image_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def reservoir_update(reservoir: list[dict[str, Any]], item: dict[str, Any], seen: int, capacity: int, rng: random.Random) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    index = rng.randrange(seen)
    if index < capacity:
        reservoir[index] = item


class ImageBatchDataset(Dataset):
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
            "image_path": str(image_path),
            "unicode": entry["unicode"],
            "group": entry["group"],
        }


def resolve_label(image_path: Path, data_root: Path, codebook: dict[str, np.ndarray]) -> str | None:
    for candidate in extract_candidate_names(image_path, data_root):
        resolved = trainmod.resolve_codebook_label(candidate, codebook)
        if resolved is not None:
            return resolved
    return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    reference_codebook = trainmod.load_codebook(Path(args.reference_codebook))
    prediction_codebook = trainmod.load_codebook(Path(args.prediction_codebook))
    if not prediction_codebook:
        raise ValueError(f"No entries were loaded from {args.prediction_codebook}.")

    reference_classes = set(reference_codebook.keys())
    prediction_keys = sorted(prediction_codebook.keys(), key=str)
    codebook_matrix = torch.stack([torch.tensor(prediction_codebook[key], dtype=torch.float32) for key in prediction_keys])

    checkpoint = torch.load(Path(args.checkpoint_path), map_location="cpu", weights_only=False)
    codebook_dim = int(checkpoint.get("codebook_dim", codebook_matrix.shape[1]))
    model = trainmod.FareRecognitionModel(codebook_dim=codebook_dim, hidden_size=256, num_layers=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device(args.device)
    model.to(device)
    codebook_matrix = codebook_matrix.to(device)

    manifest_path = Path(args.manifest_path)
    data_root = Path(args.pretrain_root)
    total_lines = count_manifest_lines(manifest_path)

    seen_class_set: set[str] = set()
    unseen_class_set: set[str] = set()
    seen_total = 0
    unseen_total = 0
    seen_test_total = 0
    unseen_test_total = 0
    seen_test_correct = 0
    unseen_test_correct = 0
    selected_rows: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    report_seen_total = 0

    print(f"Reference codebook classes: {len(reference_classes)}")
    print(f"Prediction codebook classes: {len(prediction_codebook)}")
    print(f"Scanning manifest {manifest_path} ...")

    def flush_pending(entries: list[dict[str, Any]]) -> None:
        nonlocal seen_test_total, unseen_test_total, seen_test_correct, unseen_test_correct, report_seen_total
        if not entries:
            return

        # Pre-validate images to avoid crashing DataLoader on corrupted files.
        valid_entries: list[dict[str, Any]] = []
        skipped_paths: list[str] = []
        for e in entries:
            image_path = Path(e["image_path"])
            try:
                # verify header only (fast) and close immediately
                with Image.open(image_path) as img:
                    img.verify()
                valid_entries.append(e)
            except (UnidentifiedImageError, OSError) as ex:
                skipped_paths.append(str(image_path))

        if skipped_paths:
            print(f"Skipped {len(skipped_paths)} unreadable images (first: {skipped_paths[0]})")

        if not valid_entries:
            return

        dataset = ImageBatchDataset(valid_entries, image_size=96)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting chunks", leave=False, unit="batch"):
                images = batch["image"].to(device)
                binary_code = model(images)
                logits = binary_code.mm(codebook_matrix.t())
                predictions = logits.argmax(dim=1)

                for index in range(images.size(0)):
                    pred_key = prediction_keys[int(predictions[index])]
                    true_key = batch["unicode"][index]
                    group = batch["group"][index]
                    image_path = batch["image_path"][index]
                    pred_char = unicode_to_char(pred_key)
                    true_char = unicode_to_char(true_key)
                    fare_code = "".join("1" if value > 0 else "0" for value in binary_code[index].tolist())

                    row = {
                        "image_path": image_path,
                        "fare_code": fare_code,
                        "predicted_unicode": pred_key,
                        "predicted_char": pred_char,
                        "true_unicode": true_key,
                        "true_char": true_char,
                        "group": group,
                    }

                    report_seen_total += 1
                    reservoir_update(selected_rows, row, report_seen_total, args.sample_count, rng)

                    if group == "seen":
                        seen_test_total += 1
                        seen_test_correct += int(pred_key == true_key)
                    else:
                        unseen_test_total += 1
                        unseen_test_correct += int(pred_key == true_key)

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total_lines, desc="Scanning pretrain manifest", unit="img"):
            item = line.strip()
            if not item:
                continue

            image_path = Path(item)
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue

            resolved = resolve_label(image_path, data_root, prediction_codebook)
            if resolved is None:
                continue

            if resolved in reference_classes:
                seen_class_set.add(resolved)
                seen_total += 1
                # Hold out a deterministic portion of seen-class samples for testing.
                if stable_value(args.seed, resolved, str(image_path)) >= args.train_ratio:
                    pending_entries.append({"image_path": str(image_path), "unicode": resolved, "group": "seen"})
            else:
                unseen_class_set.add(resolved)
                unseen_total += 1
                pending_entries.append({"image_path": str(image_path), "unicode": resolved, "group": "unseen"})

            if len(pending_entries) >= args.chunk_size:
                flush_pending(pending_entries)
                pending_entries = []

    flush_pending(pending_entries)

    print(f"Seen classes (CASIA ∩ reference codebook): {len(seen_class_set)}")
    print(f"Seen samples (all CASIA samples in reference classes): {seen_total}")
    print(f"Unseen classes (CASIA minus reference codebook): {len(unseen_class_set)}")
    print(f"Unseen samples (all CASIA samples outside reference classes): {unseen_total}")
    print(f"Seen test samples (held out from seen classes): {seen_test_total}")
    print(f"Unseen test samples: {unseen_test_total}")

    if seen_test_total > 0:
        print(f"Seen-class accuracy: {seen_test_correct / seen_test_total:.4f}")
    else:
        print("Seen-class accuracy: n/a (no held-out seen samples)")

    if unseen_test_total > 0:
        print(f"Unseen-class accuracy: {unseen_test_correct / unseen_test_total:.4f}")
    else:
        print("Unseen-class accuracy: n/a (no unseen samples)")

    if not selected_rows:
        raise RuntimeError("No test samples were selected. Check the manifest, codebook, and dataset root.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pretrain_results_{args.sample_count}.txt"

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("image_path\tfare_code\tpredicted_unicode\tpredicted_char\ttrue_unicode\ttrue_char\tgroup\n")
        for row in selected_rows:
            handle.write(
                f"{row['image_path']}\t{row['fare_code']}\t{row['predicted_unicode']}\t{row['predicted_char']}\t{row['true_unicode']}\t{row['true_char']}\t{row['group']}\n"
            )

    overall_total = seen_test_total + unseen_test_total
    overall_correct = seen_test_correct + unseen_test_correct
    print(f"Wrote sample report to {output_path}")
    print(f"Selected-sample accuracy: {overall_correct}/{overall_total} = {overall_correct / max(1, overall_total):.4f}")


if __name__ == "__main__":
    main()