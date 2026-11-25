from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.fare_pipeline import merge_ids_entries, read_ids_entries
from src.structured_codebook import IDSNode, STRUCTURE_ARITY, STRUCTURE_SET, canonical_ids, parse_ids_expression
from src.two_stage.pipeline.reconstruct import SLOT, ReverseIDSIndex, tree_to_skeleton_components


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def load_eligible_leaves(codes_path: Path, metadata_path: Path, image_root: Path) -> set[str]:
    with codes_path.open("rb") as handle:
        payload: Any = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported glyph-code payload: {codes_path}")
    raw_codes = payload.get("radical_to_code")
    if not isinstance(raw_codes, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"No radical_to_code mapping or records found in {codes_path}.")
        raw_codes = {
            str(record["radical"]): record["fare_code"]
            for record in records
            if isinstance(record, dict) and record.get("radical") is not None and record.get("fare_code") is not None
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported render metadata schema: {metadata_path}")
    statuses = {
        str(entry.get("text", "")): entry
        for entry in metadata.get("entries", [])
        if isinstance(entry, dict) and entry.get("text")
    }
    eligible: set[str] = set()
    for radical, code in raw_codes.items():
        values = [int(value) for value in code]
        if len(values) != 64 or any(value not in {-1, 1} for value in values):
            continue
        entry = statuses.get(str(radical), {})
        image_path = entry.get("image_path")
        if entry.get("status") != "rendered" or not entry.get("renderable") or not image_path:
            continue
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
            if not resolved.is_file():
                resolved = image_root / Path(image_path).name
        if resolved.is_file():
            eligible.add(str(radical))
    return eligible


def _reverse_subtree_index(
    parsed_ids: dict[str, IDSNode], eligible: set[str]
) -> dict[tuple[Any, ...], list[str]]:
    reverse: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for character, tree in parsed_ids.items():
        if character in eligible:
            reverse[canonical_ids(tree)].append(character)
    for candidates in reverse.values():
        candidates.sort(key=lambda item: (len(item), ord(item[0]) if item else 0, item))
    return dict(reverse)


def resolve_effective_tree(
    node: IDSNode,
    depth: int,
    max_depth: int,
    eligible: set[str],
    parsed_ids: dict[str, IDSNode],
    reverse_subtrees: dict[tuple[Any, ...], list[str]],
    ancestors: frozenset[str] = frozenset(),
) -> tuple[IDSNode, bool]:
    replacement = reverse_subtrees.get(canonical_ids(node), [])
    if node.symbol not in STRUCTURE_SET:
        if node.symbol in eligible:
            return IDSNode(node.symbol), True
        expansion = parsed_ids.get(node.symbol)
        if depth < max_depth and expansion is not None and node.symbol not in ancestors:
            if expansion.symbol != node.symbol:
                return resolve_effective_tree(
                    expansion,
                    depth + 1,
                    max_depth,
                    eligible,
                    parsed_ids,
                    reverse_subtrees,
                    ancestors | {node.symbol},
                )
        return (IDSNode(replacement[0]), True) if replacement else (IDSNode(node.symbol), False)

    if depth >= max_depth:
        return (IDSNode(replacement[0]), True) if replacement else (node, False)

    children: list[IDSNode] = []
    all_resolved = True
    for child in node.children:
        expanded, resolved = resolve_effective_tree(
            child, depth + 1, max_depth, eligible, parsed_ids, reverse_subtrees, ancestors
        )
        children.append(expanded)
        all_resolved = all_resolved and resolved
        if not resolved and replacement:
            return IDSNode(replacement[0]), True
    return IDSNode(node.symbol, tuple(children)), all_resolved


def prepare_class_records(
    data_root: Path,
    ids_files: list[Path],
    radical_codes: Path,
    render_metadata: Path,
    image_root: Path,
    max_ids_depth: int = 6,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], ReverseIDSIndex, dict[str, Any]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    ids_by_character = merge_ids_entries(read_ids_entries(ids_files))
    parsed_ids: dict[str, IDSNode] = {}
    for character, expression in ids_by_character.items():
        try:
            parsed_ids[character] = parse_ids_expression(expression)
        except ValueError:
            continue
    eligible = load_eligible_leaves(radical_codes, render_metadata, image_root)
    reverse_subtrees = _reverse_subtree_index(parsed_ids, eligible)

    class_dirs: dict[str, list[Path]] = defaultdict(list)
    for book_dir in data_root.iterdir():
        character_root = book_dir / "characters"
        if not book_dir.is_dir() or not character_root.is_dir():
            continue
        for class_dir in character_root.iterdir():
            if not class_dir.is_dir():
                continue
            key = class_dir.name
            if key.upper().startswith("U+"):
                try:
                    character = chr(int(key[2:], 16))
                except (ValueError, OverflowError):
                    continue
            elif len(key) == 1:
                character = key
            else:
                continue
            class_dirs[character].append(class_dir)

    rng = random.Random(seed)
    class_names = sorted(class_dirs, key=ord)
    if max_classes is not None and len(class_names) > max_classes:
        class_names = sorted(rng.sample(class_names, max_classes), key=ord)

    records: list[dict[str, Any]] = []
    class_trees: dict[str, IDSNode | None] = {}
    unresolved = 0
    for character in class_names:
        source_tree = parsed_ids.get(character, IDSNode(character))
        effective_tree, resolved = resolve_effective_tree(
            source_tree, 1, max_ids_depth, eligible, parsed_ids, reverse_subtrees
        )
        if not resolved:
            class_trees[character] = None
            unresolved += 1
            continue
        class_trees[character] = effective_tree
        skeleton, components = tree_to_skeleton_components(effective_tree)
        image_paths = sorted(
            path
            for class_dir in class_dirs[character]
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_samples_per_class is not None:
            image_paths = image_paths[:max_samples_per_class]
        depth = _operator_depth(effective_tree)
        for image_path in image_paths:
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "char": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "ids_full": "".join(_tree_to_ids(effective_tree)),
                    "skeleton": skeleton,
                    "components": components,
                    "depth": depth,
                }
            )
    reverse_index = ReverseIDSIndex.from_class_trees(class_trees)
    summary = {
        "selected_classes": len(class_names),
        "resolved_classes": sum(tree is not None for tree in class_trees.values()),
        "unresolved_classes": unresolved,
        "samples": len(records),
        "leaf_vocabulary_size": len({part for record in records for part in record["components"]}),
    }
    return records, reverse_index, summary


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    split_mode: str = "random",
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must satisfy train > 0, validation >= 0, and train + validation < 1.")
    by_character: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_character[record["char"]].append(record)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    if split_mode == "unseen_character":
        characters = sorted(by_character, key=ord)
        rng.shuffle(characters)
        ratios = (train_ratio, validation_ratio, 1.0 - train_ratio - validation_ratio)
        raw_counts = [len(characters) * ratio for ratio in ratios]
        class_counts = [int(value) for value in raw_counts]
        for index in sorted(range(3), key=lambda item: raw_counts[item] - class_counts[item], reverse=True)[
            : len(characters) - sum(class_counts)
        ]:
            class_counts[index] += 1
        if len(characters) >= 3:
            for index, ratio in enumerate(ratios):
                if ratio > 0 and class_counts[index] == 0:
                    donor = max(range(3), key=lambda candidate: class_counts[candidate])
                    class_counts[donor] -= 1
                    class_counts[index] += 1
        boundaries = (class_counts[0], class_counts[0] + class_counts[1])
        character_groups = (
            characters[: boundaries[0]],
            characters[boundaries[0] : boundaries[1]],
            characters[boundaries[1] :],
        )
        for split_name, group in zip(splits, character_groups):
            splits[split_name].extend(
                record for character in group for record in by_character[character]
            )
        return splits
    if split_mode != "random":
        raise ValueError(f"Unsupported split mode: {split_mode}")
    for character in sorted(by_character, key=ord):
        samples = by_character[character][:]
        rng.shuffle(samples)
        count = len(samples)
        train_end = max(1, int(count * train_ratio)) if count else 0
        validation_end = min(count, max(train_end, int(count * (train_ratio + validation_ratio))))
        splits["train"].extend(samples[:train_end])
        splits["validation"].extend(samples[train_end:validation_end])
        splits["test"].extend(samples[validation_end:])
    return splits


def expand_filled_marker_instances(
    record: dict[str, Any],
    input_token_to_id: dict[str, int],
    component_to_id: dict[str, int],
    max_skeleton_length: int | None = None,
) -> list[dict[str, Any]]:
    skeleton = list(record["skeleton"])
    slot_positions = [index for index, token in enumerate(skeleton) if token == SLOT]
    instances: list[dict[str, Any]] = []
    for slot_index, target_position in enumerate(slot_positions):
        state = skeleton[:]
        for previous_position in slot_positions[:slot_index]:
            state[previous_position] = "<FILLED>"
        if max_skeleton_length is not None:
            state.extend(["<PAD>"] * (max_skeleton_length - len(state)))
        target_component = record["components"][slot_index]
        if target_component not in component_to_id:
            continue
        instances.append(
            {
                "image_path": record["image_path"],
                "input_ids": [input_token_to_id[token] for token in state],
                "target_position": target_position,
                "target_id": component_to_id[target_component],
                "component": target_component,
                "char": record["char"],
                "depth": record["depth"],
                "slot_index": slot_index,
                "num_slots": len(slot_positions),
            }
        )
    return instances


def image_to_tensor(image_path: str | Path, image_size: int) -> torch.Tensor:
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Could not load dataset image {image_path}: {exc}") from exc
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = tensor.new_tensor(IMAGE_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGE_STD).view(3, 1, 1)
    return (tensor - mean) / std


class ImageRecordDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {**record, "image": image_to_tensor(record["image_path"], self.image_size)}


class SlotInstanceDataset(Dataset[dict[str, Any]]):
    def __init__(self, instances: list[dict[str, Any]], image_size: int) -> None:
        self.instances = instances
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance = self.instances[index]
        return {
            **instance,
            "input_ids": torch.tensor(instance["input_ids"], dtype=torch.long),
            "image": image_to_tensor(instance["image_path"], self.image_size),
        }


def _operator_depth(node: IDSNode) -> int:
    if node.symbol not in STRUCTURE_SET:
        return 0
    return 1 + max((_operator_depth(child) for child in node.children), default=0)


def _tree_to_ids(node: IDSNode) -> list[str]:
    return [node.symbol, *(token for child in node.children for token in _tree_to_ids(child))]