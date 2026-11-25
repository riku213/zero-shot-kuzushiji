from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.two_stage.data.pipeline import (
    ImageRecordDataset,
    SlotInstanceDataset,
    expand_filled_marker_instances,
    prepare_class_records,
    split_records,
)
from src.structured_codebook import STRUCTURE_SYMBOLS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "two_stage_default.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    if config.get("stage2", {}).get("input_mode") != "filled_marker":
        raise ValueError("Milestone 3 currently supports only stage2.input_mode=filled_marker.")
    if config.get("stage2", {}).get("glyph_code_mode") != "none":
        raise ValueError("Milestone 3 requires stage2.glyph_code_mode=none.")
    if config.get("stage2", {}).get("guidance_loss_weight", 0) != 0:
        raise ValueError("Milestone 3 keeps guidance loss disabled.")
    if config.get("stage2", {}).get("triplet_loss_weight", 0) != 0:
        raise ValueError("Milestone 3 keeps triplet loss disabled.")
    return config


def resolve_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], Any, dict[str, Any]]:
    data_config = config["data"]
    experiment_config = config["experiment"]
    records, reverse_index, summary = prepare_class_records(
        data_root=resolve_path(data_config["root"]),
        ids_files=[resolve_path(path) for path in data_config["ids_files"]],
        radical_codes=resolve_path(data_config["radical_codes"]),
        render_metadata=resolve_path(data_config["render_metadata"]),
        image_root=resolve_path(data_config["image_root"]),
        max_ids_depth=int(data_config["max_ids_depth"]),
        max_classes=experiment_config.get("max_classes"),
        max_samples_per_class=experiment_config.get("max_samples_per_class"),
        seed=int(experiment_config["seed"]),
    )
    splits = split_records(
        records,
        float(experiment_config["train_ratio"]),
        float(experiment_config["validation_ratio"]),
        int(experiment_config["seed"]),
        str(experiment_config.get("split", "random")),
    )
    summary["split_samples"] = {key: len(value) for key, value in splits.items()}
    summary["split_classes"] = {
        key: len({record["char"] for record in value}) for key, value in splits.items()
    }
    return splits, reverse_index, summary


def skeleton_vocabulary(train_records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, list[str]]]:
    skeletons = sorted({tuple(record["skeleton"]) for record in train_records})
    if not skeletons:
        raise ValueError("No skeleton classes found in training records.")
    token_to_id = {json.dumps(skeleton, ensure_ascii=False): index for index, skeleton in enumerate(skeletons)}
    id_to_skeleton = {index: list(skeleton) for index, skeleton in enumerate(skeletons)}
    return token_to_id, id_to_skeleton


def stage2_vocabularies(
    leaf_set: set[str], records: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, int]]:
    input_tokens = ["<PAD>", "□", "<FILLED>", *STRUCTURE_SYMBOLS]
    input_vocab = {token: index for index, token in enumerate(input_tokens)}
    components = sorted(leaf_set)
    if not components:
        components = sorted({part for record in records for part in record["components"]})
    return input_vocab, {token: index for index, token in enumerate(components)}


def loader(
    records: list[dict[str, Any]], image_size: int, batch_size: int, workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ImageRecordDataset(records, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def slot_loader(
    records: list[dict[str, Any]],
    input_vocab: dict[str, int],
    component_vocab: dict[str, int],
    image_size: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    max_skeleton_length: int | None = None,
) -> DataLoader:
    if not records:
        raise ValueError("Cannot construct a Stage 2 loader from an empty split.")
    if max_skeleton_length is None:
        max_skeleton_length = max(len(record["skeleton"]) for record in records)
    instances = [
        instance
        for record in records
        for instance in expand_filled_marker_instances(
            record,
            input_vocab,
            component_vocab,
            max_skeleton_length=max_skeleton_length,
        )
    ]
    if not instances:
        raise ValueError("No Stage 2 slot instances were created.")
    dataset = SlotInstanceDataset(instances, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def stage1_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "skeleton": [item["skeleton"] for item in batch],
        "char": [item["char"] for item in batch],
    }


def git_metadata() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current")}


def start_wandb(config: dict[str, Any], run_name: str):
    import wandb

    tracking = config["tracking"]
    mode = os.environ.get("WANDB_MODE", tracking.get("wandb_mode", "online"))
    wandb_dir = tracking.get("wandb_dir")
    return wandb.init(
        project=tracking.get("wandb_project", "zero-shot-kuzushiji"),
        entity=tracking.get("wandb_entity"),
        mode=mode,
        name=run_name,
        dir=str(resolve_path(wandb_dir)) if wandb_dir else None,
        config={"experiment_config": config, "git": git_metadata()},
        reinit=True,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: dict[str, Any],
    vocabularies: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": model_config,
            "vocabularies": vocabularies,
            "experiment_config": config,
            "git": git_metadata(),
        },
        path,
    )


def log_checkpoint_artifact(run: Any, path: Path, name: str) -> None:
    import wandb

    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")