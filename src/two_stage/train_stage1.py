from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from src.two_stage.experiment import (
    DEFAULT_CONFIG,
    git_metadata,
    load_config,
    loader,
    prepare_data,
    resolve_path,
    save_checkpoint,
    set_seed,
    skeleton_vocabulary,
    stage1_collate,
    start_wandb,
    write_json,
)
from src.two_stage.models.stage1_classifier import Stage1SkeletonClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 skeleton-classification baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _run_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    skeleton_to_id: dict[str, int],
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(data_loader, desc="train" if training else "validation", leave=False):
            known = [
                index
                for index, skeleton in enumerate(batch["skeleton"])
                if _skeleton_key(skeleton) in skeleton_to_id
            ]
            if not known:
                if not training:
                    total += len(batch["skeleton"])
                continue
            images = batch["image"][known].to(device)
            labels = torch.tensor(
                [skeleton_to_id[_skeleton_key(batch["skeleton"][index])] for index in known],
                dtype=torch.long,
                device=device,
            )
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_correct += int(logits.argmax(dim=1).eq(labels).sum())
            total += labels.numel()
            if not training:
                total += len(batch["skeleton"]) - len(known)
    return total_loss / max(1, total), total_correct / max(1, total)


def _skeleton_key(skeleton: list[str] | tuple[str, ...]) -> str:
    import json

    return json.dumps(list(skeleton), ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    set_seed(int(config["experiment"]["seed"]))
    splits, _, summary = prepare_data(config)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Stage 1 requires non-empty train and validation splits.")
    skeleton_to_id, id_to_skeleton = skeleton_vocabulary(splits["train"])
    stage_config = config["stage1"]
    data_config = config["data"]
    device = torch.device(args.device)
    model = Stage1SkeletonClassifier(
        len(skeleton_to_id),
        tuple(stage_config["encoder_channels"]),
        float(stage_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(stage_config["weight_decay"]),
    )
    train_loader = loader(
        splits["train"], int(data_config["image_size"]), int(stage_config["batch_size"]),
        int(data_config["num_workers"]), True,
    )
    train_loader.collate_fn = stage1_collate
    validation_loader = loader(
        splits["validation"], int(data_config["image_size"]), int(stage_config["batch_size"]),
        int(data_config["num_workers"]), False,
    )
    validation_loader.collate_fn = stage1_collate
    output_dir = resolve_path(config["experiment"]["output_dir"]) / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    run = start_wandb(config, "stage1-skeleton-classifier")
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = int(stage_config.get("early_stopping_patience", stage_config["epochs"]))
    min_delta = float(stage_config.get("early_stopping_min_delta", 0.0))
    try:
        run.log(
            {
                f"data/{split_name}_{unit}": count
                for split_name, values in summary.items()
                if split_name.startswith("split_")
                for unit, count in values.items()
            }
        )
        for epoch in range(1, int(stage_config["epochs"]) + 1):
            train_loss, train_accuracy = _run_epoch(model, train_loader, optimizer, device, skeleton_to_id)
            validation_loss, validation_accuracy = _run_epoch(
                model, validation_loader, None, device, skeleton_to_id
            )
            metrics = {
                "epoch": epoch,
                "stage1/train_loss": train_loss,
                "stage1/train_skeleton_accuracy": train_accuracy,
                "stage1/validation_loss": validation_loss,
                "stage1/validation_skeleton_accuracy": validation_accuracy,
            }
            run.log(metrics, step=epoch)
            print(metrics)
            if validation_accuracy > best_accuracy + min_delta:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    stage_config,
                    {"skeleton_to_id": skeleton_to_id, "id_to_skeleton": id_to_skeleton},
                    config,
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print({"stage1/early_stopping_epoch": epoch, "stage1/best_epoch": best_epoch})
                break
        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        run.log(
            {
                "stage1/best_validation_skeleton_accuracy": best_accuracy,
                "stage1/best_epoch": best_epoch,
                "stage1/stopped_epoch": epoch,
            }
        )
        import wandb

        artifact = wandb.Artifact("stage1-skeleton-classifier", type="model")
        artifact.add_file(str(output_dir / "best.pt"))
        run.log_artifact(artifact)
        write_json(output_dir / "summary.json", {"git": git_metadata(), "data": summary})
    finally:
        run.finish()


if __name__ == "__main__":
    main()