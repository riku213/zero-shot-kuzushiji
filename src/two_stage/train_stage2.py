from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.two_stage.data.pipeline import load_eligible_leaves
from src.two_stage.experiment import (
    DEFAULT_CONFIG,
    git_metadata,
    load_config,
    prepare_data,
    resolve_path,
    save_checkpoint,
    set_seed,
    slot_loader,
    stage2_vocabularies,
    start_wandb,
    write_json,
)
from src.two_stage.models.stage2_filler import Stage2SlotFiller, slot_only_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 using oracle skeletons and filled_marker inputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _run_epoch(model, data_loader, optimizer, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(data_loader, desc="train" if training else "validation", leave=False):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            positions = batch["target_position"].to(device)
            targets = batch["target_id"].to(device)
            logits = model(images, input_ids)
            loss = slot_only_cross_entropy(logits, positions, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total += targets.numel()
    return total_loss / max(1, total)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    set_seed(int(config["experiment"]["seed"]))
    splits, _, summary = prepare_data(config)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Stage 2 requires non-empty train and validation splits.")
    data_config = config["data"]
    stage_config = config["stage2"]
    leaf_set = load_eligible_leaves(
        resolve_path(data_config["radical_codes"]),
        resolve_path(data_config["render_metadata"]),
        resolve_path(data_config["image_root"]),
    )
    all_records = [record for split in splits.values() for record in split]
    input_vocab, component_vocab = stage2_vocabularies(leaf_set, all_records)
    pad_id = input_vocab["<PAD>"]
    max_skeleton_length = max(len(record["skeleton"]) for record in all_records)
    device = torch.device(args.device)
    model = Stage2SlotFiller(
        input_vocabulary_size=len(input_vocab),
        component_vocabulary_size=len(component_vocab),
        max_skeleton_length=max_skeleton_length,
        pad_id=pad_id,
        image_size=int(data_config["image_size"]),
        patch_size=int(stage_config["patch_size"]),
        d_model=int(stage_config["d_model"]),
        encoder_layers=int(stage_config["encoder_layers"]),
        decoder_layers=int(stage_config["decoder_layers"]),
        attention_heads=int(stage_config["attention_heads"]),
        feedforward_dim=int(stage_config["feedforward_dim"]),
        dropout=float(stage_config["dropout"]),
        pretrained=bool(stage_config["pretrained"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(stage_config["weight_decay"]),
    )
    train_loader = slot_loader(
        splits["train"], input_vocab, component_vocab, int(data_config["image_size"]),
        int(stage_config["batch_size"]), int(data_config["num_workers"]), True, max_skeleton_length,
    )
    validation_loader = slot_loader(
        splits["validation"], input_vocab, component_vocab, int(data_config["image_size"]),
        int(stage_config["batch_size"]), int(data_config["num_workers"]), False, max_skeleton_length,
    )
    output_dir = Path(config["experiment"]["output_dir"]) / "stage2"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run = start_wandb(config, "stage2-oracle-filled-marker")
    best_loss = float("inf")
    model_config = dict(stage_config)
    model_config.update(
        {
            "image_size": int(data_config["image_size"]),
            "max_skeleton_length": max_skeleton_length,
            "pad_id": pad_id,
        }
    )
    vocabularies = {"input_vocab": input_vocab, "component_vocab": component_vocab}
    try:
        run.log(
            {
                **{
                    f"data/{split_name}_{unit}": count
                    for split_name, values in summary.items()
                    if split_name.startswith("split_")
                    for unit, count in values.items()
                },
                "vocab/input_tokens": len(input_vocab),
                "vocab/component_leaves": len(component_vocab),
            }
        )
        for epoch in range(1, int(stage_config["epochs"]) + 1):
            train_loss = _run_epoch(model, train_loader, optimizer, device)
            validation_loss = _run_epoch(model, validation_loader, None, device)
            metrics = {
                "epoch": epoch,
                "stage2/train_loss": train_loss,
                "stage2/validation_loss": validation_loss,
            }
            run.log(metrics, step=epoch)
            print(metrics)
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, epoch, model_config, vocabularies, config
                )
        run.log({"stage2/best_validation_loss": best_loss})
        import wandb

        artifact = wandb.Artifact("stage2-oracle-filled-marker", type="model")
        artifact.add_file(str(output_dir / "best.pt"))
        run.log_artifact(artifact)
        write_json(output_dir / "summary.json", {"git": git_metadata(), "data": summary})
    finally:
        run.finish()


if __name__ == "__main__":
    main()