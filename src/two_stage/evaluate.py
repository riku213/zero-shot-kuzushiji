from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.two_stage.data.pipeline import ImageRecordDataset, load_eligible_leaves
from src.two_stage.experiment import (
    DEFAULT_CONFIG,
    git_metadata,
    load_config,
    prepare_data,
    set_seed,
    start_wandb,
    write_json,
)
from src.two_stage.models.stage1_classifier import Stage1SkeletonClassifier
from src.two_stage.models.stage2_filler import Stage2SlotFiller
from src.two_stage.pipeline.reconstruct import SLOT, reconstruct


DEPTH_BUCKETS = {
    "depth0": lambda depth: depth == 0,
    "depth1": lambda depth: depth == 1,
    "depth2plus": lambda depth: depth >= 2,
}
STAGE2_METRIC_KEYS = (
    "stage2.slot_accuracy",
    "stage2.slot_exact_match",
    "char.accuracy_oracle",
    "dict.hit_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle skeleton and slot-filling checkpoints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _metric_pack(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    slots = sum(row["slot_count"] for row in rows)
    count = len(rows)
    return {
        "samples": count,
        "stage2.slot_accuracy": 100.0 * sum(row["correct_slots"] for row in rows) / max(1, slots),
        "stage2.slot_exact_match": 100.0 * sum(row["slot_exact"] for row in rows) / max(1, count),
        "char.accuracy_oracle": 100.0 * sum(row["character_correct"] for row in rows) / max(1, count),
        "dict.hit_rate": 100.0 * sum(row["dictionary_hit"] for row in rows) / max(1, count),
    }


def _predict_components(
    model: Stage2SlotFiller,
    image: torch.Tensor,
    skeleton: list[str],
    input_vocab: dict[str, int],
    component_names: list[str],
    device: torch.device,
) -> list[str]:
    state = list(skeleton)
    slot_positions = [index for index, token in enumerate(state) if token == SLOT]
    predictions: list[str] = []
    for position in slot_positions:
        state_ids = torch.tensor(
            [[input_vocab[token] for token in state]], dtype=torch.long, device=device
        )
        logits = model(image.unsqueeze(0).to(device), state_ids)
        component_id = int(logits[0, position].argmax(dim=-1))
        predictions.append(component_names[component_id])
        state[position] = "<FILLED>"
    return predictions


@torch.no_grad()
def evaluate_oracle(
    model: Stage2SlotFiller,
    records: list[dict[str, Any]],
    reverse_index,
    input_vocab: dict[str, int],
    component_vocab: dict[str, int],
    image_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    component_names = [""] * len(component_vocab)
    for name, index in component_vocab.items():
        component_names[index] = name
    loader = DataLoader(ImageRecordDataset(records, image_size), batch_size=1, shuffle=False)
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in tqdm(loader, desc="oracle evaluation", leave=False):
        skeleton = [tokens[0] for tokens in batch["skeleton"]]
        components = [tokens[0] for tokens in batch["components"]]
        predicted = _predict_components(
            model, batch["image"][0], skeleton, input_vocab, component_names, device
        )
        reconstructed = reconstruct(skeleton, predicted, reverse_index)
        candidate = reconstructed.character
        character = batch["char"][0]
        character_correct = character in candidate if isinstance(candidate, list) else candidate == character
        rows.append(
            {
                "char": character,
                "depth": int(batch["depth"][0]),
                "slot_count": len(components),
                "correct_slots": sum(left == right for left, right in zip(predicted, components)),
                "slot_exact": predicted == components,
                "character_correct": character_correct,
                "dictionary_hit": reconstructed.dictionary_hit,
                "image_path": batch["image_path"][0],
                "ids": reconstructed.ids,
                "predicted_components": predicted,
                "target_components": components,
            }
        )
    overall = _metric_pack(rows)
    by_depth = {
        bucket: _metric_pack([row for row in rows if predicate(row["depth"])])
        for bucket, predicate in DEPTH_BUCKETS.items()
    }
    return {"overall": overall, "by_depth": by_depth}, rows


def load_stage2_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    vocabularies = checkpoint["vocabularies"]
    model = Stage2SlotFiller(
        input_vocabulary_size=len(vocabularies["input_vocab"]),
        component_vocabulary_size=len(vocabularies["component_vocab"]),
        max_skeleton_length=int(model_config["max_skeleton_length"]),
        pad_id=int(model_config["pad_id"]),
        image_size=int(
            model_config.get("image_size", checkpoint["experiment_config"]["data"]["image_size"])
        ),
        patch_size=int(model_config["patch_size"]),
        d_model=int(model_config["d_model"]),
        encoder_layers=int(model_config["encoder_layers"]),
        decoder_layers=int(model_config["decoder_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        feedforward_dim=int(model_config["feedforward_dim"]),
        dropout=float(model_config["dropout"]),
        pretrained=bool(model_config["pretrained"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), vocabularies


@torch.no_grad()
def evaluate_stage1_checkpoint(path: Path, records: list[dict[str, Any]], image_size: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocabularies = checkpoint["vocabularies"]
    model_config = checkpoint["model_config"]
    model = Stage1SkeletonClassifier(
        len(vocabularies["skeleton_to_id"]),
        tuple(model_config["encoder_channels"]),
        float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    id_to_skeleton = vocabularies["id_to_skeleton"]
    skeleton_to_id = vocabularies["skeleton_to_id"]
    counts = {bucket: {"correct": 0, "total": 0} for bucket in DEPTH_BUCKETS}
    total_correct = 0
    unseen_skeletons = 0
    loader = DataLoader(ImageRecordDataset(records, image_size), batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc="stage1 skeleton evaluation", leave=False):
        target = [tokens[0] for tokens in batch["skeleton"]]
        key = json.dumps(target, ensure_ascii=False)
        logits = model(batch["image"].to(device))
        predicted = id_to_skeleton[int(logits.argmax(dim=1))]
        correct = predicted == target
        total_correct += int(correct)
        unseen_skeletons += int(key not in skeleton_to_id)
        depth = int(batch["depth"][0])
        for bucket, predicate in DEPTH_BUCKETS.items():
            if predicate(depth):
                counts[bucket]["total"] += 1
                counts[bucket]["correct"] += int(correct)
                break
    total = len(records)
    return {
        "stage1.skeleton_accuracy": 100.0 * total_correct / max(1, total),
        "samples": total,
        "unseen_skeleton_samples": unseen_skeletons,
        "by_depth": {
            bucket: {
                "samples": values["total"],
                "stage1.skeleton_accuracy": 100.0 * values["correct"] / max(1, values["total"]),
            }
            for bucket, values in counts.items()
        },
    }


def _flatten_labnote_metrics(metrics: dict[str, Any], split_name: str = "random") -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    if "stage1" in metrics:
        flat.append(
            {
                "key": "stage1.skeleton_accuracy",
                "value": round(float(metrics["stage1"]["stage1.skeleton_accuracy"]), 2),
                "unit": "%",
            }
        )
        for depth_name, packed in metrics["stage1"]["by_depth"].items():
            flat.append(
                {
                    "key": f"{split_name}.{depth_name}.stage1.skeleton_accuracy",
                    "value": round(float(packed["stage1.skeleton_accuracy"]), 2),
                    "unit": "%",
                }
            )
    for key in STAGE2_METRIC_KEYS:
        flat.append({"key": key, "value": round(float(metrics["overall"][key]), 2), "unit": "%"})
    for depth_name, packed in metrics["by_depth"].items():
        for key in STAGE2_METRIC_KEYS:
            flat.append(
                {
                    "key": f"{split_name}.{depth_name}.{key}",
                    "value": round(float(packed[key]), 2),
                    "unit": "%",
                }
            )
    keys = [item["key"] for item in flat]
    if len(keys) != len(set(keys)):
        raise ValueError("labnote metrics contains duplicate keys.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/")
    if any(not key or set(key) - allowed for key in keys):
        raise ValueError("labnote metrics contains an invalid key.")
    return flat


def build_labnote_payload(
    config: dict[str, Any],
    metrics: dict[str, Any],
    experiment_id: str,
    checkpoint_path: Path,
    data_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experiment_config = config["experiment"]
    stage_config = config["stage2"]
    payload = {
        "experiment": {
            "code": experiment_id,
            "title": experiment_config.get("title", "Two-stage baseline"),
            "date": date.today().isoformat(),
            "git_commit": git_metadata().get("commit"),
            "checkpoint": str(checkpoint_path),
            "tags": ["two-stage", "oracle", "baseline"],
        },
        "conditions": {
            "split": experiment_config["split"],
            "stage2_input_mode": stage_config["input_mode"],
            "stage2_encoder": stage_config["encoder"],
            "stage2_pretrained": stage_config["pretrained"],
            "glyph_code_mode": stage_config["glyph_code_mode"],
            "guidance_loss_weight": stage_config["guidance_loss_weight"],
            "triplet_loss_weight": stage_config["triplet_loss_weight"],
            "train_classes": (data_summary or {}).get("split_classes", {}).get("train"),
            "test_classes": (data_summary or {}).get("split_classes", {}).get("test"),
            "validation_classes": (data_summary or {}).get("split_classes", {}).get("validation"),
            "train_samples": (data_summary or {}).get("split_samples", {}).get("train"),
            "validation_samples": (data_summary or {}).get("split_samples", {}).get("validation"),
            "test_samples": metrics["overall"]["samples"],
            "oracle_skeleton": True,
        },
        "params": {
            "stage1_epochs": config["stage1"]["epochs"],
            "stage1_learning_rate": config["stage1"]["learning_rate"],
            "stage1_early_stopping_patience": config["stage1"].get(
                "early_stopping_patience", config["stage1"]["epochs"]
            ),
            "stage2_epochs": stage_config["epochs"],
            "stage2_learning_rate": stage_config["learning_rate"],
            "stage2_early_stopping_patience": stage_config.get(
                "early_stopping_patience", stage_config["epochs"]
            ),
            "d_model": stage_config["d_model"],
            "encoder_layers": stage_config["encoder_layers"],
            "decoder_layers": stage_config["decoder_layers"],
            "patch_size": stage_config["patch_size"],
        },
        "metrics": _flatten_labnote_metrics(metrics, str(experiment_config["split"])),
    }
    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    round_trip = yaml.safe_load(serialized)
    if not isinstance(round_trip, dict) or not round_trip.get("experiment", {}).get("code"):
        raise ValueError("labnote result YAML failed self-validation.")
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    set_seed(int(config["experiment"]["seed"]))
    splits, reverse_index, data_summary = prepare_data(config)
    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    checkpoint_path = args.stage2_checkpoint or output_dir / "stage2" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {checkpoint_path}")
    device = torch.device(args.device)
    model, vocabularies = load_stage2_checkpoint(checkpoint_path, device)
    metrics, rows = evaluate_oracle(
        model,
        splits["test"],
        reverse_index,
        vocabularies["input_vocab"],
        vocabularies["component_vocab"],
        int(config["data"]["image_size"]),
        device,
    )
    if not splits["test"]:
        raise ValueError("Evaluation test split is empty.")
    stage1_path = args.stage1_checkpoint or output_dir / "stage1" / "best.pt"
    if stage1_path.is_file():
        metrics["stage1"] = evaluate_stage1_checkpoint(
            stage1_path, splits["test"], int(config["data"]["image_size"]), device
        )
    experiment_id = args.experiment_id or config["tracking"].get("result_id") or f"two-stage-{date.today():%y%m%d}"
    yaml_path = args.output or config["tracking"].get("result_yaml")
    yaml_path = Path(yaml_path) if yaml_path else output_dir / f"result_{experiment_id}.yaml"
    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).resolve().parents[2] / yaml_path
    payload = build_labnote_payload(config, metrics, experiment_id, checkpoint_path, data_summary)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output_dir / "evaluation.json", {"metrics": metrics, "data": data_summary, "rows": rows})
    write_json(output_dir / "summary.json", {"git": git_metadata(), "data": data_summary, "metrics": metrics})

    run = start_wandb(config, "two-stage-oracle-evaluation")
    try:
        split_name = str(config["experiment"]["split"])
        flattened = {
            f"{split_name}/oracle/{key}": value
            for key, value in metrics["overall"].items()
            if isinstance(value, (int, float))
        }
        for depth_name, packed in metrics["by_depth"].items():
            flattened.update(
                {
                    f"{split_name}/oracle/{depth_name}/{key}": value
                    for key, value in packed.items()
                    if isinstance(value, (int, float))
                }
            )
        if "stage1" in metrics:
            flattened[f"{split_name}/stage1/skeleton_accuracy"] = metrics["stage1"]["stage1.skeleton_accuracy"]
            for depth_name, packed in metrics["stage1"]["by_depth"].items():
                flattened[f"{split_name}/stage1/{depth_name}/skeleton_accuracy"] = packed["stage1.skeleton_accuracy"]
        run.log(flattened)
        import wandb

        table = wandb.Table(
            columns=["image", "char", "ids", "predicted_components", "target_components"],
            data=[
                [
                    wandb.Image(row["image_path"], caption=row["char"]),
                    row["char"],
                    row["ids"],
                    str(row["predicted_components"]),
                    str(row["target_components"]),
                ]
                for row in rows[:20]
            ],
        )
        run.log({"examples/oracle_predictions": table})
        artifact = wandb.Artifact(f"{experiment_id}-oracle-evaluation", type="evaluation")
        artifact.add_file(str(yaml_path))
        artifact.add_file(str(output_dir / "evaluation.json"))
        run.log_artifact(artifact)
    finally:
        run.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"labnote result YAML: {yaml_path}")


if __name__ == "__main__":
    main()
