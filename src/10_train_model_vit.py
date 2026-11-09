from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import pickle
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
from structured_codebook import STRUCTURE_SET, tokenize_ids  # noqa: E402  (src is on sys.path once 6_train_model is loaded)

IMAGE_SIZE = 224
SETTINGS = argparse.Namespace(
    lr=5e-5,
    head_lr=1e-3,
    weight_decay=0.05,
    patience=10,
    min_delta=None,
    aux_loss=True,
    aux_weight=0.5,
    augmentation=True,
    num_workers=4,
    logit_bound=5.0,
    warmup_steps=3000,
    lr_patience=5,
    monitor="val_h",
    ce_train_classes_only=True,
    kana_cap=300,
    val_max_samples=20000,
    init_from=None,
    disjoint_unseen=True,
    component_head=True,
    component_weight=1.0,
    component_min_count=1,
    component_unknown_logp=-9.0,
    codebook_path=None,
)

# Penalties subtracted from seen-class scores (calibrated stacking); candidates for the validation grid search.
GAMMA_GRID = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
# Weights of the component log-probability score added to the FaRE code score (0 = FaRE only).
LAMBDA_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
# (mode, default min_delta) per monitored validation metric.
MONITORS = {"val_h": ("max", 0.002), "val_unseen_zsl_acc": ("max", 0.002), "val_seen_loss": ("min", 0.01)}
# Shared between the training function and the patched test evaluation.
STATE: dict[str, Any] = {"seen_mask": None, "gamma": 0.0, "lambda": 0.0, "stage": "", "out_dir": None, "components": None}


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

    # Unseen classes are split by class (not by sample) so validation never reuses test-unseen classes.
    unseen_entries = [entry for entry in val_entries + test_entries if entry.get("split") in {"val_unseen", "test_unseen"}]
    if SETTINGS.disjoint_unseen and unseen_entries:
        ordered = sorted(
            {str(entry["unicode"]) for entry in unseen_entries},
            key=lambda name: original_stable_value(seed, "finetune-unseen-class-group", name),
        )
        validation_count = len(ordered) if len(ordered) == 1 else min(max(1, round(len(ordered) * validation_sample_ratio)), len(ordered) - 1)
        validation_classes = set(ordered[:validation_count])
        regrouped_val, regrouped_test = [], []
        for entry in unseen_entries:
            item = dict(entry)
            item["split"] = "val_unseen" if str(entry["unicode"]) in validation_classes else "test_unseen"
            (regrouped_val if item["split"] == "val_unseen" else regrouped_test).append(item)
        val_entries[:] = [entry for entry in val_entries if entry.get("split") == "val_seen"] + regrouped_val
        test_entries[:] = [entry for entry in test_entries if entry.get("split") == "test_seen"] + regrouped_test

    if SETTINGS.kana_cap > 0:
        kana_by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        kept_entries = []
        for entry in train_entries:
            if is_kana_label(str(entry["unicode"])):
                kana_by_class[str(entry["unicode"])].append(entry)
            else:
                kept_entries.append(entry)
        before = sum(len(items) for items in kana_by_class.values())
        for name, items in kana_by_class.items():
            items.sort(key=lambda entry: original_stable_value(seed, "kana-cap", name, str(entry["image_path"])))
            kept_entries.extend(items[: SETTINGS.kana_cap])
        train_entries = kept_entries
        after = sum(is_kana_label(str(entry["unicode"])) for entry in train_entries)
        print(
            f"Kana training samples capped at {SETTINGS.kana_cap}/class: {before} -> {after} "
            f"({after / max(1, len(train_entries)):.1%} of {len(train_entries)} training samples)."
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
        self.component_head: nn.Linear | None = None  # attached once the training-set vocabulary is known
        self.logit_bound = float(SETTINGS.logit_bound)

    def load_state_dict(self, state_dict: Any, strict: bool = True, **kwargs: Any) -> Any:
        weight = state_dict.get("component_head.weight")
        if weight is not None and self.component_head is None:
            self.component_head = nn.Linear(weight.shape[1], weight.shape[0]).to(self.projection.weight.device)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.backbone(x).float()

    def project_bits(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.projection(features)

    def project_components(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.component_head is None:
            return None
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.component_head(features)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_bits(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.forward_features(x)
        bits = base.STEBinarize.apply(self.bound(self.project_bits(features)))
        return bits, self.project_components(features)

    def bound(self, raw: torch.Tensor) -> torch.Tensor:
        # Sign is unchanged; the gradient shrinks as |raw| grows, which stops logit growth through the STE.
        return self.logit_bound * torch.tanh(raw / self.logit_bound)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.bound(self.forward_raw(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return base.STEBinarize.apply(self.forward_logits(x))


def make_model(codebook_dim: int = 400, **_: Any) -> nn.Module:
    model = ViTFareRecognitionModel(codebook_dim=codebook_dim)
    if SETTINGS.init_from:
        checkpoint = torch.load(SETTINGS.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Initialized model from {SETTINGS.init_from} (epoch {checkpoint.get('epoch')}).")
    return model


@torch.no_grad()
def evaluate_group(
    model: nn.Module,
    loader: DataLoader | None,
    matrix: torch.Tensor,
    structured: bool,
    device: torch.device,
    seen_mask: torch.Tensor,
    unseen_group: bool,
    pairs: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, Any]:
    """One pass over a group of samples: all-class (GZSL), closed-set, calibrated-stacking and component-fused accuracy."""
    model.eval()
    components = STATE["components"]
    lambdas = LAMBDA_GRID if components is not None else (0.0,)
    if pairs is None:
        pairs = tuple((lam, gam) for lam in lambdas for gam in GAMMA_GRID)
    totals: collections.Counter = collections.Counter()
    pair_correct: collections.Counter = collections.Counter()
    closed_correct: collections.Counter = collections.Counter()
    pool = ~seen_mask if unseen_group else seen_mask
    seen_float = seen_mask.float().unsqueeze(0)
    closed_lambdas = sorted({0.0, *(lam for lam, _ in pairs)})
    if loader is not None:
        for batch in loader:
            if batch is None:
                continue
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            bits, component_logits = model.forward_all(images)
            scores = base.codebook_logits(bits, matrix, structured)
            component_scores = None
            if components is not None and component_logits is not None:
                log_probs = F.log_softmax(component_logits.float(), dim=1)
                component_scores = log_probs @ components["score_matrix"] + components["unknown_score"]
                known = components["targets"].index_select(0, labels)
                has_known = known.sum(1) > 0
                hits = known.gather(1, log_probs.topk(5, dim=1).indices).sum(1) / known.sum(1).clamp_min(1)
                totals["comp_n"] += has_known.sum().item()
                totals["comp_recall5"] += (hits * has_known).sum().item()
            predictions = scores.argmax(1)
            correct = predictions == labels
            closed_scores = scores.masked_fill(~pool.unsqueeze(0), float("-inf"))
            closed_ce = closed_scores.logsumexp(1) - scores.gather(1, labels.unsqueeze(1)).squeeze(1)
            kana = torch.tensor([is_kana_label(str(name)) for name in batch["unicode"]], device=device)
            totals["n"] += labels.size(0)
            totals["ce_all"] += F.cross_entropy(scores, labels, reduction="sum").item()
            totals["ce_closed"] += closed_ce.clamp(max=100.0).sum().item()
            totals["correct"] += correct.sum().item()
            totals["into_seen"] += seen_mask[predictions].sum().item()
            totals["kana_n"] += kana.sum().item()
            totals["kana_correct"] += (correct & kana).sum().item()
            for lam in closed_lambdas:
                fused = scores if lam == 0.0 or component_scores is None else scores + lam * component_scores
                closed_correct[lam] += (fused.masked_fill(~pool.unsqueeze(0), float("-inf")).argmax(1) == labels).sum().item()
            for lam, gam in pairs:
                fused = scores if lam == 0.0 or component_scores is None else scores + lam * component_scores
                pair_correct[(lam, gam)] += ((fused - gam * seen_float).argmax(1) == labels).sum().item()
    n = totals["n"]
    kanji_n = n - totals["kana_n"]
    return {
        "n": n,
        "loss_all": totals["ce_all"] / max(1, n),
        "acc": totals["correct"] / max(1, n),
        "closed_loss": totals["ce_closed"] / max(1, n),
        "closed_acc": closed_correct[0.0] / max(1, n),
        "closed_acc_by_lambda": {lam: closed_correct[lam] / max(1, n) for lam in closed_lambdas},
        "into_seen": totals["into_seen"] / max(1, n),
        "kana_acc": totals["kana_correct"] / totals["kana_n"] if totals["kana_n"] else None,
        "kanji_acc": (totals["correct"] - totals["kana_correct"]) / kanji_n if kanji_n else None,
        "comp_recall5": totals["comp_recall5"] / totals["comp_n"] if totals["comp_n"] else None,
        "pair_acc": {pair: pair_correct[pair] / max(1, n) for pair in pairs},
    }


def build_component_assets(
    class_to_index: dict[str, int], train_label_ids: set[int], device: torch.device
) -> dict[str, Any] | None:
    """Component vocabulary from training classes, plus per-class target and scoring matrices."""
    with open(SETTINGS.codebook_path, "rb") as handle:
        entries = pickle.load(handle)["entries"]
    names = [""] * len(class_to_index)
    for name, index in class_to_index.items():
        names[index] = name
    leaves = [
        sorted({token for token in tokenize_ids(entries[name]["polish"]) if token not in STRUCTURE_SET}) if name in entries else []
        for name in names
    ]
    counts = collections.Counter(token for index in train_label_ids for token in leaves[index])
    vocabulary = sorted(token for token, count in counts.items() if count >= SETTINGS.component_min_count)
    if not vocabulary:
        return None
    position = {token: i for i, token in enumerate(vocabulary)}
    targets = torch.zeros(len(names), len(vocabulary))
    unknown = torch.zeros(len(names))
    sizes = torch.tensor([max(1, len(items)) for items in leaves], dtype=torch.float32)
    for index, items in enumerate(leaves):
        for token in items:
            if token in position:
                targets[index, position[token]] = 1.0
            else:
                unknown[index] += 1.0
    return {
        "vocabulary": vocabulary,
        "targets": targets.to(device),
        "score_matrix": (targets / sizes.unsqueeze(1)).t().contiguous().to(device),
        "unknown_score": (unknown * SETTINGS.component_unknown_logp / sizes).to(device),
        "unknown_count": int((unknown > 0).sum().item()),
    }


def harmonic_mean(first: float, second: float) -> float:
    return 2 * first * second / (first + second) if first + second > 0 else 0.0


def summarize_validation(seen: dict[str, Any], unseen: dict[str, Any]) -> dict[str, float]:
    best_h, best_pair = -1.0, (0.0, 0.0)
    best_fare_only = 0.0
    for pair, seen_acc in seen["pair_acc"].items():
        h = harmonic_mean(seen_acc, unseen["pair_acc"].get(pair, 0.0))
        if pair[0] == 0.0:
            best_fare_only = max(best_fare_only, h)
        if h > best_h:
            best_h, best_pair = h, pair
    best_lambda, best_gamma = best_pair
    nan = float("nan")
    return {
        "val_seen_loss": seen["loss_all"],
        "val_seen_acc": seen["acc"],
        "val_seen_kanji_acc": seen["kanji_acc"] if seen["kanji_acc"] is not None else nan,
        "val_unseen_loss": unseen["loss_all"],
        "val_unseen_zsl_loss": unseen["closed_loss"],
        "val_unseen_zsl_acc": unseen["closed_acc"],
        "val_unseen_zsl_acc_fused": unseen["closed_acc_by_lambda"].get(best_lambda, unseen["closed_acc"]),
        "val_unseen_gzsl_acc": unseen["acc"],
        "val_unseen_into_seen": unseen["into_seen"],
        "val_seen_comp_recall5": seen["comp_recall5"] if seen["comp_recall5"] is not None else nan,
        "val_unseen_comp_recall5": unseen["comp_recall5"] if unseen["comp_recall5"] is not None else nan,
        "cs_lambda": best_lambda,
        "cs_gamma": best_gamma,
        "cs_seen_acc": seen["pair_acc"][best_pair],
        "cs_unseen_acc": unseen["pair_acc"].get(best_pair, 0.0),
        "val_h_fare_only": best_fare_only,
        "val_h": max(best_h, 0.0),
    }


def evaluate_entries_extended(
    model: nn.Module,
    entries: list[dict[str, Any]],
    codebook: torch.Tensor,
    device: torch.device,
    batch_size: int,
    structured_codebook: bool,
) -> tuple[float, float]:
    """Replaces the base test evaluation: same (loss, accuracy) return, plus ZSL/calibrated metrics."""
    seen_mask = STATE["seen_mask"]
    split = str(entries[0].get("split", "")) if entries else ""
    unseen_group = split.endswith("unseen")
    loader = DataLoader(
        ViTCharacterDataset(entries),
        batch_size=batch_size,
        shuffle=False,
        num_workers=SETTINGS.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=base.safe_collate,
    )
    stats = evaluate_group(
        model, loader, codebook, structured_codebook, device, seen_mask, unseen_group,
        pairs=((0.0, 0.0), (STATE["lambda"], STATE["gamma"])),
    )
    gamma, lam = STATE["gamma"], STATE["lambda"]
    stage = STATE["stage"]
    kanji = "n/a" if stats["kanji_acc"] is None else f"{stats['kanji_acc']:.4f}"
    kana = "n/a" if stats["kana_acc"] is None else f"{stats['kana_acc']:.4f}"
    recall = "n/a" if stats["comp_recall5"] is None else f"{stats['comp_recall5']:.4f}"
    fused_closed = stats["closed_acc_by_lambda"].get(lam, stats["closed_acc"])
    print(
        f"[extended {stage}/{split}] n={stats['n']} gzsl_acc={stats['acc']:.4f} "
        f"{'zsl_acc' if unseen_group else 'seen_closed_acc'}={stats['closed_acc']:.4f} "
        f"{'zsl_acc' if unseen_group else 'seen_closed_acc'}_fused(lambda={lam:g})={fused_closed:.4f} "
        f"calibrated(lambda={lam:g},gamma={gamma:g})_acc={stats['pair_acc'][(lam, gamma)]:.4f} "
        f"predicted_into_seen={stats['into_seen']:.3f} component_recall5={recall} kanji_acc={kanji} kana_acc={kana}"
    )
    if STATE["out_dir"] is not None:
        path = Path(STATE["out_dir"]) / "extended_test_metrics.json"
        results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        results[f"{stage}/{split}"] = {
            **stats,
            "lambda": lam,
            "gamma": gamma,
            "closed_acc_by_lambda": {str(k): v for k, v in stats["closed_acc_by_lambda"].items()},
            "pair_acc": {f"lambda={k[0]:g},gamma={k[1]:g}": v for k, v in stats["pair_acc"].items()},
        }
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats["loss_all"], stats["acc"]


def subsample_entries(entries: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(entries) <= limit:
        return entries
    return random.Random(seed).sample(entries, limit)


def is_improvement(value: float, best: float, mode: str, delta: float) -> bool:
    return value > best + delta if mode == "max" else value < best - delta


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
    unseen_validation = [entry for entry in val_entries if entry.get("split") == "val_unseen"]
    if not seen_validation and not unseen_validation:
        seen_validation = val_entries
    seen_validation = subsample_entries(seen_validation, SETTINGS.val_max_samples, seed)
    unseen_validation = subsample_entries(unseen_validation, SETTINGS.val_max_samples, seed)
    val_seen_loader = DataLoader(ViTCharacterDataset(seen_validation), batch_size=batch_size, shuffle=False, **loader_options)
    val_unseen_loader = (
        DataLoader(ViTCharacterDataset(unseen_validation), batch_size=batch_size, shuffle=False, **loader_options)
        if unseen_validation
        else None
    )

    num_classes = codebook_matrix.shape[0]
    train_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    train_mask[sorted({int(entry["label"]) for entry in train_entries})] = True
    STATE.update(
        seen_mask=train_mask,
        gamma=0.0,
        **{"lambda": 0.0},
        stage="pretrain" if checkpoint_path.name.startswith("pretrain") else "finetune",
        out_dir=checkpoint_path.parent,
        components=None,
    )
    if SETTINGS.component_head:
        components = build_component_assets(class_to_index, {int(entry["label"]) for entry in train_entries}, device)
        if components is not None:
            STATE["components"] = components
            if model.component_head is None:
                model.component_head = nn.Linear(768, len(components["vocabulary"])).to(device)
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_history.jsonl")
    monitor_mode, default_delta = MONITORS[SETTINGS.monitor]
    min_delta = default_delta if SETTINGS.min_delta is None else SETTINGS.min_delta
    if verbose:
        vocabulary_text = (
            f"component head: {len(STATE['components']['vocabulary'])} components, "
            f"{STATE['components']['unknown_count']} classes contain a component outside the vocabulary"
            if STATE["components"] is not None
            else "component head: off"
        )
        print(
            f"Training classes: {int(train_mask.sum())}/{num_classes} (CE denominators "
            f"{'limited to training classes' if SETTINGS.ce_train_classes_only else 'cover all classes'}); "
            f"validation seen={len(seen_validation)} unseen={len(unseen_validation)}; "
            f"monitor={SETTINGS.monitor} ({monitor_mode}, min_delta={min_delta}); {vocabulary_text}."
        )

    backbone_params = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": SETTINGS.lr, "target_lr": SETTINGS.lr},
            {"params": head_params, "lr": SETTINGS.head_lr, "target_lr": SETTINGS.head_lr},
        ],
        weight_decay=SETTINGS.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=monitor_mode, factor=0.5, patience=SETTINGS.lr_patience, threshold=min_delta,
        threshold_mode="abs", cooldown=1, min_lr=[1e-6, 1e-5],
    )
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = min(SETTINGS.warmup_steps, steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    best_gamma = 0.0
    best_lambda = 0.0
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
            if state.get("monitor") == SETTINGS.monitor:
                best_metric = float(state.get("best_metric", best_metric))
                best_gamma = float(state.get("best_gamma", best_gamma))
                best_lambda = float(state.get("best_lambda", best_lambda))
                epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            best_accuracy = float(state.get("best_accuracy", best_accuracy))
            best_epoch = int(state.get("best_epoch", best_epoch))
            start_epoch = int(state.get("epoch", 0)) + 1
            if verbose:
                print(f"Resumed training from {state_path} at epoch {start_epoch}")
    for group, target in zip(optimizer.param_groups, (SETTINGS.lr, SETTINGS.head_lr)):
        group.setdefault("target_lr", target)

    if start_epoch > epochs:
        if checkpoint_path.exists():
            best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_state_dict = best_state.get("state_dict") if isinstance(best_state, dict) else None
            if isinstance(best_state, dict):
                best_gamma = float(best_state.get("gamma", best_gamma))
                best_lambda = float(best_state.get("lambda", best_lambda))
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        STATE["gamma"] = best_gamma
        STATE["lambda"] = best_lambda
        return best_accuracy, best_epoch

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals = {"total": 0.0, "cls": 0.0, "aux": 0.0, "comp": 0.0, "abs_logit": 0.0, "bit_error": 0.0}
        seen = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch}", leave=False)):
            if batch is None:
                continue
            global_step = (epoch - 1) * steps_per_epoch + batch_index
            if global_step < warmup_steps:
                warmup_factor = (global_step + 1) / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = group["target_lr"] * warmup_factor

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            features = model.forward_features(images)
            raw_logits = model.project_bits(features)
            bounded_logits = model.bound(raw_logits)
            encoded = base.STEBinarize.apply(bounded_logits)
            class_scores = base.codebook_logits(encoded, codebook_matrix, structured_codebook)
            if SETTINGS.ce_train_classes_only:
                class_scores = class_scores.masked_fill(~train_mask.unsqueeze(0), float("-inf"))
            classification_loss = criterion(class_scores, labels)
            targets = codebook_matrix.index_select(0, labels)
            valid_bits = targets.ne(0)
            loss = classification_loss
            if SETTINGS.aux_loss:
                binary_targets = targets.gt(0).to(bounded_logits.dtype)
                bit_loss = F.binary_cross_entropy_with_logits(bounded_logits, binary_targets, reduction="none")
                per_sample_aux = (bit_loss * valid_bits).sum(dim=1) / valid_bits.sum(dim=1).clamp_min(1)
                auxiliary_loss = per_sample_aux.mean()
                loss = loss + SETTINGS.aux_weight * auxiliary_loss
            else:
                auxiliary_loss = bounded_logits.new_zeros(())
            component_loss = bounded_logits.new_zeros(())
            component_logits = model.project_components(features)
            if component_logits is not None:
                component_targets = STATE["components"]["targets"].index_select(0, labels)
                positives = component_targets.sum(dim=1)
                has_positive = positives > 0
                per_sample = -(F.log_softmax(component_logits, dim=1) * component_targets).sum(dim=1) / positives.clamp_min(1)
                component_loss = (per_sample * has_positive).sum() / has_positive.sum().clamp_min(1)
                loss = loss + SETTINGS.component_weight * component_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                wrong_bits = ((raw_logits > 0) != targets.gt(0)) & valid_bits
                bit_error = (wrong_bits.sum(dim=1) / valid_bits.sum(dim=1).clamp_min(1)).mean()
            batch_size_now = images.size(0)
            totals["total"] += loss.item() * batch_size_now
            totals["cls"] += classification_loss.item() * batch_size_now
            totals["aux"] += auxiliary_loss.item() * batch_size_now
            totals["comp"] += component_loss.item() * batch_size_now
            totals["abs_logit"] += raw_logits.detach().abs().mean().item() * batch_size_now
            totals["bit_error"] += bit_error.item() * batch_size_now
            seen += batch_size_now

        averages = {key: value / max(1, seen) for key, value in totals.items()}
        train_loss = averages["total"]
        seen_stats = evaluate_group(model, val_seen_loader, codebook_matrix, structured_codebook, device, train_mask, False)
        unseen_stats = evaluate_group(model, val_unseen_loader, codebook_matrix, structured_codebook, device, train_mask, True)
        metrics = summarize_validation(seen_stats, unseen_stats)
        monitored = metrics[SETTINGS.monitor]
        scheduler.step(monitored)
        learning_rates = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        if verbose:
            aug_label = "on" if SETTINGS.augmentation else "off"
            aux_label = "on" if SETTINGS.aux_loss else "off"
            aux_text = f"train_aux_loss={averages['aux']:.4f} | " if SETTINGS.aux_loss else ""
            comp_text = (
                f"train_comp_loss={averages['comp']:.4f} | comp_recall5 seen={metrics['val_seen_comp_recall5']:.3f} "
                f"unseen={metrics['val_unseen_comp_recall5']:.3f} | "
                if STATE["components"] is not None
                else ""
            )
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | "
                f"train_cls_loss={averages['cls']:.4f} | {aux_text}{comp_text}"
                f"mean_abs_raw_logit={averages['abs_logit']:.1f} | bit_error={averages['bit_error']:.3f} | "
                f"val_seen_loss={metrics['val_seen_loss']:.4f} | val_seen_acc={metrics['val_seen_acc']:.4f} | "
                f"val_unseen_loss={metrics['val_unseen_loss']:.4f} | val_unseen_zsl_acc={metrics['val_unseen_zsl_acc']:.4f} | "
                f"val_unseen_zsl_acc_fused={metrics['val_unseen_zsl_acc_fused']:.4f} | "
                f"val_unseen_gzsl_acc={metrics['val_unseen_gzsl_acc']:.4f} | into_seen={metrics['val_unseen_into_seen']:.3f} | "
                f"cs_lambda={metrics['cs_lambda']:g} | cs_gamma={metrics['cs_gamma']:g} | cs_seen={metrics['cs_seen_acc']:.4f} | "
                f"cs_unseen={metrics['cs_unseen_acc']:.4f} | val_h_fare_only={metrics['val_h_fare_only']:.4f} | "
                f"val_h={metrics['val_h']:.4f} | lr={learning_rates} | aug={aug_label} | aux={aux_label}"
            )
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, "stage": STATE["stage"], **{f"train_{k}": v for k, v in averages.items()}, **metrics, "lr": learning_rates}) + "\n")

        if is_improvement(monitored, best_metric, monitor_mode, min_delta):
            best_metric = monitored
            best_gamma = metrics["cs_gamma"]
            best_lambda = metrics["cs_lambda"]
            best_accuracy = metrics["val_h"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "monitor": SETTINGS.monitor,
                    "monitor_value": monitored,
                    "gamma": best_gamma,
                    "lambda": best_lambda,
                    "component_vocabulary": STATE["components"]["vocabulary"] if STATE["components"] is not None else None,
                    "val_metrics": metrics,
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
                "best_metric": best_metric,
                "best_gamma": best_gamma,
                "best_lambda": best_lambda,
                "monitor": SETTINGS.monitor,
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
            print(f"Early stopping: {SETTINGS.monitor} did not improve for {SETTINGS.patience} epochs.")
            break

    if best_state_dict is None:
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_state_dict = saved.get("state_dict") if isinstance(saved, dict) else None
            if isinstance(saved, dict):
                best_gamma = float(saved.get("gamma", best_gamma))
                best_lambda = float(saved.get("lambda", best_lambda))
    if best_state_dict is None:
        raise RuntimeError("No best ViT checkpoint was saved.")
    model.load_state_dict(best_state_dict)
    STATE["gamma"] = best_gamma
    STATE["lambda"] = best_lambda
    print(
        f"Restored best epoch {best_epoch} ({SETTINGS.monitor}={best_metric:.4f}); "
        f"calibrated-stacking gamma={best_gamma:g}, component fusion lambda={best_lambda:g}."
    )
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
    parser.add_argument("--vit-logit-bound", type=float, default=SETTINGS.logit_bound)
    parser.add_argument("--vit-warmup-steps", type=int, default=SETTINGS.warmup_steps)
    parser.add_argument("--lr-scheduler-patience", type=int, default=SETTINGS.lr_patience)
    parser.add_argument("--monitor", choices=sorted(MONITORS), default=SETTINGS.monitor, help="Validation metric for model selection, LR decay and early stopping.")
    parser.add_argument("--ce-train-classes-only", action=argparse.BooleanOptionalAction, default=SETTINGS.ce_train_classes_only, help="Restrict the training CE denominator to classes present in the training set.")
    parser.add_argument("--kana-max-samples-per-class", type=int, default=SETTINGS.kana_cap, help="Cap on kana training samples per class (0 disables the cap).")
    parser.add_argument("--val-max-samples", type=int, default=SETTINGS.val_max_samples, help="Max validation samples per group (seen/unseen) per epoch; 0 uses all.")
    parser.add_argument("--init-from", default=None, help="Initialize from a checkpoint (e.g. pretrain_best_fare_model.pth); skips pretraining.")
    parser.add_argument("--disjoint-unseen", action=argparse.BooleanOptionalAction, default=SETTINGS.disjoint_unseen, help="Split unseen classes (not samples) between validation and test.")
    parser.add_argument("--component-head", action=argparse.BooleanOptionalAction, default=SETTINGS.component_head, help="Add the auxiliary component (bag-of-components) classification head and fuse it at evaluation.")
    parser.add_argument("--component-loss-weight", type=float, default=SETTINGS.component_weight)
    parser.add_argument("--component-min-count", type=int, default=SETTINGS.component_min_count, help="Minimum number of training classes a component must appear in to enter the vocabulary.")
    parser.add_argument("--component-unknown-logp", type=float, default=SETTINGS.component_unknown_logp, help="Log-probability assigned to candidate components outside the vocabulary.")
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
    SETTINGS.logit_bound = options.vit_logit_bound
    SETTINGS.warmup_steps = options.vit_warmup_steps
    SETTINGS.lr_patience = options.lr_scheduler_patience
    SETTINGS.monitor = options.monitor
    SETTINGS.ce_train_classes_only = options.ce_train_classes_only
    SETTINGS.kana_cap = options.kana_max_samples_per_class
    SETTINGS.val_max_samples = options.val_max_samples
    SETTINGS.init_from = options.init_from
    SETTINGS.disjoint_unseen = options.disjoint_unseen
    SETTINGS.component_head = options.component_head
    SETTINGS.component_weight = options.component_loss_weight
    SETTINGS.component_min_count = options.component_min_count
    SETTINGS.component_unknown_logp = options.component_unknown_logp
    sys.argv = [sys.argv[0], *remaining]

    original_parse_args = base.parse_args
    original_save_json = base.save_json

    def parse_args_with_early_stopping() -> argparse.Namespace:
        args = original_parse_args()
        SETTINGS.codebook_path = args.codebook
        if SETTINGS.init_from and args.pretrain_root:
            raise SystemExit("--init-from skips pretraining; do not pass --pretrain-root together with it.")
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
            "kana_max_samples_per_class": SETTINGS.kana_cap,
            "ce_train_classes_only": SETTINGS.ce_train_classes_only,
            "disjoint_unseen_val_test_classes": SETTINGS.disjoint_unseen,
            "init_from": SETTINGS.init_from,
            "validation_max_samples_per_group": SETTINGS.val_max_samples,
            "calibrated_stacking_gamma_grid": list(GAMMA_GRID),
            "component_head_enabled": SETTINGS.component_head,
            "component_loss_weight": SETTINGS.component_weight,
            "component_min_count": SETTINGS.component_min_count,
            "component_unknown_logp": SETTINGS.component_unknown_logp,
            "component_fusion_lambda_grid": list(LAMBDA_GRID),
            "component_vocabulary_size": len(STATE["components"]["vocabulary"]) if STATE["components"] is not None else 0,
            "auxiliary_bce_enabled": SETTINGS.aux_loss,
            "auxiliary_bce_weight": SETTINGS.aux_weight,
            "augmentation_enabled": SETTINGS.augmentation,
            "backbone_learning_rate": SETTINGS.lr,
            "projection_learning_rate": SETTINGS.head_lr,
            "weight_decay": SETTINGS.weight_decay,
            "early_stopping_monitor": SETTINGS.monitor,
            "early_stopping_patience": SETTINGS.patience,
            "early_stopping_min_delta": SETTINGS.min_delta,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_patience": SETTINGS.lr_patience,
            "logit_bound": SETTINGS.logit_bound,
            "warmup_steps": SETTINGS.warmup_steps,
        }
        original_save_json(path, metadata)

    base.parse_args = parse_args_with_early_stopping
    base.save_json = save_json_with_vit_settings
    base.FareRecognitionModel = make_model
    base.CharacterImageDataset = ViTCharacterDataset
    base.train_model_on_dataset = train_vit_on_dataset
    base.evaluate_entries = evaluate_entries_extended
    torch.set_float32_matmul_precision("high")
    base.main()


if __name__ == "__main__":
    main()
