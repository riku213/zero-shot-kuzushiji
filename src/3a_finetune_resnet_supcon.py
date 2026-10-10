from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm

from fare_pipeline import FontSet, _draw_radical, load_rendered_radicals

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RadicalViewDataset(Dataset):
    """Renders each radical with every font that covers it and yields augmented views.

    mode="train": (view1, view2, label)   mode="eval": (clean, view, label)
    """

    def __init__(
        self,
        radicals: list[str],
        fonts: FontSet,
        image_size: int = 96,
        strength: float = 1.0,
        mode: str = "train",
        repeats: int = 1,
        show_progress: bool = True,
    ) -> None:
        self.image_size = image_size
        self.mode = mode
        self.repeats = repeats
        self.radicals: list[str] = []
        self.bases: list[list[Image.Image]] = []

        for radical in tqdm(radicals, desc="render", disable=not show_progress):
            images = []
            for path, cmap in zip(fonts.paths, fonts.cmaps):
                if not all(ord(char) in cmap for char in radical):
                    continue
                try:
                    canvas = _draw_radical(radical, path, image_size)
                except Exception:
                    continue
                if (np.asarray(canvas) < 128).any():
                    images.append(canvas)
            if images:
                self.radicals.append(radical)
                self.bases.append(images)

        s = strength
        self.affine = T.RandomAffine(
            degrees=10 * s,
            translate=(0.06 * s, 0.06 * s),
            scale=(1 - 0.2 * s, 1 + 0.1 * s),
            shear=8 * s,
            fill=255,
        )
        self.elastic = T.ElasticTransform(alpha=25.0 * s, sigma=5.0, fill=255)
        self.erase = T.RandomErasing(p=0.25 * min(s, 1.0), scale=(0.02, 0.08), value=0)
        self.strength = s

    def __len__(self) -> int:
        return len(self.radicals) * self.repeats

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        return self._normalize(array)

    @staticmethod
    def _normalize(gray: torch.Tensor) -> torch.Tensor:
        tensor = gray.unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    def _augment(self, image: Image.Image) -> torch.Tensor:
        if self.strength <= 0:
            return self._to_tensor(image)
        if random.random() < 0.5:
            image = image.filter(ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3))
        image = self.affine(image)
        if random.random() < 0.4:
            image = self.elastic(image)
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        gray = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        if random.random() < 0.5:
            gray = (gray + torch.randn_like(gray) * 0.04 * self.strength).clamp(0, 1)
        if random.random() < 0.5:
            gray = 1.0 - gray
        return self.erase(self._normalize(gray))

    def __getitem__(self, index: int):
        index = index % len(self.radicals)
        base = random.choice(self.bases[index])
        if self.mode == "eval":
            return self._to_tensor(self.bases[index][0]), self._augment(base), index
        return self._augment(base), self._augment(base), index


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """z: (N, D) L2-normalized features of all views; labels: (N,)."""
    z = z.float()
    n = z.shape[0]
    logits = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positives = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    pos_count = positives.sum(1)
    valid = pos_count > 0
    if not valid.any():
        return z.sum() * 0.0
    mean_log_prob_pos = (log_prob * positives).sum(1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola uniformity on the unit hypersphere."""
    z = z.float()
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()


def build_backbone() -> nn.Module:
    try:
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    except AttributeError:
        backbone = models.resnet34(pretrained=True)
    return nn.Sequential(*list(backbone.children())[:-1])


def build_head(in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))


def set_stage(backbone: nn.Module, full: bool) -> None:
    # Sequential children: 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4-7 layer1-4, 8 avgpool.
    for index, child in enumerate(backbone.children()):
        trainable = full or index == 7
        for param in child.parameters():
            param.requires_grad = trainable


def set_train_mode(backbone: nn.Module, head: nn.Module, full: bool, freeze_bn: bool) -> None:
    backbone.train()
    head.train()
    for index, child in enumerate(backbone.children()):
        if not (full or index == 7):
            child.eval()
    if freeze_bn:
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def feature_stats(clean: torch.Tensor, aug: torch.Tensor) -> dict[str, float]:
    clean_n = F.normalize(clean.float(), dim=1)
    aug_n = F.normalize(aug.float(), dim=1)
    n = clean_n.shape[0]
    sims = clean_n @ clean_n.T
    off_diag = (sims.sum() - sims.diag().sum()) / max(n * (n - 1), 1)

    centered = clean.float() - clean.float().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    p = singular / singular.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())

    cross = aug_n @ clean_n.T
    return {
        "offdiag_cos": float(off_diag),
        "view_cos": float((clean_n * aug_n).sum(1).mean()),
        "top1": float((cross.argmax(1) == torch.arange(n)).float().mean()),
        "eff_rank": float(effective_rank),
    }


@torch.no_grad()
def evaluate(backbone: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    backbone.eval()
    clean, aug = [], []
    for clean_batch, aug_batch, _ in loader:
        clean.append(backbone(clean_batch.to(device)).flatten(1).float().cpu())
        aug.append(backbone(aug_batch.to(device)).flatten(1).float().cpu())
    return feature_stats(torch.cat(clean), torch.cat(aug))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet34 on radical images with SupCon + uniformity.")
    parser.add_argument("--manifest", default="outputs/radical_render_manifest.pkl")
    parser.add_argument("--fonts", nargs="+", default=["dataset/fonts/HanaMinA.ttf", "dataset/fonts/HanaMinB.ttf"])
    parser.add_argument("--output-dir", default="outputs/finetune_supcon")
    parser.add_argument("--state-path", default=None, help="Resume state; defaults to <output-dir>/training_state.pth.")
    parser.add_argument("--log-path", default=None, help="Log file; defaults to <output-dir>/train.log.")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=4, help="Views drawn per radical per epoch.")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Epochs training only layer4 and head.")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--uniformity-weight", type=float, default=0.5)
    parser.add_argument("--uniformity-t", type=float, default=2.0)
    parser.add_argument("--augment-strength", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path else output_dir / "training_state.pth"

    log_path = Path(args.log_path) if args.log_path else output_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        tqdm.write(message)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    radicals = sorted({r.radical for r in load_rendered_radicals(Path(args.manifest)) if r.renderable})
    random.Random(args.seed).shuffle(radicals)
    val_count = max(2, int(len(radicals) * args.val_ratio))
    val_radicals, train_radicals = radicals[:val_count], radicals[val_count:]

    fonts = FontSet([Path(p) for p in args.fonts])
    train_set = RadicalViewDataset(
        train_radicals, fonts, args.image_size, args.augment_strength, "train", args.repeats
    )
    val_set = RadicalViewDataset(val_radicals, fonts, args.image_size, args.augment_strength, "eval")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    log(f"train radicals={len(train_set.radicals)} val radicals={len(val_set.radicals)}")

    backbone, head = build_backbone().to(device), build_head().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": args.lr_backbone},
            {"params": head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    start_epoch, best_score = 0, None
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_score = state["epoch"], state["best_score"]
        log(f"resumed from epoch {start_epoch}")
    else:
        log(f"baseline (ImageNet) val: {evaluate(backbone, val_loader, device)}")

    for epoch in range(start_epoch, args.epochs):
        full = epoch >= args.freeze_epochs
        set_stage(backbone, full)
        set_train_mode(backbone, head, full, args.freeze_bn)

        running = {"loss": 0.0, "supcon": 0.0, "unif": 0.0}
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for view1, view2, labels in progress:
            images = torch.cat([view1, view2]).to(device, non_blocking=True)
            labels = torch.cat([labels, labels]).to(device)
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                z = F.normalize(head(backbone(images).flatten(1)).float(), dim=1)
            sup = supcon_loss(z, labels, args.temperature)
            unif = uniformity_loss(z, args.uniformity_t)
            loss = sup + args.uniformity_weight * unif

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += loss.item()
            running["supcon"] += sup.item()
            running["unif"] += unif.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        stats = evaluate(backbone, val_loader, device)
        averaged = {key: value / steps_per_epoch for key, value in running.items()}
        log(f"epoch {epoch + 1}: train={averaged} val={stats}")

        score = (round(stats["top1"], 3), -stats["offdiag_cos"])
        if best_score is None or score > tuple(best_score):
            best_score = score
            torch.save(backbone.state_dict(), output_dir / "resnet34_supcon_best.pth")
        torch.save(backbone.state_dict(), output_dir / "resnet34_supcon_last.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
            },
            state_path,
        )


if __name__ == "__main__":
    main()
