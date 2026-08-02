from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from fare_pipeline import RadicalFeatureRecord, load_rendered_radicals, image_to_tensor, save_pickle_and_json


class RadicalImageDataset(Dataset):
    def __init__(self, rendered_radicals: list, image_size: int = 96) -> None:
        self.rendered_radicals = rendered_radicals
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rendered_radicals)

    def __getitem__(self, index: int):
        record = self.rendered_radicals[index]
        tensor = image_to_tensor(Path(record.image_path), image_size=self.image_size)
        return {
            "image": tensor,
            "radical": record.radical,
            "codepoint": record.codepoint,
            "image_path": record.image_path,
            "renderable": record.renderable,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ResNet34 features from rendered radicals.")
    parser.add_argument(
        "--manifest",
        default="outputs/radical_render_manifest.pkl",
        help="Render manifest produced by 2_render_radicals.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/radical_features.pkl",
        help="Output pickle storing radical features.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=96, help="Input image size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_feature_extractor(device: torch.device) -> nn.Module:
    try:
        weights = models.ResNet34_Weights.DEFAULT
        backbone = models.resnet34(weights=weights)
    except AttributeError:
        backbone = models.resnet34(pretrained=True)

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rendered_radicals = load_rendered_radicals(Path(args.manifest))

    dataset = RadicalImageDataset(rendered_radicals, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    extractor = build_feature_extractor(device)

    feature_records: list[RadicalFeatureRecord] = []
    for batch in dataloader:
        batch_images = batch["image"].to(device)
        features = extractor(batch_images).flatten(1).cpu()

        batch_size = batch_images.shape[0]
        for index in range(batch_size):
            feature_vector = features[index].tolist()
            feature_records.append(
                RadicalFeatureRecord(
                    radical=batch["radical"][index],
                    codepoint=batch["codepoint"][index],
                    image_path=batch["image_path"][index],
                    renderable=bool(batch["renderable"][index]),
                    feature=feature_vector,
                )
            )

    payload = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "device": str(device),
        "feature_dim": len(feature_records[0].feature) if feature_records else 0,
        "records": [record.to_dict() for record in feature_records],
    }
    save_pickle_and_json(payload, Path(args.output))
    print(f"Extracted {len(feature_records)} radical features to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
