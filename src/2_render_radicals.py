from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fare_pipeline import FontSet, load_inventory, render_radicals, save_render_manifest, save_render_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render radicals into images with a TTF font.")
    parser.add_argument(
        "--inventory",
        default="outputs/radical_inventory.pkl",
        help="Inventory pickle produced by 1_parse_ids.py.",
    )
    parser.add_argument(
        "--fonts",
        nargs="+",
        default=["dataset/fonts/HanaMinA.ttf", "dataset/fonts/HanaMinB.ttf"],
        help="TTF fonts in priority order; the first font covering a radical is used.",
    )
    parser.add_argument(
        "--metadata",
        default="outputs/radical_render_metadata.json",
        help="Output JSON describing the render result of every radical.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/radical_images",
        help="Directory that will receive rendered radical images.",
    )
    parser.add_argument("--image-size", type=int, default=96, help="Rendered image size in pixels.")
    parser.add_argument(
        "--manifest",
        default="outputs/radical_render_manifest.pkl",
        help="Output manifest for rendered and alien radicals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = load_inventory(Path(args.inventory))
    radicals = inventory.get("radicals", [])
    fonts = FontSet(Path(p) for p in args.fonts)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    rendered, alien_radicals = render_radicals(radicals, fonts, output_dir, image_size=args.image_size)
    save_render_manifest(rendered, alien_radicals, Path(args.manifest))
    save_render_metadata(rendered, fonts, inventory.get("source_files", []), args.image_size, Path(args.metadata))

    print(f"Rendered {len(rendered) - len(alien_radicals)} radicals into {output_dir.resolve()}")
    print(f"Saved metadata to {Path(args.metadata).resolve()}")
    print(f"Alien radicals: {len(alien_radicals)}")
    print(f"Saved manifest to {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
