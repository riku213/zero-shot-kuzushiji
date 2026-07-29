from __future__ import annotations

import argparse
from pathlib import Path

from fare_pipeline import build_inventory, save_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse IDS files and extract unique radicals for FaRE.")
    parser.add_argument(
        "--ids-files",
        nargs="+",
        default=["dataset/ids_text/ids.txt", "dataset/ids_text/ids-cdp.txt"],
        help="Input IDS files to parse.",
    )
    parser.add_argument(
        "--output",
        default="outputs/radical_inventory.pkl",
        help="Output pickle file for the parsed radical inventory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids_files = [Path(path) for path in args.ids_files]
    inventory = build_inventory(ids_files)
    save_inventory(inventory, Path(args.output))

    print(f"Parsed {len(inventory.ids_entries)} IDS entries from {len(ids_files)} files.")
    print(f"Unique radicals: {len(inventory.radicals)}")
    print(f"Structures: {''.join(inventory.structures)}")
    print(f"Saved inventory to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
