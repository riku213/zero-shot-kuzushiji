from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from umap import UMAP

from fare_pipeline import FareCodeRecord, load_render_manifest, save_pickle_and_json, set_reproducible_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 64-dim FaRE codes from radical features.")
    parser.add_argument(
        "--features",
        default="outputs/radical_features.pkl",
        help="Feature pickle produced by 3_extract_features.py.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/radical_render_manifest.pkl",
        help="Render manifest produced by 2_render_radicals.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/radical_codes.pkl",
        help="Output pickle storing 64-dim FaRE codes.",
    )
    parser.add_argument("--n-components", type=int, default=64, help="UMAP target dimension.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic output.")
    return parser.parse_args()


def load_feature_payload(features_path: Path) -> dict:
    import pickle

    with features_path.open("rb") as handle:
        return pickle.load(handle)


def build_unique_binary_code(existing_codes: set[tuple[int, ...]], rng: np.random.Generator, length: int = 64) -> list[int]:
    while True:
        candidate = rng.choice([-1, 1], size=length).tolist()
        candidate_key = tuple(candidate)
        if candidate_key not in existing_codes:
            existing_codes.add(candidate_key)
            return candidate


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.random_state)
    rng = np.random.default_rng(args.random_state)

    feature_payload = load_feature_payload(Path(args.features))
    render_manifest = load_render_manifest(Path(args.manifest))
    alien_radicals = set(render_manifest.get("alien_radicals", []))

    feature_records = feature_payload.get("records", [])
    renderable_records = [record for record in feature_records if record.get("renderable", True) and record.get("radical") not in alien_radicals]
    alien_records = [record for record in feature_records if not record.get("renderable", True) or record.get("radical") in alien_radicals]

    if not renderable_records:
        raise RuntimeError("No renderable radicals were found in the feature payload.")

    feature_matrix = np.asarray([record["feature"] for record in renderable_records], dtype=np.float32)
    if feature_matrix.shape[0] < 2:
        raise RuntimeError("UMAP requires at least two renderable radicals.")

    n_neighbors = min(15, feature_matrix.shape[0] - 1)
    reducer = UMAP(
        n_components=args.n_components,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=args.random_state,
        transform_seed=args.random_state,
        init="random",
    )
    projected = reducer.fit_transform(feature_matrix)

    existing_codes: set[tuple[int, ...]] = set()
    payload_records: list[FareCodeRecord] = []

    for record, vector in zip(renderable_records, projected, strict=False):
        fare_code = [1 if value > 0 else -1 for value in np.tanh(vector).tolist()]
        code_key = tuple(fare_code)
        if code_key in existing_codes:
            fare_code = build_unique_binary_code(existing_codes, rng, length=args.n_components)
        else:
            existing_codes.add(code_key)

        payload_records.append(
            FareCodeRecord(
                radical=record["radical"],
                codepoint=record["codepoint"],
                image_path=record["image_path"],
                renderable=True,
                feature=record["feature"],
                projected=[float(value) for value in vector.tolist()],
                fare_code=fare_code,
                source="fare",
            )
        )

    for record in alien_records:
        fare_code = build_unique_binary_code(existing_codes, rng, length=args.n_components)
        payload_records.append(
            FareCodeRecord(
                radical=record["radical"],
                codepoint=record["codepoint"],
                image_path=record["image_path"],
                renderable=False,
                feature=record["feature"],
                projected=[float(value) for value in fare_code],
                fare_code=fare_code,
                source="random_alien",
            )
        )

    payload_records.sort(key=lambda item: item.radical)
    payload = {
        "source_features": str(Path(args.features).resolve()),
        "source_manifest": str(Path(args.manifest).resolve()),
        "n_components": args.n_components,
        "records": [record.to_dict() for record in payload_records],
        "radical_to_code": {record.radical: record.fare_code for record in payload_records},
    }
    save_pickle_and_json(payload, Path(args.output))
    print(f"Generated {len(payload_records)} FaRE codes to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
