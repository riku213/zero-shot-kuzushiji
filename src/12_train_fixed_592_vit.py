from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SRC_DIR.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "261119_fixed_bit_ViT"


def load_vit_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("fixed_592_vit_training", SRC_DIR / "10_train_model_vit.py")
    if spec is None or spec.loader is None:
        raise ImportError("既存ViT学習モジュールを読み込めません。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_defaults(arguments: list[str]) -> list[str]:
    defaults = [
        ("--codebook", str(OUTPUT_ROOT / "fixed_592_codebook.pkl")),
        ("--output-dir", str(OUTPUT_ROOT / "training")),
        ("--checkpoint-path", str(OUTPUT_ROOT / "training" / "best_fare_model.pth")),
        ("--pretrain-checkpoint-path", str(OUTPUT_ROOT / "training" / "pretrain_best_fare_model.pth")),
        ("--state-path", str(OUTPUT_ROOT / "training" / "training_state.pth")),
        ("--pretrain-state-path", str(OUTPUT_ROOT / "training" / "pretrain_training_state.pth")),
        ("--metadata-path", str(OUTPUT_ROOT / "training" / "run_metadata.json")),
        ("--manifest-path", str(OUTPUT_ROOT / "manifests" / "kuzushiji_images.txt")),
        ("--pretrain-root", str(WORKSPACE_ROOT.parent / "kuzushiji-recognition" / "CASIA-HWDB")),
        ("--pretrain-manifest-path", str(OUTPUT_ROOT / "manifests" / "casia_train_images.txt")),
        ("--pretrain-test-manifest-path", str(OUTPUT_ROOT / "manifests" / "casia_test_images.txt")),
        ("--pretrain-label-map", str(WORKSPACE_ROOT / "outputs" / "261007_correct_dict" / "class_union_audit.json")),
        ("--pretrain-split-manifest-train", str(OUTPUT_ROOT / "manifests" / "casia_train.txt")),
        ("--pretrain-split-manifest-seen-test", str(OUTPUT_ROOT / "manifests" / "casia_test_seen.txt")),
        ("--pretrain-split-manifest-unseen-test", str(OUTPUT_ROOT / "manifests" / "casia_test_unseen.txt")),
        ("--pretrain-split-manifest-val-seen", str(OUTPUT_ROOT / "manifests" / "casia_val_seen.txt")),
        ("--pretrain-split-manifest-val-unseen", str(OUTPUT_ROOT / "manifests" / "casia_val_unseen.txt")),
        ("--finetune-split-manifest-train", str(OUTPUT_ROOT / "manifests" / "kuzushiji_train.txt")),
        ("--finetune-split-manifest-val", str(OUTPUT_ROOT / "manifests" / "kuzushiji_validation.txt")),
        ("--finetune-split-manifest-test", str(OUTPUT_ROOT / "manifests" / "kuzushiji_test.txt")),
        ("--log-path", str(OUTPUT_ROOT / "training" / "train.log")),
    ]
    provided = {argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")}
    prefix: list[str] = []
    for option, value in defaults:
        if option not in provided:
            prefix.extend((option, value))
    return [sys.argv[0], *prefix, *arguments]


def main() -> None:
    arguments = sys.argv[1:]
    if any(argument == "--init-from" or argument.startswith("--init-from=") for argument in arguments):
        raise SystemExit("この実験はImageNet重みから開始します。--init-fromは指定できません。")
    sys.argv = add_defaults(arguments)

    vit = load_vit_training_module()
    base = vit.base
    original_save_json = base.save_json
    original_load_codebook = base.load_codebook

    def load_fixed_592_codebook(path: Path) -> Any:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("layout_format") != "fixed-592-tree-v1":
            raise ValueError(f"固定592-bit codebookではありません: {path}")
        if payload.get("codebook_dim") != 592 or payload.get("max_code_length") != 592:
            raise ValueError(f"Codebook次元が592ではありません: {path}")
        if len(payload.get("codes", {})) != len(payload.get("masks", {})):
            raise ValueError(f"Codebookのcode/mask件数が一致しません: {path}")
        entries = payload.get("entries", {})
        missing_polish = [
            name for name, entry in entries.items()
            if not isinstance(entry, dict) or "polish" not in entry
        ]
        if missing_polish:
            examples = ", ".join(missing_polish[:5])
            raise ValueError(
                f"component head用polish metadataがありません: {path} "
                f"(不足{len(missing_polish)}件、例: {examples})。"
                "src/11_build_fixed_592_codebook.pyでcodebookを再生成してください。"
            )
        excluded = set(payload.get("build", {}).get("excluded_kuzushiji_unicode_keys", []))
        if len(excluded) != 41 or excluded.intersection(payload.get("codes", {})):
            raise ValueError(f"不適合41クラスの除外状態が正しくありません: {path}")
        return original_load_codebook(path)

    def save_fixed_592_metadata(path: Path, payload: dict[str, Any]) -> None:
        metadata = dict(payload)
        vit_settings = dict(metadata.get("vit_settings", {}))
        vit_settings["output_code_bits"] = 592
        vit_settings["initialization"] = "ImageNet ViT-B/16 weights; no experiment checkpoint"
        vit_settings["fixed_codebook_format"] = "fixed-592-tree-v1"
        vit_settings["excluded_kuzushiji_classes"] = 41
        metadata["vit_settings"] = vit_settings
        codebook_path = Path(vit.SETTINGS.codebook_path)
        metadata["fixed_592_codebook"] = {
            "path": str(codebook_path),
            "layout": "4-bit root + 3 * (4-bit branch + 3 * 64-bit leaf)",
            "class_filter": "classes absent from the strict fixed-layout codebook are excluded from every split",
            "excluded_list": str(codebook_path.with_name("codebook_exclusions.json")),
        }
        original_save_json(path, metadata)

    base.save_json = save_fixed_592_metadata
    base.load_codebook = load_fixed_592_codebook
    torch.set_float32_matmul_precision("high")
    vit.main()


if __name__ == "__main__":
    main()
