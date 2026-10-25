from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import os
import pickle
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from structured_codebook import STRUCTURE_ARITY, STRUCTURE_BITS  # noqa: E402


def load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vit = load_module("train_model_vit", "10_train_model_vit.py")
base = vit.base

STRUCTURE_BY_BITS = {bits: symbol for symbol, bits in STRUCTURE_BITS.items()}
RADICAL_BITS = 64


@dataclass
class TreeNode:
    symbol: str
    start: int
    end: int
    children: list[TreeNode] = field(default_factory=list)


def decode_tree(bits: tuple[int, ...], radicals: dict[tuple[int, ...], str]) -> TreeNode | None:
    """Reads a class code front to back: 4-bit structure symbols and 64-bit radical codes."""

    def parse(position: int) -> Iterator[tuple[TreeNode, int]]:
        if position + 4 <= len(bits):
            symbol = STRUCTURE_BY_BITS.get(bits[position : position + 4])
            if symbol is not None:
                for children, end in parse_children(position + 4, STRUCTURE_ARITY[symbol]):
                    yield TreeNode(symbol, position, position + 4, children), end
        if position + RADICAL_BITS <= len(bits):
            radical = radicals.get(bits[position : position + RADICAL_BITS])
            if radical is not None:
                yield TreeNode(radical, position, position + RADICAL_BITS), position + RADICAL_BITS

    def parse_children(position: int, count: int) -> Iterator[tuple[list[TreeNode], int]]:
        if count == 0:
            yield [], position
            return
        for child, end in parse(position):
            for rest, final_end in parse_children(end, count - 1):
                yield [child, *rest], final_end

    for node, end in parse(0):
        if end == len(bits):
            return node
    return None


def tree_lines(node: TreeNode, predicted: np.ndarray, code: np.ndarray, prefix: str = "", connector: str = "") -> Iterator[str]:
    matched = int((predicted[node.start : node.end] == code[node.start : node.end]).sum())
    length = node.end - node.start
    yield f"{prefix}{connector}{node.symbol}  ビット[{node.start}:{node.end}] 一致 {matched}/{length} ({matched / length:.0%})"
    child_prefix = prefix if not connector else prefix + ("   " if connector.startswith("└") else "│  ")
    for index, child in enumerate(node.children):
        last = index == len(node.children) - 1
        yield from tree_lines(child, predicted, code, child_prefix, "└─ " if last else "├─ ")


def describe_class(
    key: str,
    code: np.ndarray,
    predicted: np.ndarray,
    entries: dict[str, Any],
    radicals: dict[tuple[int, ...], str],
) -> list[str]:
    info = entries.get(key, {})
    length = int(np.count_nonzero(code))
    code_bits = tuple(int(value) for value in code[:length])
    matched = int((predicted[:length] == code[:length]).sum())
    lines = [f"{info.get('character', '?')} ({key})  IDS: {info.get('ids', '')}  全体一致 {matched}/{length} ({matched / length:.1%})"]
    tree = decode_tree(code_bits, radicals)
    if tree is None:
        lines.append("(コードを木に復元できませんでした)")
    else:
        lines.extend(tree_lines(tree, predicted, code))
    return lines


def thumbnail_html(path: str) -> str:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((96, 96))
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
    except OSError:
        return "(画像を読み込めません)"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" width="96" alt="input">'


def code_block(predicted: np.ndarray) -> str:
    text = "".join("+" if value > 0 else "-" for value in predicted)
    return "\n".join(text[start : start + 100] for start in range(0, len(text), 100))


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    model = vit.ViTFareRecognitionModel(codebook_dim=int(checkpoint["codebook_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), class_to_index


def codebook_matrix_for(class_to_index: dict[str, int], codebook: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    keys = [""] * len(class_to_index)
    for key, index in class_to_index.items():
        keys[index] = key
    matrix = torch.stack([torch.tensor(codebook[key], dtype=torch.float32) for key in keys]).to(device)
    return matrix, keys


@torch.no_grad()
def scan_group(
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    class_to_index: dict[str, int],
    matrix: torch.Tensor,
    structured: bool,
    device: torch.device,
    per_group: int,
    scan_limit: int,
    batch_size: int,
    workers: int,
    rng: random.Random,
) -> dict[str, Any]:
    pool = [entry for entry in entries if entry["unicode"] in class_to_index]
    rng.shuffle(pool)
    pool = pool[:scan_limit]
    loader = DataLoader(
        vit.ViTCharacterDataset(pool),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=base.safe_collate,
    )
    half = per_group // 2
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    scanned = 0
    correct_count = 0
    for batch in loader:
        if batch is None:
            continue
        outputs = model(batch["image"].to(device))
        scores = base.codebook_logits(outputs, matrix, structured)
        top_scores, top_indices = scores.topk(3, dim=1)
        for row, (path, unicode_key) in enumerate(zip(batch["image_path"], batch["unicode"])):
            is_correct = int(top_indices[row, 0]) == class_to_index[unicode_key]
            sample = {
                "path": path,
                "true_key": unicode_key,
                "predicted": outputs[row].cpu().numpy().astype(np.int8),
                "top3": [(int(top_indices[row, rank]), float(top_scores[row, rank])) for rank in range(3)],
                "correct": is_correct,
            }
            scanned += 1
            correct_count += is_correct
            (correct if is_correct else wrong).append(sample)
        if len(correct) >= half and len(wrong) >= half:
            break
    chosen = correct[:half] + wrong[:half]
    if len(chosen) < per_group:
        extra = correct[half:] if len(correct) > half else wrong[half:]
        chosen += extra[: per_group - len(chosen)]
    return {"samples": chosen, "scanned": scanned, "correct_count": correct_count}


def casia_entries(manifest: Path, class_to_index: dict[str, int], codebook: dict[str, np.ndarray], label_map: dict[str, str], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed = {key for key in class_to_index if base.is_valid_unicode_codebook_key(key)}
    entries = []
    for line in rng.sample(lines, min(limit, len(lines))):
        path = Path(line).resolve()
        label = base.resolve_pretrain_true_label(path, codebook, allowed, label_map)
        if label is not None:
            entries.append({"book_id": "casia", "unicode": label, "label": class_to_index[label], "image_path": str(path)})
    return entries


def kuzushiji_test_groups(args: argparse.Namespace, codebook: dict[str, np.ndarray], cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    print("くずし字データの分割を再現します（初回のみ数分かかります）...", flush=True)
    _, entries = base.collect_class_samples(Path(args.data_root), codebook, manifest_path=Path(args.main_manifest))
    _, _, test_entries, _ = base.build_seen_unseen_sample_split(
        entries,
        train_class_ratio=args.finetune_train_class_ratio,
        train_sample_ratio=args.train_ratio,
        validation_sample_ratio=args.validation_sample_ratio,
        seed=args.split_seed,
    )
    groups = {
        "seen": [entry for entry in test_entries if entry.get("split") == "test_seen"],
        "unseen": [entry for entry in test_entries if entry.get("split") == "test_unseen"],
    }
    with cache_path.open("wb") as handle:
        pickle.dump(groups, handle)
    return groups


def write_group(
    lines: list[str],
    title: str,
    result: dict[str, Any],
    keys: list[str],
    matrix_cpu: np.ndarray,
    entries: dict[str, Any],
    radicals: dict[tuple[int, ...], str],
    class_to_index: dict[str, int],
) -> None:
    scanned, correct_count = result["scanned"], result["correct_count"]
    lines += [f"## {title}", "", f"走査 {scanned} 件中、正解 {correct_count} 件（走査範囲での正解率 {correct_count / max(1, scanned):.1%}）。"
              "正解・不正解を半々で抽出しています（片方が足りない場合はもう片方で補充）。", ""]
    for number, sample in enumerate(result["samples"], start=1):
        true_key = sample["true_key"]
        predicted = sample["predicted"]
        top_key = keys[sample["top3"][0][0]]
        info = entries.get(true_key, {})
        lines += [
            f"### {title} #{number}  {'正解' if sample['correct'] else '不正解'}",
            "",
            thumbnail_html(sample["path"]),
            "",
            f"- 入力画像: `{Path(sample['path']).resolve()}`",
            f"- 正解ラベル: {info.get('character', '?')} ({true_key})  IDS: {info.get('ids', '')}",
            "",
            "| 順位 | 文字 | Unicode | 類似度 (a) 学習時スコア | 類似度 (b) マスク付きcos | IDS |",
            "|---|---|---|---|---|---|",
        ]
        for rank, (index, score) in enumerate(sample["top3"], start=1):
            candidate = keys[index]
            candidate_info = entries.get(candidate, {})
            lines.append(
                f"| {rank} | {candidate_info.get('character', '?')} | {candidate} | {score:.4f} | {score / 64:.4f} | {candidate_info.get('ids', '')} |"
            )
        lines += ["", "推測 (1位) の木構造（構造文字・部首文字ごとの予測ビット一致）:", "", "```"]
        lines += describe_class(top_key, matrix_cpu[class_to_index[top_key]], predicted, entries, radicals)
        lines += ["```", "", "正解の木構造（同じ予測ビットとの一致）:", "", "```"]
        lines += describe_class(true_key, matrix_cpu[class_to_index[true_key]], predicted, entries, radicals)
        lines += ["```", "", "<details><summary>ViTの出力した1212ビット二値化コード（+ = +1, - = -1）</summary>", "", "```",
                  code_block(predicted), "```", "", "</details>", ""]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a markdown report of ViT predictions with decoded code trees.")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "outputs/261019_ViT_umap_cos/training")
    parser.add_argument("--codebook", type=Path, default=ROOT / "outputs/261019_ViT_umap_cos/final_codebook_1212.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/261023_ViT_result_outline/prediction_report.md")
    parser.add_argument("--data-root", default="../kuzushiji-recognition/char_sep_datas")
    parser.add_argument("--main-manifest", default="outputs/manifests/main_manifest.txt")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/261007_correct_dict/class_union_audit.json")
    parser.add_argument("--samples", type=int, default=30, help="Samples per group.")
    parser.add_argument("--scan-limit", type=int, default=5000, help="Maximum samples inspected per group while searching for correct and wrong ones.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used by training for the fine-tuning split.")
    parser.add_argument("--finetune-train-class-ratio", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-sample-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codebook = base.load_codebook(args.codebook)
    structured = bool(getattr(codebook, "structured", False))
    with args.codebook.open("rb") as handle:
        payload = pickle.load(handle)
    entries = payload["entries"]
    radical_path = Path(payload["radical_codes_path"])
    radical_path = radical_path if radical_path.is_absolute() else ROOT / radical_path
    with radical_path.open("rb") as handle:
        radicals = {tuple(int(v) for v in code): radical for radical, code in pickle.load(handle)["radical_to_code"].items()}

    header = [
        "# ViT 推論結果レポート",
        "",
        f"- 作成日時: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- コードブック: `{args.codebook}`（{len(codebook)} クラス、1212ビット）",
        f"- 部首コード表: `{radical_path}`（{len(radicals)} 件）",
        f"- 学習出力: `{args.training_dir}`",
        "- 既知クラス = 学習に使ったクラス、未知クラス = 学習から除外したクラス",
        "- 類似度 (a) は学習・評価に使うスコア（予測コードとクラスコードの内積 × 64 / 有効ビット長）。"
        "(b) は有効ビットに限ったcos類似度で、(a) = 64 × (b) のため順位は同じです。",
        "- 木構造は、クラスコードを先頭から読み（構造文字4ビット、部首文字64ビット）、部首コード表で部首を引いて復元しています。"
        "各行の「一致」は、ViTの予測ビットとそのクラスコードのビット一致数です。",
        "",
    ]
    lines = list(header)

    groups: list[tuple[str, Path, str]] = [
        ("くずし字 既知クラス", args.training_dir / "best_fare_model.pth", "kuzushiji_seen"),
        ("くずし字 未知クラス", args.training_dir / "best_fare_model.pth", "kuzushiji_unseen"),
        ("CASIA 既知クラス", args.training_dir / "pretrain_best_fare_model.pth", "casia_seen"),
        ("CASIA 未知クラス", args.training_dir / "pretrain_best_fare_model.pth", "casia_unseen"),
    ]
    kuzushiji_groups = kuzushiji_test_groups(args, codebook, output_path.parent / "finetune_test_entries.pkl")
    label_map = base.load_pretrain_folder_label_map(args.audit)
    loaded: dict[Path, tuple[torch.nn.Module, dict[str, int], torch.Tensor, list[str], np.ndarray]] = {}

    for title, checkpoint, group_id in groups:
        if checkpoint not in loaded:
            model, class_to_index = load_checkpoint_model(checkpoint, device)
            matrix, keys = codebook_matrix_for(class_to_index, codebook, device)
            loaded[checkpoint] = (model, class_to_index, matrix, keys, matrix.cpu().numpy())
        model, class_to_index, matrix, keys, matrix_cpu = loaded[checkpoint]
        if group_id.startswith("kuzushiji"):
            group_entries = kuzushiji_groups[group_id.split("_")[1]]
        else:
            manifest = args.training_dir / f"pretrain_test_{group_id.split('_')[1]}_manifest.txt"
            group_entries = casia_entries(manifest, class_to_index, codebook, label_map, args.scan_limit, rng)
        print(f"{title}: 候補 {len(group_entries)} 件から推論します", flush=True)
        result = scan_group(
            model, group_entries, class_to_index, matrix, structured, device,
            args.samples, args.scan_limit, args.batch_size, args.num_workers, rng,
        )
        write_group(lines, title, result, keys, matrix_cpu, entries, radicals, class_to_index)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {output_path}", flush=True)


if __name__ == "__main__":
    main()
