from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from fare_pipeline import STRUCTURE_SYMBOLS
except ModuleNotFoundError:
    from src.fare_pipeline import STRUCTURE_SYMBOLS


STRUCTURE_ARITY = {symbol: 3 if symbol in {"⿲", "⿳"} else 2 for symbol in STRUCTURE_SYMBOLS}
STRUCTURE_SET = set(STRUCTURE_SYMBOLS)
STRUCTURE_BITS = {
    symbol: tuple(1 if bit == "1" else -1 for bit in f"{index:04b}")
    for index, symbol in enumerate(STRUCTURE_SYMBOLS)
}
CODEBOOK_FORMAT = "fare-structured-polish-v1"


@dataclass(frozen=True)
class IDSNode:
    symbol: str
    children: tuple[IDSNode, ...] = ()


class CodebookMapping(dict[str, np.ndarray]):
    def __init__(self, *args: Any, structured: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.structured = structured


def tokenize_ids(expression: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if char == "&":
            end = expression.find(";", index + 1)
            if end >= 0:
                tokens.append(expression[index : end + 1])
                index = end + 1
                continue
        tokens.append(char)
        index += 1
    return tokens


def parse_ids_expression(expression: str) -> IDSNode:
    tokens = tokenize_ids(expression)
    position = 0

    def parse_node() -> IDSNode:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("IDS expression ended before all operator operands were read.")
        symbol = tokens[position]
        position += 1
        arity = STRUCTURE_ARITY.get(symbol, 0)
        return IDSNode(symbol, tuple(parse_node() for _ in range(arity)))

    if not tokens:
        raise ValueError("IDS expression is empty.")
    root = parse_node()
    if position != len(tokens):
        raise ValueError(f"IDS expression has {len(tokens) - position} trailing token(s).")
    return root


def canonical_ids(node: IDSNode) -> tuple[Any, ...]:
    return (node.symbol, *(canonical_ids(child) for child in node.children))


def ids_to_polish(node: IDSNode) -> str:
    return node.symbol + "".join(ids_to_polish(child) for child in node.children)


def tree_depth(node: IDSNode) -> int:
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def parse_codebook_payload(payload: Any, path: Path) -> CodebookMapping:
    if isinstance(payload, dict) and payload.get("format") == CODEBOOK_FORMAT:
        raw_codes = payload.get("codes")
        raw_masks = payload.get("masks")
        if not isinstance(raw_codes, dict) or not isinstance(raw_masks, dict):
            raise ValueError(f"Structured codebook {path} must contain codes and masks dictionaries.")
        if set(raw_codes) != set(raw_masks):
            raise ValueError(f"Structured codebook {path} has mismatched code and mask keys.")
        converted: dict[str, np.ndarray] = {}
        expected_dim = int(payload.get("max_code_length", 0))
        for key, raw_code in raw_codes.items():
            code = np.asarray(raw_code, dtype=np.float32)
            mask = np.asarray(raw_masks[key], dtype=bool)
            if code.ndim != 1 or mask.shape != code.shape:
                raise ValueError(f"Malformed code or mask for {key!r} in {path}.")
            if expected_dim and code.size != expected_dim:
                raise ValueError(f"Code length mismatch for {key!r} in {path}.")
            if np.any(code[mask] == 0) or np.any(code[~mask] != 0):
                raise ValueError(f"Code padding/mask mismatch for {key!r} in {path}.")
            converted[str(key)] = code
        return CodebookMapping(converted, structured=True)

    if isinstance(payload, dict) and "codebook" in payload and isinstance(payload["codebook"], dict):
        payload = payload["codebook"]
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        converted = {}
        for item in payload["entries"]:
            if not isinstance(item, dict):
                continue
            key = item.get("unicode") or item.get("class") or item.get("label")
            vector = item.get("code") or item.get("vector") or item.get("embedding")
            if key is not None and vector is not None:
                converted[str(key)] = np.asarray(vector, dtype=np.float32)
        if converted:
            return CodebookMapping(converted)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported codebook format in {path}: {type(payload)!r}")

    converted = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in payload.items()
        if isinstance(value, (list, tuple, np.ndarray))
    }
    if not converted:
        raise ValueError(f"No vector-based entries were found in {path}.")
    return CodebookMapping(converted)


def load_codebook(path: Path) -> CodebookMapping:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return parse_codebook_payload(payload, path)


def codebook_logits(
    outputs: torch.Tensor,
    codebook_matrix: torch.Tensor,
    structured: bool,
) -> torch.Tensor:
    logits = outputs.mm(codebook_matrix.t())
    if structured:
        valid_lengths = codebook_matrix.ne(0).sum(dim=1).clamp_min(1)
        logits = logits * (64.0 / valid_lengths.to(dtype=logits.dtype)).unsqueeze(0)
    return logits