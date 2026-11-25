from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from src.structured_codebook import IDSNode, STRUCTURE_ARITY, STRUCTURE_SET, ids_to_polish


SLOT = "□"


@dataclass(frozen=True)
class ReconstructionResult:
    ids: str
    character: str | list[str]
    dictionary_hit: bool


class ReverseIDSIndex:
    """Maps a complete prefix IDS expression to one or more Unicode characters."""

    def __init__(self, candidates_by_ids: Mapping[str, Iterable[str]]) -> None:
        self._candidates: dict[str, tuple[str, ...]] = {}
        self._unresolved_characters: tuple[str, ...] = ()
        for expression, candidates in candidates_by_ids.items():
            ordered = tuple(sorted(set(candidates), key=_character_sort_key))
            if ordered:
                self._candidates[expression] = ordered

    @classmethod
    def from_trees(
        cls, trees_by_character: Mapping[str, IDSNode | None]
    ) -> ReverseIDSIndex:
        candidates: defaultdict[str, list[str]] = defaultdict(list)
        for character, tree in trees_by_character.items():
            if tree is not None:
                candidates[ids_to_polish(tree)].append(character)
        index = cls(candidates)
        index._unresolved_characters = tuple(
            sorted(
                (character for character, tree in trees_by_character.items() if tree is None),
                key=_character_sort_key,
            )
        )
        return index

    @property
    def unresolved_characters(self) -> tuple[str, ...]:
        return self._unresolved_characters

    @classmethod
    def from_class_trees(
        cls, trees_by_character: Mapping[str, IDSNode | None]
    ) -> ReverseIDSIndex:
        return cls.from_trees(trees_by_character)

    def lookup(self, ids_expression: str) -> str | list[str]:
        candidates = self._candidates.get(ids_expression)
        if not candidates:
            return ids_expression
        if len(candidates) == 1:
            return candidates[0]
        return list(candidates)

    def contains(self, ids_expression: str) -> bool:
        return ids_expression in self._candidates


def tree_to_skeleton_components(tree: IDSNode) -> tuple[list[str], list[str]]:
    skeleton: list[str] = []
    components: list[str] = []

    def visit(node: IDSNode) -> None:
        if node.symbol in STRUCTURE_SET:
            expected_arity = STRUCTURE_ARITY[node.symbol]
            if len(node.children) != expected_arity:
                raise ValueError(
                    f"IDS operator {node.symbol!r} requires {expected_arity} children, "
                    f"got {len(node.children)}."
                )
            skeleton.append(node.symbol)
            for child in node.children:
                visit(child)
            return
        if node.children:
            raise ValueError(f"Terminal IDS token {node.symbol!r} cannot have children.")
        skeleton.append(SLOT)
        components.append(node.symbol)

    visit(tree)
    return skeleton, components


def restore_ids(skeleton: Iterable[str], components: Iterable[str]) -> str:
    skeleton_tokens = list(skeleton)
    component_tokens = list(components)
    skeleton_position = 0
    component_position = 0

    def restore_node() -> str:
        nonlocal skeleton_position, component_position
        if skeleton_position >= len(skeleton_tokens):
            raise ValueError("Skeleton ended before the IDS tree was complete.")
        token = skeleton_tokens[skeleton_position]
        skeleton_position += 1

        if token == SLOT:
            if component_position >= len(component_tokens):
                raise ValueError("Component count does not match the number of skeleton slots.")
            component = component_tokens[component_position]
            component_position += 1
            if component in STRUCTURE_SET:
                raise ValueError(f"Component {component!r} is not a terminal part token.")
            return component

        arity = STRUCTURE_ARITY.get(token)
        if arity is None:
            raise ValueError(f"Unknown skeleton token {token!r}; expected an operator or {SLOT!r}.")
        return token + "".join(restore_node() for _ in range(arity))

    ids_expression = restore_node()
    if skeleton_position != len(skeleton_tokens):
        raise ValueError("Skeleton contains tokens after the IDS tree is complete.")
    if component_position != len(component_tokens):
        raise ValueError("Component count does not match the number of skeleton slots.")
    return ids_expression


def reconstruct(
    skeleton: Iterable[str],
    components: Iterable[str],
    reverse_index: ReverseIDSIndex,
) -> ReconstructionResult:
    ids_expression = restore_ids(skeleton, components)
    character = reverse_index.lookup(ids_expression)
    return ReconstructionResult(
        ids=ids_expression,
        character=character,
        dictionary_hit=reverse_index.contains(ids_expression),
    )


def _character_sort_key(character: str) -> tuple[int, str]:
    return (ord(character[0]) if character else -1, character)