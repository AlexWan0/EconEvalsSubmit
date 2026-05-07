from collections.abc import Sequence
import string
import warnings
from typing import Mapping, Sequence, cast, get_type_hints
from dataclasses import fields

from .types import Turn, DataclassLike


def extract_template_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            continue
        field_name = field_name.strip()
        if field_name:
            placeholders.add(field_name)
    return placeholders


def extract_turn_placeholders(
    turns: Sequence[Turn],
) -> set[str]:
    placeholders: set[str] = set()
    for turn in turns:
        content = turn["content"]
        placeholders |= extract_template_placeholders(content)
    return placeholders


def extract_field_names(input_type: type[DataclassLike]) -> set[str]:
    return {f.name for f in fields(input_type)}


REQUIRED_TYPES = get_type_hints(Turn)
REQUIRED_KEYS = frozenset(REQUIRED_TYPES)

def cast_turns(
    input_turns: Sequence[Mapping[str, object]],
    warn_extra_fields: bool = True,
) -> list[Turn]:
    """
    Validate and normalize a sequence of mappings into Turn objects.

    Ensures required keys exist with correct types. Extra keys are removed.
    Optionally emits a single warning if extra keys are encountered.
    Returns original dicts when no extras are present; otherwise returns
    shallow copies containing only required fields.
    """
    warned = False
    out: list[Turn] = []

    for i, turn in enumerate(input_turns):
        if not isinstance(turn, Mapping):
            raise ValueError(f"Turn {i} is not a mapping: {turn!r}")

        for key, expected_type in REQUIRED_TYPES.items():
            try:
                val = turn[key]
            except KeyError:
                raise ValueError(f"Turn {i} is missing required key {key!r}: {turn!r}")
            
            # Note: isinstance works for simple types like `str`. 
            # It will fail if Turn grows to include complex types like `list[str]` or `Union`.
            if not isinstance(val, expected_type):
                raise ValueError(
                    f"Turn {i} has wrong type for key {key!r}: expected {expected_type}, got {type(val)}"
                )

        # Fast path: It's exactly the right size AND is a standard dict.
        if len(turn) == len(REQUIRED_KEYS) and type(turn) is dict:
            out.append(cast(Turn, turn))
            continue

        # Extras exist (or it's a non-standard mapping that needs to be coerced to dict)
        extras = set(turn) - REQUIRED_KEYS
        if extras and warn_extra_fields and not warned:
            warnings.warn(f"Turn {i} has extra keys not in Turn type: {extras}")
            warned = True
        
        # Build the normalized dict and cast it cleanly for type checkers
        normalized_turn = {k: turn[k] for k in REQUIRED_KEYS}
        out.append(cast(Turn, normalized_turn))

    return out
