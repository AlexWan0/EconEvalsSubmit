
import pandas as pd
from typing import Mapping, Any, Iterable, Iterator
from dataclasses import asdict, dataclass, fields, is_dataclass

from .types import DataclassLike, PromptInputT, PromptOutputT, PromptRenderT
from .base import BasePrompt
from .utils import extract_field_names


def cast_to_input(
    prompt_input_type: type[PromptInputT],
    data: Mapping[Any, object] | DataclassLike | pd.Series,
) -> PromptInputT:
    """
    Uses a subset of fields from either a dataclass or mapping to construct an
    instance of the input type. Extra fields are ignored. Missing fields error.
    """
    input_fields = extract_field_names(prompt_input_type)

    if isinstance(data, Mapping):
        missing = [f for f in input_fields if f not in data]
        if missing:
            raise ValueError(
                f"Missing fields for {prompt_input_type}: {', '.join(sorted(missing))}"
            )
        relevant_data = {f: data[f] for f in input_fields}
        return prompt_input_type(**relevant_data)

    if is_dataclass(data):
        missing = [f for f in input_fields if not hasattr(data, f)]
        if missing:
            raise ValueError(
                f"Missing fields for {prompt_input_type}: {', '.join(sorted(missing))}"
            )
        relevant_data = {f: getattr(data, f) for f in input_fields}
        return prompt_input_type(**relevant_data)

    if isinstance(data, pd.Series):
        missing = [f for f in input_fields if f not in data]
        if missing:
            raise ValueError(
                f"Missing fields for {prompt_input_type}: {', '.join(sorted(missing))}"
            )
        relevant_data = {f: data[f] for f in input_fields}
        return prompt_input_type(**relevant_data)

    raise TypeError(f"Data must be a mapping, dataclass instance, or pandas Series, got: {type(data)}")


def cast_from_df(
    prompt_input_type: type[PromptInputT],
    df: pd.DataFrame,
) -> Iterable[PromptInputT]:
    for _, row in df.iterrows():
        yield cast_to_input(prompt_input_type, row)


def render_from_df(
    prompt: BasePrompt[PromptInputT, PromptOutputT, PromptRenderT],
    df: pd.DataFrame,
) -> Iterable[PromptRenderT]:
    for _, row in df.iterrows():
        prompt_input = cast_to_input(prompt.input_type, row)
        yield prompt.render(prompt_input)
