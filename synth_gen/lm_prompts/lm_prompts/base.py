from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from collections.abc import Mapping
from typing import Any, Generic
import warnings

from .utils import (
    extract_template_placeholders,
    extract_turn_placeholders,
    extract_field_names,
)
from .types import PromptInputT, PromptOutputT, PromptRenderT, Turn, DataclassLike


def _ensure_dataclass_type(tp: type[Any]) -> None:
    if not is_dataclass(tp):
        raise TypeError(f"Expected a dataclass type, got: {tp}")


def _validate_placeholders(
    placeholders: set[str],
    allowed_fields: set[str],
    *,
    context_name: str,
) -> None:
    missing = sorted(placeholders - allowed_fields)
    if missing:
        missing_str = ", ".join(missing)
        allowed_str = ", ".join(sorted(allowed_fields))
        raise ValueError(
            f"{context_name} uses unknown placeholders: {missing_str}. "
            f"Allowed fields: {allowed_str}"
        )


@dataclass
class BasePrompt(Generic[PromptInputT, PromptOutputT, PromptRenderT], ABC):
    """
    Generic base class for prompt+parser configurations.
    """

    input_type: type[PromptInputT]
    output_type: type[PromptOutputT]
    template: PromptRenderT

    def __post_init__(self) -> None:
        _ensure_dataclass_type(self.input_type)
        _ensure_dataclass_type(self.output_type)
        self.validate_template_fields()

    @abstractmethod
    def render(self, prompt_input: PromptInputT) -> PromptRenderT:
        raise NotImplementedError

    @abstractmethod
    def parse_output_str(
        self,
        output_str: str,
    ) -> PromptOutputT:
        raise NotImplementedError

    def parse_output(
        self,
        output_str: str | None,
        *,
        verbose: bool = False,
    ) -> PromptOutputT | None:
        if output_str is None:
            if verbose:
                print(f"{self.__class__.__name__}: output is None.")
            return None

        try:
            parsed = self.parse_output_str(output_str)
            if parsed is None:
                if verbose:
                    print(f"{self.__class__.__name__}: parser returned None.")
                return None
            return parsed
        except Exception as e:
            if verbose:
                print(f"{self.__class__.__name__}: failed to parse output: {e}")
            return None

    def validate_output(self, output_str: str | None) -> bool:
        return self.parse_output(output_str, verbose=False) is not None

    @abstractmethod
    def validate_template_fields(self) -> None:
        raise NotImplementedError
    
    @property
    def input_field_names(self) -> set[str]:
        return extract_field_names(self.input_type)

    @property
    def output_field_names(self) -> set[str]:
        return extract_field_names(self.output_type)


@dataclass
class StringPrompt(BasePrompt[PromptInputT, PromptOutputT, str], ABC):
    def validate_template_fields(self) -> None:
        placeholders = extract_template_placeholders(self.template)
        allowed_fields = self.input_field_names
        _validate_placeholders(
            placeholders,
            allowed_fields,
            context_name=f"{self.__class__.__name__}.template",
        )

    def render(self, prompt_input: PromptInputT) -> str:
        context = asdict(prompt_input)
        return self.template.format(**context)


@dataclass
class TurnsPrompt(BasePrompt[PromptInputT, PromptOutputT, list[Turn]], ABC):
    def validate_template_fields(self) -> None:
        placeholders = extract_turn_placeholders(self.template)
        allowed_fields = self.input_field_names
        _validate_placeholders(
            placeholders,
            allowed_fields,
            context_name=f"{self.__class__.__name__}.template",
        )

    def render(self, prompt_input: PromptInputT) -> list[Turn]:
        context = asdict(prompt_input)

        rendered_turns: list[Turn] = []
        for turn in self.template:
            rendered_turns.append(
                {
                    "role": turn["role"],
                    "content": turn["content"].format(**context),
                }
            )
        return rendered_turns
