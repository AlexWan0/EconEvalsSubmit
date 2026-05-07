from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, TypeVar, TypedDict


class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


PromptInputT = TypeVar("PromptInputT", bound=DataclassLike)
PromptOutputT = TypeVar("PromptOutputT", bound=DataclassLike)


class Turn(TypedDict):
    role: str
    content: str

# A rendered prompt can be either a plain string or chat-like turns.
RenderedPrompt = str | list[Turn]
PromptRenderT = TypeVar("PromptRenderT", bound=RenderedPrompt)
