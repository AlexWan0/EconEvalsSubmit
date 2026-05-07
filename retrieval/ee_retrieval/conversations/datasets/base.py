from abc import ABC, abstractmethod
from typing import Iterator, Callable

from ...utils.data import Turn, convo_to_str


class ConvoDataset(ABC):
    def __init__(self, support_get_item: bool = True):
        self.support_get_item = support_get_item

    @abstractmethod
    def iter_turns(self) -> Iterator[list[Turn]]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_convo(self, idx: int) -> list[Turn]:
        raise NotImplementedError()

    def get_convo_string(self, idx: int) -> str:
        return convo_to_str(self.get_convo(idx))

    def iter_strings(self, to_string: Callable[[list[Turn]], str] = convo_to_str) -> Iterator[str]:
        for convo in self.iter_turns():
            yield to_string(convo)
