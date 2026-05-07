from typing import Iterator
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

from .base import ConvoDataset, Turn


class Wildchat48Dataset(ConvoDataset):
    def __init__(self, support_get_item: bool = False):
        if support_get_item:
            hf_dataset = load_dataset('allenai/WildChat-4.8M')
            self.train_col = hf_dataset['train'] # type: ignore
        else:
            hf_dataset = load_dataset('allenai/WildChat-4.8M', streaming=True)
            self.train_col = None
    
        self.train = hf_dataset['train'] # type: ignore

        super().__init__(support_get_item=support_get_item)

    def iter_turns(self) -> Iterator[list[Turn]]:
        def _unpack() -> Iterator[list[Turn]]:
            for row in self.train:
                yield row['conversation'] # type: ignore

        return _unpack()

    def get_convo(self, idx: int) -> list[Turn]:
        if not self.support_get_item:
            raise ValueError('random access is not supported')

        assert self.train_col is not None

        return self.train_col[int(idx)]['conversation']

    def __len__(self) -> int:
        return 3_199_860
