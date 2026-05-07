from typing import Iterator
from datasets import load_dataset

from .base import ConvoDataset, Turn


class WildchatDataset(ConvoDataset):
    def __init__(self, support_get_item: bool = False):
        if support_get_item:
            hf_dataset = load_dataset('allenai/WildChat-1M')
            # self.text_col = hf_dataset['train']['conversation'] # type: ignore
        else:
            hf_dataset = load_dataset('allenai/WildChat-1M', streaming=True)
            # self.text_col = None
    
        self.train = hf_dataset['train'] # type: ignore

        super().__init__(support_get_item=support_get_item)

    def iter_turns(self) -> Iterator[list[Turn]]:
        def _unpack() -> Iterator[list[Turn]]:
            for row in self.train:
                yield row['conversation'] # type: ignore

        return _unpack()

    def get_convo(self, idx: int) -> list[Turn]:
        if not self.support_get_item:
            raise ValueError("support_get_item set to False")

        row = self.train[int(idx)] # type: ignore
        return row['conversation']

    def __len__(self) -> int:
        return 837_989
