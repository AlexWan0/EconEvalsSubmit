from typing import Iterator
from datasets import load_dataset
import pandas as pd

from .base import ConvoDataset, Turn
from ...utils import Turn, cached_func


@cached_func(fn_ident_override="arena_100k")
def _load_arena_100k() -> pd.DataFrame:
    arena_100k = load_dataset('lmarena-ai/arena-human-preference-100k')
    df = pd.DataFrame(arena_100k['train']) # type: ignore

    return df

class Arena100kDataset(ConvoDataset):
    def __init__(self, use_a: bool = True, support_get_item: bool = True):
        assert support_get_item

        self.df = _load_arena_100k()

        self.use_a = use_a

        super().__init__(support_get_item=support_get_item)
    
    def iter_turns(self) -> Iterator[list[Turn]]:
        for _, row in self.df.iterrows():
            yield row['conversation_a'] if self.use_a else row['conversation_b']

    def get_convo(self, idx: int) -> list[Turn]:
        row = self.df.iloc[idx]
        
        return row['conversation_a'] if self.use_a else row['conversation_b']

    def __len__(self) -> int:
        return len(self.df)
