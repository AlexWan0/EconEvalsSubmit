from typing import Iterator
from datasets import load_dataset
import pandas as pd
import os

from .base import ConvoDataset, Turn
from ...utils import data, cached_func


@cached_func(fn_ident_override="arena_140k")
def _load_arena_140k_disk() -> pd.DataFrame:
    arena_140k = load_dataset('lmarena-ai/arena-human-preference-140k')
    df = pd.DataFrame(arena_140k['train'])[['conversation_a', 'conversation_b']] # type: ignore

    return df

def _load_arena_140k() -> pd.DataFrame:
    # if os.path.isdir('/dev/shm/.cache_memfs'):
    #     print('loading from memfs')
    #     return _load_arena_140k_memfs()

    return _load_arena_140k_disk()

class Arena140kDataset(ConvoDataset):
    def __init__(self, use_a: bool = True, support_get_item: bool = True):
        self.df = _load_arena_140k()

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

class Arena140kFixedDataset(ConvoDataset):
    def __init__(self, use_a: bool = True, support_get_item: bool = True):
        self.df = _load_arena_140k()

        self.use_a = use_a

        super().__init__(support_get_item=support_get_item)
    
    def _fix_content(self, content: list[dict[str, str]]) -> str:
        assert len(content) <= 1

        if len(content) == 0:
            return ''

        datum = content[0]

        assert datum['image'] is None
        assert datum['type'] == 'text'

        assert isinstance(datum['text'], str)

        return datum['text']

    def _fix_turns(self, convo: list[dict]) -> list[Turn]:
        convo_res: list[Turn] = []

        for turn in convo:
            turn_res: Turn = {
                'content': self._fix_content(turn['content']),
                'role': turn['role']
            }
            
            assert turn['role'] in ['user', 'assistant']

            convo_res.append(turn_res)
        
        return convo_res

    def iter_turns(self) -> Iterator[list[Turn]]:
        for _, row in self.df.iterrows():
            yield self._fix_turns(row['conversation_a']) if self.use_a else self._fix_turns(row['conversation_b'])

    def get_convo(self, idx: int) -> list[Turn]:
        row = self.df.iloc[idx]
        
        return self._fix_turns(row['conversation_a']) if self.use_a else self._fix_turns(row['conversation_b'])

    def __len__(self) -> int:
        return len(self.df)
