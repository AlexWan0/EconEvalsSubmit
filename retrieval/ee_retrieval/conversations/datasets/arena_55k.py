from typing import Iterator
from datasets import load_dataset
import pandas as pd


from ...utils import Turn, cached_func
from .base import ConvoDataset

# https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora
def _process_text(text: str) -> list[str]:
    return [
        x.encode('utf-8', 'replace').decode()
        for x in eval(text, {"null": ""})
    ]

@cached_func(fn_ident_override="arena_55k")
def _load_arena_55k() -> pd.DataFrame:
    arena_55k = load_dataset('lmarena-ai/arena-human-preference-55k')
    df = pd.DataFrame(arena_55k['train']) # type: ignore

    df.loc[:, 'prompt_parsed'] = df['prompt'].apply(_process_text)
    df.loc[:, 'response_a_parsed'] = df['response_a'].apply(_process_text)
    df.loc[:, 'response_b_parsed'] = df['response_b'].apply(_process_text)

    return df

class Arena55kDataset(ConvoDataset):
    def __init__(self, use_a: bool = True, support_get_item: bool = True):
        assert support_get_item

        self.df = _load_arena_55k()

        self.use_a = use_a

        super().__init__(support_get_item=support_get_item)
    
    def iter_turns(self) -> Iterator[list[Turn]]:
        for _, row in self.df.iterrows():
            target_responses: list[str] = row['response_a_parsed'] if self.use_a else row['response_b_parsed']
            user_prompts: list[str] = row['prompt_parsed']

            turns = []
            for user_chat, model_chat in zip(user_prompts, target_responses):
                turns.append({
                    'role': 'user',
                    'content': user_chat
                })

                turns.append({
                    'role': 'assistant',
                    'content': model_chat
                })

            yield turns

    def get_convo(self, idx) -> list[Turn]:
        row = self.df.iloc[idx]

        target_responses: list[str] = row['response_a_parsed'] if self.use_a else row['response_b_parsed']
        user_prompts: list[str] = row['prompt_parsed']

        turns = []
        for user_chat, model_chat in zip(user_prompts, target_responses):
            turns.append({
                'role': 'user',
                'content': user_chat
            })

            turns.append({
                'role': 'assistant',
                'content': model_chat
            })

        return turns

    def __len__(self) -> int:
        return len(self.df)
