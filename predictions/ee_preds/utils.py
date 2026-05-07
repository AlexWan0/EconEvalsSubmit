import pandas as pd
from typeguard import check_type

from .types import Turn
from ee_retrieval.utils import convo_to_str, convo_to_str_human_only


def df_to_convos(turns_subset_df: pd.DataFrame) -> list[list[Turn]]:
    assert 'convo_subset' in turns_subset_df.columns, ('convo_subset', turns_subset_df.columns)

    results: list[list[Turn]] = []

    for i, row in turns_subset_df.iterrows():
        args_minimal = [
            {
                'role': t['role'],
                'content': t['content']
            }
            for t in row['convo_subset']
        ]
        check_type(args_minimal, list[Turn])

        assert args_minimal[-1]['role'] == 'user'

        results.append(args_minimal) # type: ignore
    
    return results

def normalize_model_str(model_str: str) -> str:
    return model_str.replace('/', ':')
