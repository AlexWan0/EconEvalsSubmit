import pandas as pd
from functools import reduce
import logging
logger = logging.getLogger(__name__)

from .qual_general import apply_qual_general
from .qual_general import PROMPT_MAP as GENERAL_PROMPT_MAP
from .qual_homework import PROMPTS_MODULES as HW_PROMPT_MAP
from ..utils import extract_tag, extract_answer, apply_prompt


def apply_hw_pipeline(
        df: pd.DataFrame,
        prompts_version: str = 'v6',
        output_col: str = 'is_hw'
    ) -> str:
    """Classify each row of the input dataframe as homework or not-homework.

    Adds a new column with the classification result (whether to drop the category-conversation pair; i.e., whether the conversation is judged to be a homework problem) to the dataframe in-place. Returns the added column name.

    Also stores intermediate prompts/model predictions in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing conversations to be classified. The expected columns depend on the prompt used, but, the current prompts just use `instance_text` and `instance_text_human_only`, produced by ``ee_retrieval.utils.data.convo_to_str`` and ``ee_retrieval.utils.data.convo_to_str_human_only``. The input may be pairs of conversations and categories (as long as the conversation column exists) as redundant predictions are handled by the cache.
        prompts_version (str): Prompt version to use (see: ``qual_homework/prompt_versions``).
        output_col (str, optional): Name of the column to store the predictions. Defaults to ``is_hw``.

    Returns:
        str: Name of the column containing the final boolean is-homework prediction.
    """

    apply_prompt(
        df,
        HW_PROMPT_MAP[prompts_version].HW_PROMPT,
        f'pred_{output_col}',
        model_name='openai/gpt-4.1-mini',
        parser=lambda x: extract_answer(x, choices=['Yes', 'No'])
    )

    df[f'_{output_col}_thinking'] = df[f'_pred_{output_col}_model_output'].apply(
        lambda x: extract_tag(x.output, tag='thinking')
    )

    df[output_col] = (df[f'pred_{output_col}'] == 'Yes')

    return output_col

def apply_general_qual_pipeline(
        df: pd.DataFrame,
        output_col: str = 'is_lq',
        key: str = 'v5'
    ) -> str:
    """Classify each row of the input dataframe for low-quality samples.

    Adds a new column with the classification result (whether to drop the category-conversation pair; i.e., whether the conversation is judged to be low-quality) to the dataframe in-place. Returns the added column name.

    The quality classification is performed by an ensemble of prompts (see ``qual_general/prompts.py``)

    Also stores intermediate prompts/model predictions in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing conversations to be classified. The expected columns depend on the prompt used, but, the current prompts just use ``instance_text`` and ``instance_text_human_only``, produced by ``ee_retrieval.utils.data.convo_to_str`` and ``ee_retrieval.utils.data.convo_to_str_human_only``. The input may be pairs of conversations and categories (as long as the conversation column exists) as redundant predictions are handled by the cache.
        output_col (str, optional): Name of the column to store the predictions. Defaults to ``is_lq``.
        key (str): Quality classification method to use (see: ``qual_general/prompts.py``).

    Returns:
        str: Name of the column containing the final boolean is-low-quality prediction.
    """

    qual_prompts = GENERAL_PROMPT_MAP[key]

    qual_cols = []
    for p in qual_prompts:
        qual_cols.append(apply_qual_general(
            df,
            p,
            'instance_text_human_only',
            trial=1
        ))

    df[output_col] = reduce(
        lambda x, y: x | y,
        (
            df[c]
            for c in qual_cols
        )
    )

    return output_col