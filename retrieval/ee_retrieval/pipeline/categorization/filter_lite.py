import re
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from ...utils.prompt_utils import extract_tag, apply_prompt


def _parse_filter_light(raw: str | None) -> str | None:
    if raw is None:
        return None
    
    indices_lst_str = extract_tag(raw, 'answer')

    if indices_lst_str is None:
        return None
    
    if indices_lst_str.strip() == '':
        return ''

    try:
        indices = [
            int(x.strip())
            for x in indices_lst_str.split(',')
        ]
    except Exception as e:
        logger.warning(f"filter parse error: {raw, e}")
        return None
    
    return '|'.join([str(x) for x in indices])

def _split_turns(text, prefix="Human:"):
    pattern = rf'(?m)(?=^{re.escape(prefix)})'
    parts = re.split(pattern, text)

    return [p.strip() for p in parts if p.strip()]

def _unsplit_turns(turns):
    return "\n\n".join(turns)

def filter_turns(
        df: pd.DataFrame,
        filter_prompt_lite: str,
        target_col: str = 'filter-irr',
        model_name: str = 'openai/gpt-4.1-mini',
        cache_flag: str = 'trial_1'
    ):
    """Filter the user requests in each row of the dataframe.

    Turns are first numbered. Filtering is done using an LM using the given prompt.
    
    Args:
        df (pd.DataFrame): Dataframe where each row contains a conversation (and possibly also other keys depending on the prompt). Expects at least columns: ``instance_text_human_only`` as produced by ``ee_retrieval.utils.prompt_utils.data.convo_to_str_human_only``.
        filter_prompt_lite (str): Prompt to use to perform formatting. Model output must be a list of turn indices to exclude delineated by ``|``.
        target_col (str): Output column of the filtered conversation
        model_name (str): Model to use to do the filtering
        cache_flag (str): Cache flag which gets used in the cache key for LM inference
    """

    logger.info('running filter_turns')

    df['_filter_inst_human_only'] = df['instance_text_human_only'].apply(
        lambda full: _unsplit_turns([
            f'({i}) {turn}'
            for i, turn in enumerate(_split_turns(full))
        ])
    )

    apply_prompt(
        df,
        filter_prompt_lite.replace('numbered_instance_text_human_only', '_filter_inst_human_only'),
        target_col='_filter_indices',
        model_name=model_name,
        temperature=0.5,
        max_tokens=4096,
        parser=_parse_filter_light,
        num_workers=64,
        cache_flag=cache_flag,
        prefill_cache=True
    )

    df['_filter_indices_lst'] = df['_filter_indices'].apply(
        lambda indices_str: [
            int(excl_idx)
            for excl_idx in indices_str.split('|')
        ] if (indices_str is not None and indices_str != '') else []
    )

    df[target_col] = df.apply(
        lambda row: _unsplit_turns([
            turn
            for i, turn in enumerate(_split_turns(row['instance_text_human_only']))
            if i not in row['_filter_indices_lst']
        ]),
        axis=1
    )
