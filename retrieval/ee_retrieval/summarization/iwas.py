import pandas as pd
from fast_openai import run_auto_str, RequestArgs
from typing import Callable, Literal
import re
from itertools import chain
from functools import partial
import logging
logger = logging.getLogger(__name__)

from ..utils import cached_func, extract_tag
from .lm_utils import summarize_apply_prompt


iwa_summarize_prompt = """
# Background
You are an expert economist analyzing occupational tasks. Specifically, you want to use an LLM to categorize chatbot conversations based on intermediate work activities (IWAs). These IWAs are built by clustering and summarizing lower-level work activities. However, these IWAs are a bit too vague to be used for classification. You want to add detail to the IWAs.

# Instructions
Supplement the given IWA with more detail by summarizing the lower-level work activities. You should not change the meaning of the IWA: you should just add more detail using the context of the lower-level work activities. The added details must not substantially change the meaning of the IWA.

IWA: {iwa}

Lower-level work activities:
{dwa_lst_str}

First, plan out your answer using <thinking></thinking> tags, then place only your final answer in <answer></answer> tags. Your answer should have the following structure:
1) First, repeat the IWA exactly.
2) Second, in two sentences, provide a comprehensive summary of the lower-level work activities. Make sure *every* lower-level work activity is covered under the summary.
3) Third, in two sentences, provide examples. Similar to (2), these examples must also cover every single lower-level work activity.

# Output structure
Follow this structure for your answer exactly:
<thinking>
Because the resulting summary needs to cover every lower-level work activity, I should note down the important part of each work activity that I should include:
(1) [important part of work activity to include]
(2) [...]
[continue for *every* lower-level work activity]

I also should think about including examples that cover every lower-level work activity. [coming up with examples and making sure they cover every lower-level work activity; while you're thinking, to keep track of which task-statements you've covered, make sure to explicitly include the number corresponding to the covered work-activity (e.g., "(2)"); make sure that you've mentioned (and thus covered) every number; however do NOT include the numbers in the actual summary]
</thinking>
<answer>
[five sentence summary following the structure above; the summary should be a single paragraph with no line-breaks]
</answer>
""".strip()

@cached_func()
def _make_iwa_summ_map_v1(dwa_df: pd.DataFrame, iwa_df: pd.DataFrame, dwa_summ_map: dict[str, str]) -> pd.DataFrame:
    '''
    Produces a more detailed IWA description by summarizing the child DWAs. Uses DWAs with added detail (as defined by ``dwa_summ_map``).
    '''
    dwa_df['dwa_detailed'] = dwa_df['DWA Title'].map(
        dwa_summ_map
    )
    assert not any(dwa_df['dwa_detailed'].isna())
    
    merged_df = dwa_df.merge(
        iwa_df,
        on=['DWA Title', 'DWA ID'],
        how='left',
        validate='m:1'
    )

    merged_df['category_data'] = merged_df.apply(
        lambda row: {'occupation': row['Title'], 'task': row['Task']},
        axis=1
    )

    iwa_dwa_df = merged_df.groupby('IWA Title').agg({
        'dwa_detailed': lambda x: list(sorted(set(x))),
        'DWA Title': lambda x: list(sorted(set(x))),
    }).reset_index()

    iwa_dwa_df['dwa_lst_str'] = iwa_dwa_df['dwa_detailed'].apply(
        lambda x: '\n'.join([
            f'({i + 1}) {dwa}'
            for i, dwa in enumerate(x)
        ])
    )

    iwa_dwa_df = iwa_dwa_df.rename(columns={'IWA Title': 'iwa'})

    summarize_apply_prompt(
        iwa_dwa_df,
        prompt=iwa_summarize_prompt,
        target_col='iwa_detailed',
        temperature=0.5,
        model_name='openai/gpt-4.1',
        max_tokens=8192,
        parser=lambda x: extract_tag(x, 'answer'),
        num_workers=64
    )

    return iwa_dwa_df.explode(['dwa_detailed', 'DWA Title'])

#: A mapping from versions to detailed IWA title generation methods
IWA_SUMM_METHODS: dict[str, Callable[[pd.DataFrame, pd.DataFrame, dict[str, str]], pd.DataFrame]] = {
    'v1': _make_iwa_summ_map_v1,
}

def get_iwa_summ_map(
        dwa_df: pd.DataFrame,
        iwa_df: pd.DataFrame,
        dwa_summ_map: dict[str, str],
        method: Literal['v1'] = 'v1'
    ) -> pd.DataFrame:
    """Produces more detailed IWA descriptions by summarizing adjacent information from the O*NET hierarchy (the adjacent lower-level DWAs).

    Wrapper around several summarization methods (but, currently, only one).

    Args:
        dwa_df (pd.DataFrame): Input dataframe containing DWA & adjacent task statements/occupations (e.g., ``Tasks to DWAs 29.2.xlsx`` in the O*NET data). Expects columns: 'Title', 'Task', 'DWA Title', 'DWA ID'.
        iwa_df (pd.DataFrame): Input dataframe containing DWA & IWA connections (e.g., ``DWA Reference 29.2.xlsx`` in the O*NET data). Expects columns: 'DWA Title', 'IWA Title'
        dwa_summ_map (dict[str, str]): The mapping from the original ``DWA Title`` to the more detailed DWA (``dwa_detailed``) produced by ``get_dwa_summ_map``; this function produces a dataframe where you can find the columns ``DWA Title`` and ``dwa_detailed``.
        method (str): The summarization method to use. Defaults to ``v1`` (the only version available currently).

    Returns:
        pd.DataFrame: dataframe with columns 'DWA Title', 'iwa', and 'iwa_detailed'
    """

    return IWA_SUMM_METHODS[method](dwa_df, iwa_df, dwa_summ_map)
