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



summarize_prompt = """
# Background
You are an expert economist analyzing occupational tasks. Specifically, you want use an LLM to categorize chatbot conversations based on detailed work activities (DWAs). These DWAs are built by finding common attributes of lower-level task statements: clusters of task statements will have commonalities and differences; the commonalities will get summarized into a DWA.

However, these DWAs are a bit too vague. Add detail to the DWAs following the same procedure that was used to create them.

# Instructions
Supplement the given DWA with more detail by finding commonalities in the task statements. You should not change the meaning of the DWA: you should just add more detail using the context of the task statements. The added details must not substantially change the meaning of the DWA.

DWA: {dwa}

Task statements:
{task_statements_str}

# Output structure
First, plan out your answer in four sentences using <thinking></thinking> tags. Your planning should include finding commonalities between the task statements, finding specific details missing from the original DWA statement, and making sure that it doesn't change the original meaning.
Put your final answer in <answer></answer> tags. The first answer of your answer must be the original DWA statement exactly.
Follow this structure exactly:
<thinking>
Plan out your answer in four sentences. Your planning should include finding commonalities between the task statements, finding specific details missing from the original DWA statement, and making sure that it doesn't change the original meaning.
</thinking>
<answer>{dwa} For example, [starting in the second sentence, add more details]</answer>
""".strip()


@cached_func()
def _make_dwa_summ_map_v1(dwa_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Produces a more detailed DWA Title based on existing DWA Title & adjacent task statements.

    Expects input dataframe with columns: 'Title', 'Task', 'DWA Title'
    Produces output with cols: 'Title', 'Task', 'DWA Title', 'dwa_detailed'
    '''
    dwa_agg_df = dwa_df.groupby('DWA Title').agg({
        'Title': list,
        'Task': list,
    }).reset_index()

    # get grouped task statements for DWAs
    dwa_agg_df['cat_data'] = dwa_agg_df.apply(
        lambda row: [
            {
                'occupation': occ,
                'task': task
            }
            for occ, task in zip(row['Title'], row['Task'], strict=True)
        ],
        axis=1
    )

    dwa_agg_df['task_statements_str'] = dwa_agg_df['cat_data'].apply(
        lambda data_all: '\n'.join([
            f"- {data['task']} (Occupation: {data['occupation']})"
            for data in data_all
        ])
    )
    dwa_agg_df['dwa'] = dwa_agg_df['DWA Title']

    # run model
    summarize_apply_prompt(
        dwa_agg_df,
        prompt=summarize_prompt,
        target_col='dwa_detailed',
        model_name='openai/gpt-4.1',
        temperature=0.5,
        max_tokens=4096,
        num_workers=32,
        parser=lambda x: extract_tag(x, 'answer')
    )

    return dwa_agg_df.explode(['Title', 'Task'])[['Title', 'Task', 'DWA Title', 'dwa_detailed']].reset_index(drop=True)

does_not_include_prompt = """
Additionally, this DWA is distinct from the following set of DWAs. Any detail that you add *must not* relate to the following Unrelated DWAs:
{unrelated}
""".strip()

does_not_include_criteria = """
<criteria-four>The added details must *not* relate to any of the Unrelated DWAs. The added details must also specifically distinguish itself from the Unrelated DWAs by ending with a sentence that starts with "However, this does not include...".</criteria-four>
<criteria-five>The answer as a whole should be easily understandable. In particular, the last sentence should not seem contradictory to the previous sentences.</criteria-five>
""".strip()

does_not_include_criteria_posonly = """
<criteria-four>The added details must *not* relate to any of the Unrelated DWAs. However, the added details may only be positive: e.g., the added details must not included phrases like "this does not include". You should instead adjust the added details to be more specific.</criteria-four>
<criteria-five>The answer as a whole should be easily understandable. In particular, the last sentence should not seem contradictory to the previous sentences.</criteria-five>
""".strip()

summarize_prompt_contrast = """
# Background
You are an expert economist analyzing occupational tasks. Specifically, you want to use an LLM to categorize chatbot conversations based on detailed work activities (DWAs). These DWAs are built by finding common attributes of lower-level task statements: clusters of task statements will have commonalities and differences; the commonalities are then summarized into a DWA.

However, these DWAs are a bit too vague. Add detail to the DWAs following the same procedure that was used to create them.

# Instructions
Supplement the given DWA with more detail by finding commonalities in the task statements. You should not change the meaning of the DWA: you should just add more detail using the context of the task statements. The added details must not substantially change the meaning of the DWA.

DWA: {dwa}{does_not_include_prompt}

Task statements:
{task_statements_str}

# Output structure
First, plan out your answer using <thinking></thinking> tags, then place only your final answer in <answer></answer> tags. Your answer must satisfy the following criteria:
<criteria-one>It must have the structure "{dwa} For example, [starting in the second sentence, add more details]". The answer as a whole must be 2-3 sentences long.</criteria-one>
<criteria-two>The added details must be specific to {dwa} *only*. It must *not* just summarize every aspect of the task statements: it must only include the aspects that are specific to {dwa}.</criteria-two>
<criteria-three>The added details must be common to *all* task statements. It must *not* include any details that are specific to a particular or only a subset of the task statements.</criteria-three>{does_not_include_criteria}

Follow this structure exactly for your planning and final answer:
<thinking>
    <commonalities>
        [in two sentences, identify the commonalities between the task statements]
    </commonalities>
    <excluded>
        [in two sentences, identify things that should not be included in the answer]
    </excluded>
    <candidate-1>
        <candidate-answer>[first candidate answer]</candidate-answer>
        <candidate-rating>
            <criteria-one>Because [brief explanation], I would rate this [1-5].</criteria-one>
            <criteria-two>Because [brief explanation], I would rate this [1-5].</criteria-two>
            ...
        </candidate-rating>
    </candidate-1>
    [...]
    <candidate-3>
        <candidate-answer>[third candidate answer]</candidate-answer>
        <candidate-rating>
            <criteria-one>Because [brief explanation], I would rate this [1-5].</criteria-one>
            <criteria-two>Because [brief explanation], I would rate this [1-5].</criteria-two>
            ...
        </candidate-rating>
    </candidate-3>
    <decision>Because [brief explanation based on the ratings], the best candidate is [chosen candidate].</decision>
</thinking>
<answer>[chosen answer, following the format "{dwa} For example, [starting in the second sentence, add more details]"]</answer>
""".strip()

@cached_func()
def _make_dwa_summ_map_v2(
        dwa_df: pd.DataFrame,
        try_contrast: bool = True,
        dnic_prompt: str = does_not_include_criteria
    ) -> pd.DataFrame:
    '''
    Produces a more detailed DWA Title based on existing DWA Title & adjacent task statements.
    Contrasts DWAs w/ adjacent DWAs (if try_constrast is set to True).

    Expects input dataframe with columns: 'Title', 'Task', 'DWA Title'
    Produces output with cols: 'Title', 'Task', 'DWA Title', 'dwa_detailed'
    '''

    # connect dwas to adjacent dwas
    task_to_dwa_df = dwa_df.groupby(['Title', 'Task']).agg({
        'DWA Title': list,
    }).reset_index()

    dwa_df_other = dwa_df.merge(
        task_to_dwa_df,
        on=['Title', 'Task'],
        suffixes=('', ' Other')
    )

    dwa_df_other['DWA Title Other'] = dwa_df_other.apply(
        lambda row: [
            t
            for t in row['DWA Title Other']
            if t != row['DWA Title']
        ],
        axis=1
    )

    # connect dwas to adjacent tasks
    dwa_agg_df = dwa_df_other.groupby('DWA Title').agg({
        'Title': list,
        'Task': list,
        'DWA Title Other': lambda x: list(set(chain(*x)))
    }).reset_index()

    dwa_agg_df['cat_data'] = dwa_agg_df.apply(
        lambda row: [
            {
                'occupation': occ,
                'task': task
            }
            for occ, task in zip(row['Title'], row['Task'], strict=True)
        ],
        axis=1
    )

    # build prompt input for tasks
    dwa_agg_df['task_statements_str'] = dwa_agg_df['cat_data'].apply(
        lambda data_all: '\n'.join([
            f"- {data['task']} (Occupation: {data['occupation']})"
            for data in data_all
        ])
    )

    # build prompt input for dwas
    dwa_agg_df['dwa'] = dwa_agg_df['DWA Title']

    if try_contrast:
        # build prompt input for adjacent dwas
        dwa_agg_df['does_not_include_prompt'] = dwa_agg_df['DWA Title Other'].apply(
            lambda x: '' if len(x) == 0 else '\n\n' + does_not_include_prompt.format(
                unrelated='\n'.join([
                    f'- {y}'
                    for y in x
                ])
            )
        )
        dwa_agg_df['does_not_include_criteria'] = dwa_agg_df['does_not_include_prompt'].apply(
            lambda x: '' if x == '' else '\n' + dnic_prompt
        )

    else:
        dwa_agg_df['does_not_include_prompt'] = ''
        dwa_agg_df['does_not_include_criteria'] = ''

    # run model
    summarize_apply_prompt(
        dwa_agg_df,
        prompt=summarize_prompt_contrast,
        target_col='dwa_detailed',
        model_name='openai/gpt-4.1-mini',
        temperature=0.5,
        max_tokens=4096,
        num_workers=128,
        parser=lambda x: extract_tag(x, 'answer')
    )

    return dwa_agg_df.explode(['Title', 'Task'])[
        ['Title', 'Task', 'DWA Title', 'dwa_detailed']
    ].reset_index(drop=True)

#: A mapping from versions to detailed DWA title generation methods
DWA_SUMM_METHODS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'v1': _make_dwa_summ_map_v1,
    'v2': _make_dwa_summ_map_v2,
    'v2_nocontrast': partial(_make_dwa_summ_map_v2, try_contrast=False),
    'v2_posonly': partial(_make_dwa_summ_map_v2, dnic_prompt=does_not_include_criteria_posonly),
}

def get_dwa_summ_map(
        dwa_df: pd.DataFrame,
        method: Literal['v1', 'v2', 'v2_nocontrast', 'v2_posonly'] = 'v2'
    ) -> pd.DataFrame:
    """Produces more detailed DWA descriptions by summarizing adjacent information from the O*NET hierarchy (the task statements).

    Wrapper around several summarization methods.

    Args:
        dwa_df (pd.DataFrame): Input dataframe containing DWA & adjacent task statements/occupations (e.g., ``Tasks to DWAs 29.2.xlsx`` in the O*NET data). Expects columns: 'Title', 'Task', 'DWA Title'.
        method (str): The summarization method to use.

    Returns:
        pd.DataFrame: dataframe with columns 'Title', 'Task', 'DWA Title', 'dwa_detailed' (`dwa_detailed` is the newly generated data based on the summaries).
    """

    return DWA_SUMM_METHODS[method](dwa_df)
