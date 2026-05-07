import pandas as pd
from typing import Callable, Literal
import logging
logger = logging.getLogger(__name__)

from ..utils import extract_tag, apply_prompt


CATEGORY_PROMPT = """
I have this description of a task for the occupation of "{category_occ}" that's a bit vague: "{category_task}". I also have this list of other task descriptions that correspond to this occupation:
```
{task_lst_ft_str}
```

Minimize the overlap of the task description "{category_task}" (for the occupation "{category_occ}") with the other task descriptions by adding two additional sentences of detail to the task description that describes what this task does *not* do. Before you give your final answer, first spend three sentences planning and reasoning about your answer. Part of your planning must include contrasting and summarizing the other tasks. Answer with the following format (make sure "This does not include" is on a new line!):
```
<thinking>Planning and reasoning about your answer for three sentences.</thinking>
<answer>
[original task description]
This does not include [answer]. Instead, [continue your answer].
</answer>
```
""".strip()

def _remove_self(row: pd.Series) -> list:
    lst = row['task_lst']
    key = row['Task']

    idx = lst.index(key)

    return [
        x
        for i, x in enumerate(lst)
        if i != idx
    ]

def _get_map(dwa_df: pd.DataFrame) -> pd.DataFrame:
    logger.info('making new contrastcategory map')

    dwa_df = dwa_df[['Title', 'Task']]

    dwa_df = dwa_df.drop_duplicates().copy()

    dwa_df['task_lst'] = dwa_df['Title'].map(dwa_df.groupby('Title').agg({
        'Task': list
    })['Task'])

    dwa_df['task_lst_ft'] = dwa_df.apply(_remove_self, axis=1)

    return dwa_df

def _parse_category_contrast(text: str | None, expect_prefix: str = 'This does not include') -> str | None:
    answer = extract_tag(text, tag='answer')

    if answer is None:
        return None
    
    answer = answer.strip()
    contrast_sent_spl = answer.split('\n')
    if len(contrast_sent_spl) < 2:
        return None
    
    contrast_sent = ' '.join([x.strip() for x in contrast_sent_spl[1:]])

    if not contrast_sent.lower().startswith(expect_prefix.lower().strip()):
        return None

    return contrast_sent

def _apply_category_contrast(
        df: pd.DataFrame,
        contrast_category_map: pd.DataFrame,
        contrast_model_name: str = 'openai/gpt-4.1'
    ) -> str:
    task_lst_map: dict[tuple[str, str], list[str]] = contrast_category_map.set_index(
        ['Title', 'Task']
    )['task_lst_ft'].to_dict()

    df['task_lst_ft'] = df.apply(
        lambda row: task_lst_map[(row['category_occ'], row['category_task'])],
        axis=1
    )

    df['task_lst_ft_str'] = df['task_lst_ft'].apply(lambda x: '\n'.join(x))

    apply_prompt(
        df,
        CATEGORY_PROMPT,
        target_col=f'task_category_contrast',
        model_name=contrast_model_name,
        parser=_parse_category_contrast,
        num_workers=16,
        prefill_cache=False
    )

    df['category_task_detailed'] = df['category_task'].str.strip() + ' ' + df[f'task_category_contrast']

    return 'category_task_detailed'

def _make_task_summ_map_v1(
        dwa_df: pd.DataFrame
    ) -> pd.DataFrame:

    contrast_category_map = _get_map(dwa_df)

    logger.info('running contrast category')

    output_df = dwa_df[['Title', 'Task']].rename(columns={
        'Title': 'category_occ',
        'Task': 'category_task'
    })

    _apply_category_contrast(
        output_df,
        contrast_category_map
    )
    
    return output_df[['category_occ', 'category_task', 'category_task_detailed']]

#: A mapping from versions to detailed task generation methods
TASK_SUMM_METHODS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'v1': _make_task_summ_map_v1
}

def get_task_summ_map(dwa_df: pd.DataFrame, method: Literal['v1'] = 'v1') -> pd.DataFrame:
    """Produces more detailed task statement descriptions for each occupation by summarizing information from other tasks that that occupation performs. Specifically, given a task occupation & description, it contrasts that task description with other task descriptions that fall under the occupation to produce a more detailed task description.

    Wrapper around several summarization methods (but, currently, only one).

    Args:
        dwa_df (pd.DataFrame): Input dataframe containing DWA & adjacent task statements/occupations (e.g., ``Tasks to DWAs 29.2.xlsx`` in the O*NET data). Expects columns: 'Title', 'Task', 'DWA Title'.
        method (str): The summarization method to use.

    Returns:
        pd.DataFrame: dataframe with columns 'category_occ', 'category_task', 'category_task_detailed' ('category_task_detailed' is the newly added column containing the more detailed task descriptions).
    """

    return TASK_SUMM_METHODS[method](dwa_df)
