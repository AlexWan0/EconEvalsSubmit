import pandas as pd
from typing import Literal
import logging
logger = logging.getLogger(__name__)

from .categorization import (
    CategorizationStepConfig,
    PROMPTS_MODULES,
    apply_pipeline,
    apply_do_keep
)


PREDS_COL = 'pred_choice'
ENS_RES_COL = 'pred_choice_res'

def apply_categorization(
        df: pd.DataFrame,
        prompts_version: str,
        run_cfg_id: int,
        keep_method: Literal['ABC', 'ABCD', 'A'] = 'ABC'
    ) -> str:
    """Categorize each row of the input dataframe (each of which corresponds to a category-conversation pair).
    
    Adds a new column with the categorization result (whether to keep the category-conversation pair; i.e., whether the conversation belongs to the DWA/IWA/task statement category) to the dataframe in-place. Returns the added column name.

    Also stores intermediate prompts/model predictions in the dataframe.

    Expected existing columns (if any of these keys are present in the prompt, then they must also exist as a column in the dataframe):
        {dwa, iwa, category_task, category_occ}: DWA titles, IWA titles, Task statements, Task occupations
        {dwa_detailed, iwa_detailed, category_task_detailed}: More detailed DWA titles, IWA titles, Task statements as specified in ``ee_retrieval/summarization`` and get_dwa_summ_map, get_iwa_summ_map, get_task_summ_map in ``ee_retrieval/onet.py``.
        {instance_text}: conversation between human and AI assistant produced using ee_retrieval.utils.data.convo_to_str
        {instance_text_human_only}: only the human turns in the conversation produced using ee_retrieval.utils.data.convo_to_str_human_only
    
    Columns produced at runtime (if any of these keys are specified, then no additional keys are needed):
        {numbered_instance_text_human_only}: only the human turns in the conversation produced using ee_retrieval.utils.data.convo_to_str_human_only
        {summ_instance_text}: summarized instance text produced by the instance summarization
    
    Args:
        df (pd.DataFrame): dataframe consisting of each category-conversation pair we'd like to categorize. The columns required depend on the prompt, as described above.
        prompts_version (str): version of the prompts to use (defined in categorization/prompt_versions).
        run_cfg_id (int): which of the below prompt configurations to use
        keep_method (int): method for mapping categorical predictions to a binary keep/drop for the category-conversation pair. Currently, all prompts are multiple choice, so the only defined mappings specify the letters that corresopnds to a keep prediction (e.g., ``ABC`` means predicted choices A, B, and C correspond to a keep). Defaults to ``"ABC"``
    
    Returns:
        str: name of column containing the boolean keep prediction
    """

    # all available prompt configurations for categorization (not all are used)
    cfgs = [
        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            1,
            prompts_version,
            classif_type='replace',
            convo_filter_reqs=True,
            convo_summarize=True
        ),

        CategorizationStepConfig(
            'google/gemini-2.0-flash',
            2,
            prompts_version,
            classif_type='replace',
            convo_filter_reqs=True,
            convo_summarize=True
        ),

        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            3,
            prompts_version,
            classif_type='skills',
            convo_filter_reqs=True,
            convo_summarize=True
        ),

        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            4,
            prompts_version,
            classif_type='direct',
            convo_filter_reqs=False,
            convo_summarize=False
        ),

        CategorizationStepConfig(
            'google/gemini-2.0-flash',
            5,
            prompts_version,
            classif_type='direct',
            convo_filter_reqs=False,
            convo_summarize=False
        ),
        
        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            6,
            prompts_version,
            classif_type='dwa',
            convo_filter_reqs=True,
            convo_summarize=True
        ),

        CategorizationStepConfig(
            'google/gemini-2.0-flash',
            7,
            prompts_version,
            classif_type='dwa',
            convo_filter_reqs=True,
            convo_summarize=True
        ),
        
        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            8,
            prompts_version,
            classif_type='dwa_2',
            convo_filter_reqs=True,
            convo_summarize=True
        ),
        
        CategorizationStepConfig(
            'openai/gpt-4.1-nano',
            9,
            prompts_version,
            classif_type='direct_iwa',
            convo_filter_reqs=False,
            convo_summarize=False
        ),

        CategorizationStepConfig(
            'openai/gpt-4.1-nano',
            10,
            prompts_version,
            classif_type='direct_dwa_2',
            convo_filter_reqs=False,
            convo_summarize=False
        ),
        
        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            11,
            prompts_version,
            classif_type='replace_occ',
            convo_filter_reqs=True,
            convo_summarize=True
        ),
        CategorizationStepConfig(
            'openai/gpt-4.1-mini',
            12,
            prompts_version,
            classif_type='dwa_occ',
            convo_filter_reqs=True,
            convo_summarize=True
        ),
    ]

    matched_cfgs = [
        c
        for c in cfgs
        if c.cfg_id == run_cfg_id
    ]
    assert len(matched_cfgs) == 1
    cfg = matched_cfgs[0]

    output_cols: list[str] = []
    output_cols.append(
        apply_pipeline(
            df,
            model_name=cfg.model_name,
            trial=cfg.cfg_id,
            prompts_version=cfg.prompts_version,
            classif_type=cfg.classif_type,
            convo_filter_reqs=cfg.convo_filter_reqs,
            convo_summarize=cfg.convo_summarize
        )
    )
    logger.info(f'{df.columns}, {output_cols}')

    df[PREDS_COL] = list(zip(*[df[c] for c in output_cols], strict=True))

    apply_do_keep(
        df,
        keep_method=keep_method,
        preds_col=PREDS_COL,
        ens_res_col=ENS_RES_COL
    )

    return ENS_RES_COL
