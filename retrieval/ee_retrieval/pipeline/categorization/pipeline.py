import pandas as pd
from dataclasses import dataclass
from typing import Literal
import logging
logger = logging.getLogger(__name__)

from ...utils import extract_tag, extract_answer, apply_prompt
from .filter_lite import filter_turns

from .prompt_versions import PROMPTS_MODULES, get_prompts


def apply_pipeline(
        df: pd.DataFrame,
        prompts_version: str,
        model_name: str = 'openai/gpt-4.1-mini',
        trial: int = 1,
        classif_type: str = 'replace',

        # convo preprocessing steps
        convo_filter_reqs: bool = False,
        convo_summarize: bool = True
    ) -> str:
    """Runs the categorization pipeline to each row of a dataframe.

    Expected existing columns (if any of these keys are present in the prompt, then they must also exist as a column in the dataframe):
        {dwa, iwa, category_task, category_occ}: DWA titles, IWA titles, Task statements, Task occupations
        {dwa_detailed, iwa_detailed, category_task_detailed}: More detailed DWA titles, IWA titles, Task statements as specified in ``ee_retrieval/summarization`` and get_dwa_summ_map, get_iwa_summ_map, get_task_summ_map in ``ee_retrieval/onet.py``.
        {instance_text}: conversation between human and AI assistant produced using ee_retrieval.utils.data.convo_to_str
        {instance_text_human_only}: only the human turns in the conversation produced using ee_retrieval.utils.data.convo_to_str_human_only
    
    Columns produced at runtime (if any of these keys are specified, then no additional keys are needed):
        {numbered_instance_text_human_only}: only the human turns in the conversation produced using ee_retrieval.utils.data.convo_to_str_human_only
        {summ_instance_text}: summarized instance text produced by the instance summarization
    
    Args:
        df (pd.DataFrame): Dataframe containing conversation-category pairs; requires columns depending on prompts (as described above)
        prompts_versions (str): Version of prompts to use as determined by the suffix of the filename (see ``./prompt_versions/__init__.py``). e.g., ``prompts_v5.py`` has id ``v5``.
        model_name (str): LM to use for categorization
        trial (int): Trial id to use; gets incorporated into the cache key and the output column names
        classif_type (str): Categorization method to use; see: ``.__init__.py``
        convo_filter_reqs (bool): Use an LM to filter user requests first (see: ``.filter_lite.py``)
        convo_summarize (bool): Summarize the user requests first with an LM.
    """

    # setup some args
    num_workers = 32 if 'google' in model_name else 256
    cache_flag = f'trial_{trial}'

    # get the actual prompts given the version
    prompts = get_prompts(classif_type, prompts_version)
    classif_prompt = prompts.classif_prompt
    summarize_prompt = prompts.summarize_prompt
    summarize_parser = prompts.summarize_parser

    # maybe preprocess convos: filter user requests
    if convo_filter_reqs:
        logger.info('running filter reqs')
        filter_turns(
            df,
            PROMPTS_MODULES[prompts_version].FILTER_PROMPT_LIGHT,
            target_col=f't{trial}-filter_req',
            model_name='openai/gpt-4.1-mini',
            cache_flag='1'
        )
        
        summarize_prompt = summarize_prompt.replace('instance_text_human_only', f't{trial}-filter_req')

    # maybe preprocess convos: summarize conversation (instance) text & update prompt to use the summary
    if convo_summarize:
        logger.info('running summarize instance text')
        apply_prompt(
            df,
            summarize_prompt,
            target_col=f't{trial}-summ_instance_text',
            model_name=model_name,
            cache_flag=cache_flag,
            num_workers=num_workers,
            parser=summarize_parser,
            prefill_cache=True
        )

    # perform the actual classification
    logger.info('running classification')
    output_col = f't{trial}-pred_choice'

    classif_prompt = classif_prompt.replace('summ_instance_text', f't{trial}-summ_instance_text')

    apply_prompt(
        df,
        classif_prompt,
        target_col=output_col,
        parser=lambda x: extract_answer(x, choices=['A', 'B', 'C', 'D', 'E', 'F']),
        model_name=model_name,
        cache_flag=cache_flag,
        num_workers=num_workers,
        prefill_cache=True
    )

    df[f't{trial}-thinking'] = df[f'_t{trial}-pred_choice_model_output'].apply(lambda x: extract_tag(x.output, 'thinking'))

    return output_col

from dataclasses import dataclass
from typing import Literal

@dataclass
class CategorizationStepConfig:
    """Configuration for a single categorization step.

    This configuration defines parameters for a categorization pipeline.
    Flags such as `convo_filter_reqs` and `convo_summarize` control
    whether conversation preprocessing steps are enabled.

    Args:
        model_name (str): LM identifier to use.
        cfg_id (int): Unique ID for this pipeline step.
        prompts_version (str): Version of the prompts to use.
        classif_type (Literal["replace", "skills", "direct", "dwa", "dwa_2", "direct_iwa", "direct_dwa_2", "replace_occ", "dwa_occ",]): Categorization method to use. This determines which prompts are selected from the `prompts_v*.py` files. Certain classification methods expect specific columns in the input DataFrame to be present (e.g., DWA-based classification requires DWA-related columns).
        convo_filter_reqs (bool): If True, filters the conversations to only include those relevant to the category before classification.
        convo_summarize (bool): If True, summarizes conversation instances before classification.
    """

    model_name: str
    cfg_id: int
    prompts_version: str
    classif_type: Literal[
        "replace",
        "skills",
        "direct",
        "dwa",
        "dwa_2",
        "direct_iwa",
        "direct_dwa_2",
        "replace_occ",
        "dwa_occ",
    ] = "replace"
    convo_filter_reqs: bool = True
    convo_summarize: bool = True
