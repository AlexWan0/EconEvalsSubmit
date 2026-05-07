import pandas as pd
import logging

logger = logging.getLogger(__name__)

from fast_openai import run_openai_struct_str, RequestArgs, run_auto_str

from .prompts import QualPromptConfig


def apply_qual_general(
        df: pd.DataFrame,
        prompt_data: QualPromptConfig,
        text_col: str,
        trial: int = 1,
    ) -> str:
    """Use LMs to classify whether the conversation in each row of the input dataframe is generally low-quality.

    Args:
        df (pd.DataFrame): input dataframe; must have column specified in ``text_col`` which specifies the conversation text that we perform classification on.
        prompt_data (QualPromptConfig): configuration of the classification
        text_col (str): column name containing the conversation text we want to classify on
        trial (int): trial_id identifies the trial for the new columns added & is incorporated into the LM cache key
    
    Returns:
        str: column added to the dataframe that contains the parsed model predictions
    """

    cache_flag = f'trial_{trial}'
    op_name = f"t{trial}-{prompt_data['name']}"

    df[f'_model_input_{op_name}'] = df[text_col].apply(
        lambda x: prompt_data['prompt'].format(input_text=x)
    )

    if prompt_data["is_struct"]:
        df[f'_model_output_{op_name}'] = run_openai_struct_str(
            model_inputs=df[f'_model_input_{op_name}'],
            **prompt_data["model_args"],
            cache_flag=cache_flag,
            request_args=RequestArgs(
                hash_keys=True,
                num_retries=4
            )
        )
    else:
        df[f'_model_output_{op_name}'] = run_auto_str(
            model_inputs=df[f'_model_input_{op_name}'],
            **prompt_data["model_args"],
            cache_flag=cache_flag,
            request_args=RequestArgs(
                hash_keys=True,
                num_retries=4
            )
        )

    df[op_name] = df[f'_model_output_{op_name}'].apply(
        lambda x: prompt_data['pred_extract'](x.output) if x.output is not None else None
    )

    return op_name

