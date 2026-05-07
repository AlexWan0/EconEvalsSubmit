import pandas as pd
from fast_openai import run_auto_str, RequestArgs
import re
from typing import Callable


def summarize_apply_prompt(
        df: pd.DataFrame,
        prompt: str,
        target_col: str,
        model_name: str = 'openai/gpt-4.1-mini',
        temperature: float = 0.5,
        max_tokens: int = 4096,
        parser: Callable[[str], str | None] = lambda x: x,
        cache_flag: str = 'trial_1',
        num_workers: int = 64
    ):
    """Gets LM outputs for a prompt across all rows of a dataframe using the dataframe columns as keys for the prompt.

    For example, you may have a dataframe (`df`) with column "english_text", a `prompt` "Translate {english_text} into Spanish.", and a `target_col` of "spanish_text". This uses the `fast_openai` package to place the model translations into the column `spanish_text`. This transforms dataframes in-place.

    Args:
        df (pd.DataFrame): Input dataframe. Must contain all columns referenced
            in `prompt`. The column named by `target_col` will be created or
            overwritten. The dataframe may have extra columns not used by the prompt.
        prompt (str): A Python format string whose placeholders correspond to
            column names in `df` (e.g., ``"Translate {english_text} into Spanish."``).
        target_col (str): Name of the dataframe column to write parsed model
            outputs to.
        model_name (str, optional): Full model identifier passed through to the
            underlying client. All OpenAI models can be specified through
            `openai/{openai_model_name}`. Also supports `google/...`, `together/...`. Defaults to ``'openai/gpt-4.1-mini'``.
        temperature (float, optional): Sampling temperature for generation.
            Defaults to ``0.5``.
        max_tokens (int, optional): Maximum number of tokens for each response.
            Defaults to ``4096``.
        parser (Callable[[str], str | None], optional): Function that converts a
            raw model output string into the final value stored in `target_col`.
            Should return ``None`` for invalid outputs.
            Defaults to ``lambda x: x``.
        cache_flag (str, optional): By default, caches all model inputs. The cache key is a hash of the model input, the request arguments (e.g., temperature), and the cache_flag. This is important if you want to e.g., sample n times from a model but cache each of the n trials.
            Defaults to ``'trial_1'``.
        num_workers (int, optional): Maximum concurrent workers used for requests.
            Defaults to ``64``.

    Returns:
        None: Operates **in-place** on `df`. Adds/overwrites:
            - ``f'_{target_col}_model_input'``: the formatted prompts per row.
            - ``f'_{target_col}_model_output'``: raw response objects.
            - ``target_col``: parsed outputs per row.
    """

    df[f'_{target_col}_model_input'] = df.apply(
        lambda row: prompt.format(**row),
        axis=1
    )

    df[f'_{target_col}_model_output'] = run_auto_str(
        df[f'_{target_col}_model_input'].tolist(),
        full_model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        num_workers=num_workers,
        cache_flag=cache_flag,
        validation_function=lambda x: (parser(x) is not None),
        request_args=RequestArgs(
            hash_keys=True,
            num_retries=5
        ),
        pbar_name=target_col + '; {model_name}'
    )

    df[target_col] = df[f'_{target_col}_model_output'].apply(
        lambda x: parser(x.output)
    )
