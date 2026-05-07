import re
from fast_openai import run_auto_str, RequestArgs
from typing import Callable
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def extract_answer(text: str | None, choices: list[str] = ['A', 'B', 'C', 'D', 'E', 'F']) -> str | None:
    """Extract a categorical answer from text enclosed in <answer> tags.

    Looks for text content inside ``<answer></answer>`` tags and tries to match it against one of the ``choices``.
    If no match is found or no valid answer tags found, then returns None.

    Args:
        text (str | None): The raw input text. If the input text is None, then returns None.
        choices (list[str], optional): List of answer categories. Defaults to
            ``['A', 'B', 'C', 'D', 'E', 'F']``.

    Returns:
        str | None: The matched category (will be one of ``choices``, exactly) if one is found; otherwise None.
    """

    if text is None:
        logger.warning("Input text is None")
        return None

    choices_norm = {
        c.lower().strip(): c
        for c in choices
    }

    match = re.search(r'<answer>(.*?)</answer>', text)
    
    if not match:
        logger.warning("No <answer> tags found.")
        logger.debug('-' * 80 + text + '-' * 80)
        return None
    
    content = match.group(1)
    cleaned = re.sub(r'[^A-Za-z]', '', content)
    cleaned = cleaned.strip().lower()
    
    if cleaned in choices_norm:
        return choices_norm[cleaned]
    else:
        logger.warning("Extracted content is not a valid answer")
        logger.debug('-' * 80 + '\n' + text + '\n' + '-' * 80)
        return None

def extract_tag(text: str | None, tag: str) -> str | None:
    """Extracts the content of an XML-style tag.

    Searches for a pair of tags with name ``tag`` (i.e., ``<{tag}></{tag}>``) and returns the content.

    Args:
        text (str | None):The raw input text. If the input text is None, then returns None.
        tag (str): The name of the tag to extract (without angle brackets).

    Returns:
        str | None: The text content enclosed within the tag if found; otherwise ``None``.
    """

    if text is None:
        return None

    match = re.search(f'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL)

    if not match:
        return None

    return match.group(1).strip()

def apply_prompt(
        df: pd.DataFrame,
        prompt: str,
        target_col: str,
        model_name: str = 'openai/gpt-4.1-mini',
        temperature: float = 0.5,
        max_tokens: int = 4096,
        parser: Callable[[str], str | None] = lambda x: x,
        cache_flag: str = 'trial_1',
        num_workers: int = 64,
        prefill_cache: bool = False
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
        prefill_cache (bool, optional): If ``True``, warms the cache by issuing
            requests for the unique model inputs before performing the actual run.
            Defaults to ``False``.

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

    if prefill_cache:
        run_auto_str(
            df[f'_{target_col}_model_input'].unique().tolist(),
            full_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            num_workers=num_workers,
            cache_flag=cache_flag,
            validation_function=lambda x: (parser(x) is not None),
            request_args=RequestArgs(
                use_cache=True,
                hash_keys=True,
                num_retries=4,
                retries_delay_start=3,
                post_timeout=5 * 60,
                total_timeout=5 * 60
            ),
            pbar_name=target_col + '; {model_name}; prefill cache'
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
            use_cache=True,
            hash_keys=True,
            num_retries=4,
            retries_delay_start=3,
            post_timeout=5 * 60,
            total_timeout=5 * 60
        ),
        pbar_name=target_col + '; {model_name}'
    )

    df[target_col] = df[f'_{target_col}_model_output'].apply(
        lambda x: parser(x.output)
    )

    logger.info(f'number na ({target_col}): {df[target_col].isna().sum()}')
