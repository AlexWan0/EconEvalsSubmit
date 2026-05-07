from pathlib import Path

import pandas as pd

from exposure_lib.prompts.interview import (
    INTERVIEW_BETA_MAP,
    INTERVIEW_BETA_MAP_CAT,
    INTERVIEW_BETA_MAP_CAT10,
    PROMPT_INTERVIEW_BINARIZE,
    PROMPT_INTERVIEW_BINARIZE_CAT,
    PROMPT_INTERVIEW_BINARIZE_CAT10,
    INTERVIEW_BETA_MAP_SPAN,
    PROMPT_INTERVIEW_BINARIZE_SPAN,
    PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS,
)
from exposure_lib.scoring_newsyn_ret_int import InterviewArgs
from exposure_lib.utils import read_pickle_gzip
from dataclasses import replace


def _select_prompt(version: str) -> list[dict]:
    """
    which exposure prompt to use
    """
    if version == "v1":
        from exposure_lib.prompts.interview import INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE

        return INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE

    if version == "v2":
        from exposure_lib.prompts.interview_2 import INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE

        return INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE

    if version == 'v2_span':
        from exposure_lib.prompts.interview_2 import INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN

        return INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN
    
    if version == 'v3_span':
        from exposure_lib.prompts.interview_3 import INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN

        return INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN

    if version == 'v4_span':
        from exposure_lib.prompts.interview_3 import INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN

        return INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN

    raise ValueError(f"Unsupported prompt version: {version}")


def _apply_prompt_version_setting(args, prompt_version):
    """
    Apply scoring behavior that is tied to a specific prompt version.

    v4_span reuses the v3 interview but adds an activity-to-task matching LLM
    preprocessing step before the final categorization prompt.
    """
    if prompt_version == "v4_span":
        args.result_processor_prompt_template = PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS
        args.result_processor_llm_args = replace(
            args.interview_llm_args,
            model_name='openai/gpt-5.4-mini@reasoning_effort=low',
            max_tokens=100_000
        )
        args.score_map = INTERVIEW_BETA_MAP_SPAN
        args.use_manual_activity_task_scoring = True
        args.output_col = args.manual_score_col
    elif prompt_version in {"v1", "v2", "v2_span", "v3_span"}:
        args.result_processor_prompt_template = ""
        args.use_manual_activity_task_scoring = False
    else:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")

    return args


def _load_interview_input(input_path: str) -> pd.DataFrame:
    loaded_obj: object
    tried_pickle_gzip = False
    try:
        loaded_obj = read_pickle_gzip(input_path)
        tried_pickle_gzip = True
    except Exception:
        loaded_obj = pd.read_pickle(input_path)

    if not isinstance(loaded_obj, dict):
        raise ValueError(
            "Input must be a pandas DataFrame or a dict-like object with 'verified_df' key "
            f"(pickle gzip success: {tried_pickle_gzip}, path={input_path})."
        )

    if "verified_df" not in loaded_obj:
        raise ValueError(
            f"Input dict is missing key 'verified_df'. Found keys: {list(loaded_obj.keys())}"
        )

    verified_df = loaded_obj["verified_df"]
    if not isinstance(verified_df, pd.DataFrame):
        raise ValueError("Input dict key 'verified_df' must be a DataFrame")

    return verified_df


def _resolve_model_config_path(fn: str, model_config_dir: str) -> str:
    config_path = Path(fn)
    if config_path.parent == Path("."):
        config_path = Path(model_config_dir) / config_path
    return str(config_path)


def _load_llm_args_list(filenames: list[str], model_config_dir: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fn in filenames:
        config_path = _resolve_model_config_path(fn, model_config_dir)
        out.append((fn, config_path))
    return out


def _ensure_query_convo_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "convo_subset" not in df.columns:
        out = df.copy()
        out["convo_subset"] = out["query"].apply(
            lambda q: [{"role": "user", "content": str(q)}]
        )
        return out
    return df.copy()


def _apply_occupations_filter(
    input_df: pd.DataFrame,
    occupations_path: str,
) -> pd.DataFrame:
    if not occupations_path:
        return input_df

    with open(occupations_path, "r", encoding="utf-8") as f:
        occupations = [line.strip() for line in f if line.strip()]

    if not occupations:
        return input_df

    return input_df[input_df["category_occ"].isin(occupations)].copy()


def _apply_categorize_setting(
    args: InterviewArgs,
    categorize_setting: str,
) -> InterviewArgs:
    """
    how to categorize semi-free-text exposure responses to categories
    """
    if categorize_setting == "binary":
        args.categorization_prompt_template = PROMPT_INTERVIEW_BINARIZE
        args.score_map = INTERVIEW_BETA_MAP
    elif categorize_setting == "cat5":
        args.categorization_prompt_template = PROMPT_INTERVIEW_BINARIZE_CAT
        args.score_map = INTERVIEW_BETA_MAP_CAT
    elif categorize_setting == "cat10":
        args.categorization_prompt_template = PROMPT_INTERVIEW_BINARIZE_CAT10
        args.score_map = INTERVIEW_BETA_MAP_CAT10
    elif categorize_setting == 'span5':
        args.categorization_prompt_template = PROMPT_INTERVIEW_BINARIZE_SPAN
        args.score_map = INTERVIEW_BETA_MAP_SPAN
    else:
        raise ValueError(f"Unsupported categorize setting: {categorize_setting}")

    return args
