from dataclasses import dataclass, field

import pandas as pd

from .prompts.interview import (
    INTERVIEW_BETA_MAP,
    INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE,
    build_categorize_prompt,
    parse_interview_label,
    PROMPT_INTERVIEW_BINARIZE,
)
from .utils import (
    LLMArgs,
    assert_cols,
    lint_convo_column,
    map_score_column,
    run_multiturn_prompts,
    run_prompt_column,
)
from typing import Literal


@dataclass
class InterviewArgs:
    # input args
    required_cols: frozenset[str] = frozenset(
        {
            "category_occ",
            "category_task",
            "task_detailed",
            "job_title",
            "years_experience",
            "company_description",
            "query",
            "convo_subset",
            "response",
        }
    )

    # method args
    messages: list[dict] = field(default_factory=lambda: INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE)
    score_map: dict[str, float] = field(default_factory=lambda: INTERVIEW_BETA_MAP)
    interview_llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=1.0,
            max_tokens=32_000,
            num_workers=128,
            cache_flag="",
        )
    )
    # if set, will only run interview scoring prompts starting from this turn index
    # set to 'skip' to skip the interview scoring
    interview_start_from: Literal['skip'] | int | None = None

    categorization_prompt_template: str = PROMPT_INTERVIEW_BINARIZE
    categorize_llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=128,
            num_workers=128,
            cache_flag="",
        )
    )

    # output args
    output_col: str = "interview:binary"
    beta_col: str = "beta:interview"

    # misc args
    prompt_col: str = "score_prompt"


def build_interview_prompt_table(
    df: pd.DataFrame,
    args: InterviewArgs,
) -> pd.DataFrame:
    assert_cols(df, sorted(args.required_cols))

    out_df = lint_convo_column(df, convo_col="convo_subset")
    return run_multiturn_prompts(
        out_df,
        messages=args.messages,
        llm_args=args.interview_llm_args,
        pbar_name="Interview response turns ({model_name})",
    )


def run_interview_scoring(
    df: pd.DataFrame,
    args: InterviewArgs,
) -> pd.DataFrame:
    if not args.interview_start_from == 'skip':
        prompt_df = build_interview_prompt_table(df, args)
    else:
        prompt_df = df.copy()

    final_res_idx = len(args.messages) - 1

    final_res_col = f"_result_{final_res_idx}"
    if final_res_col not in prompt_df.columns:
        raise ValueError(
            f"Judge turn index {final_res_idx} has no generated result column {final_res_col}"
        )

    prompt_df[args.prompt_col] = prompt_df.apply(
        lambda row: build_categorize_prompt(
            row,
            row[final_res_col] if pd.notna(row[final_res_col]) else "",
            final_turn_template=args.categorization_prompt_template
        ),
        axis=1,
    )

    scored_df = run_prompt_column(
        prompt_df,
        prompt_col=args.prompt_col,
        llm_args=args.categorize_llm_args,
        output_col=args.output_col,
        parser=lambda x: parse_interview_label(x, allowable_entries=tuple(args.score_map.keys())),
        pbar_name="Interview categorize prompts ({model_name})",
    )

    scored_df = map_score_column(
        scored_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=args.score_map,
    )

    return scored_df
