from dataclasses import dataclass, field

import pandas as pd

from .prompts.orig import ORIG_BETA_MAP, PROMPT_ORIG_ZS, parse_orig_label
from .utils import LLMArgs, assert_cols, map_score_column, run_prompt_column


@dataclass
class OrigZSArgs:
    # method args
    prompt_zs: str = PROMPT_ORIG_ZS

    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=2048,
            num_workers=128,
            cache_flag="zs_scoring",
        )
    )

    # output args
    output_col: str = "exposure:zs"
    beta_col: str = "beta:zs"

    # misc args
    prompt_col: str = "score_prompt"
    method_col: str = "score_method"


def build_orig_zs_prompt_table(
    task_space_df: pd.DataFrame,
    args: OrigZSArgs,
) -> pd.DataFrame:
    assert_cols(task_space_df, ["category_occ", "category_task"])

    out_df = task_space_df.copy()
    out_df[args.prompt_col] = out_df.apply(
        lambda row: args.prompt_zs.format(
            occupation=row["category_occ"],
            task=row["category_task"],
        ),
        axis=1,
    )
    out_df[args.method_col] = "zs"

    return out_df


def run_orig_zs_scoring(
    task_space_df: pd.DataFrame,
    args: OrigZSArgs,
) -> pd.DataFrame:
    prompt_df = build_orig_zs_prompt_table(task_space_df, args)

    scored_df = run_prompt_column(
        prompt_df,
        prompt_col=args.prompt_col,
        llm_args=args.llm_args,
        output_col=args.output_col,
        parser=parse_orig_label,
        pbar_name="Scoring orig-zs prompts ({model_name})",
    )

    scored_df = map_score_column(
        scored_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=ORIG_BETA_MAP,
    )

    return scored_df
