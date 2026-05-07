from dataclasses import dataclass, field

import pandas as pd

from .prompts.newsyn import (
    NEWSYN_BETA_MAP,
    PROMPT_NEWSYN_RET,
    make_newsyn_examples_str,
    parse_newsyn_label,
)
from .utils import (
    LLMArgs,
    assert_cols,
    lint_convo_column,
    map_score_column,
    run_prompt_column,
)


@dataclass
class NewsynRetArgs:
    # input args
    other_cols: frozenset[str] = frozenset()

    # method args
    prompt: str = PROMPT_NEWSYN_RET

    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=medium",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="synthetic_scoring",
        )
    )

    # output args
    output_col: str = "exposure:retrieval"
    beta_col: str = "beta:retrieval"

    # misc args
    prompt_col: str = "score_prompt"


def build_newsyn_ret_prompt_table(
    df: pd.DataFrame,
    args: NewsynRetArgs,
) -> pd.DataFrame:
    assert_cols(df, ["category_occ", "category_task", "convo_subset", "response"])

    out_df = lint_convo_column(df, convo_col="convo_subset")
    out_df[args.prompt_col] = out_df.apply(
        lambda row: args.prompt.format(
            occupation=row["category_occ"],
            task=row["category_task"],
            examples_strs=make_newsyn_examples_str([(row["convo_subset"], row["response"])]),
            **{
                other_col: row[other_col] for other_col in args.other_cols
            }
        ),
        axis=1,
    )

    return out_df


def run_newsyn_ret_scoring(
    df: pd.DataFrame,
    args: NewsynRetArgs,
) -> pd.DataFrame:
    prompt_df = build_newsyn_ret_prompt_table(
        df,
        args,
    )

    out_df = run_prompt_column(
        prompt_df,
        prompt_col=args.prompt_col,
        llm_args=args.llm_args,
        output_col=args.output_col,
        parser=parse_newsyn_label,
        pbar_name="Scoring newsyn-ret prompts ({model_name})",
    )

    out_df = map_score_column(
        out_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=NEWSYN_BETA_MAP,
    )

    return out_df


def run_newsyn_ret_pipeline(
    df: pd.DataFrame,
    args: NewsynRetArgs,
) -> pd.DataFrame:
    return run_newsyn_ret_scoring(
        df,
        args,
    )
