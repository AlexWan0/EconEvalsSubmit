from dataclasses import dataclass, field
import pandas as pd
from typing import Callable

from .prompts.summary import PROMPT_SUMMARY, parse_summary
from .utils import LLMArgs, assert_cols, map_score_column, run_prompt_column


@dataclass
class SummaryArgs:
    # method args
    prompt: str = PROMPT_SUMMARY
    parser: Callable[[str | None], str | None] = parse_summary

    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=low",
            temperature=1.0,
            max_tokens=4096,
            num_workers=128,
            cache_flag="summary",
        )
    )

    # output args
    output_col: str = "summary"

def singleton_turns(
    turns: list[dict[str, str]],
    role: str
) -> str:
    for turn in turns:
        if turn["role"] == role:
            return turn["content"]
    
    raise ValueError(f"No turn with role {role} found in turns: {turns}")


def summarize_conversations(
    convo_df: pd.DataFrame,
    args: SummaryArgs,
) -> pd.DataFrame:
    assert_cols(convo_df, ["category_occ", "category_task", "convo_subset"])

    prompt_df = convo_df.copy()
    prompt_df['single_user_request'] = prompt_df['convo_subset'].apply(lambda convo: singleton_turns(convo, role="user"))
    prompt_df[f'_summary_input_{args.output_col}'] = prompt_df.apply(
        lambda row: args.prompt.format(**row),
        axis=1,
    )

    print('=' * 80)
    print(prompt_df[f'_summary_input_{args.output_col}'].iloc[0])
    print('=' * 80)

    scored_df = run_prompt_column(
        prompt_df,
        prompt_col=f'_summary_input_{args.output_col}',
        llm_args=args.llm_args,
        output_col=args.output_col,
        parser=args.parser,
        pbar_name="Generating summaries ({model_name})",
    )

    print('=' * 80)
    print(scored_df[f"_model_output_str_{args.output_col}"].iloc[0])
    print('-' * 80)
    print(scored_df[args.output_col].iloc[0])
    print('=' * 80)

    return scored_df
