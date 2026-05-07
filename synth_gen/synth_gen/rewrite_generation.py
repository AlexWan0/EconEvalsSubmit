from dataclasses import dataclass, field

import pandas as pd

from fast_openai import run_auto
from lm_prompts import render_from_df

from .prompts.misc import RewritePrompt
from .utils import LLMArgs, assert_cols, format_turns


@dataclass
class RewriteArgs:
    turn_idx: int
    llm_args: LLMArgs = field(default_factory=lambda: LLMArgs(
        model_name="openai/gpt-4.1-mini",
        temperature=0.2,
        max_tokens=256,
        num_workers=256
    ))
    prompt: RewritePrompt = field(default_factory=RewritePrompt)


def generate_rewrite(
    df: pd.DataFrame,
    args: RewriteArgs,
    rewrite_col: str,
) -> pd.DataFrame:
    """
    For rewriting template-filled dialogue turns to be more natural.
    """
    assert_cols(df, [rewrite_col])

    df = df.copy()

    df['user_turn'] = df[rewrite_col].apply(lambda turns: turns[args.turn_idx - 1]["content"])
    df['assistant_turn'] = df[rewrite_col].apply(lambda turns: turns[args.turn_idx]["content"])

    # generate prompt
    model_inputs = list(render_from_df(args.prompt, df))

    # run LM on prompt
    model_outputs = run_auto(
        model_inputs=model_inputs, # type: ignore
        full_model_name=args.llm_args.model_name,
        temperature=args.llm_args.temperature,
        max_tokens=args.llm_args.max_tokens,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        pbar_name='Rewriting dialogue turns ({model_name})'
    )

    # store in df
    df[f'_model_input_rewrite_{rewrite_col}_{args.turn_idx}'] = model_inputs # type: ignore[assignment]
    df[f"_model_output_rewrite_{rewrite_col}_{args.turn_idx}"] = model_outputs # type: ignore[assignment]
    df[rewrite_col] = df.apply(
        lambda row: [
            (
                {
                    "role": turn["role"],
                    "content": row[f"_model_output_rewrite_{rewrite_col}_{args.turn_idx}"].output,
                }
                if i == args.turn_idx
                else turn
            )
            for i, turn in enumerate(row[rewrite_col])
        ],
        axis=1
    )

    return df
