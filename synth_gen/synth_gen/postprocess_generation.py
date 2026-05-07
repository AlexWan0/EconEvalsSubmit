from dataclasses import dataclass, field

import pandas as pd

from fast_openai import run_auto_str

from .prompts.misc import POSTPROCESS_QUERY_PROMPT
from .utils import LLMArgs


@dataclass
class QueryPostArgs:
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-nano@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="postprocess_query",
        )
    )
    prompt: str = POSTPROCESS_QUERY_PROMPT


def postprocess_queries(
    df: pd.DataFrame,
    args: QueryPostArgs,
    query_col: str = "query",
) -> pd.DataFrame:
    """
    Postprocess generated queries with an LM.
    """
    out_df = df.copy()
    out_df[f"_{query_col}_postprocess_prompt"] = out_df[query_col].apply(
        lambda q: args.prompt.format(query=str(q).strip())
    )
    out_df[f"_{query_col}_postprocess_output"] = run_auto_str(
        out_df[f"_{query_col}_postprocess_prompt"],
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        pbar_name="Postprocessing queries ({model_name})",
    )
    out_df[query_col] = out_df[f"_{query_col}_postprocess_output"].apply(lambda x: x.output.strip())
    return out_df
