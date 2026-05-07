from dataclasses import dataclass, field

import pandas as pd

from .filters import CoverageFilterArgs
from .prompts.new import (
    NEW_BETA_MAP,
    PROMPT_NEW_RET,
    make_new_examples_str,
    parse_new_label,
)
from .retrieval import (
    RetrievalArgs,
    build_retrieval_prompt_table,
)
from .utils import (
    LLMArgs,
    map_score_column,
    run_prompt_column,
)


@dataclass
class NewRetArgs:
    # input args
    other_cols: frozenset[str] = frozenset()

    # method args
    prompt: str = PROMPT_NEW_RET
    retrieval_args: RetrievalArgs = field(
        default_factory=lambda: RetrievalArgs(
            use_task_retrieval=True,
            use_dwa_retrieval=False,
            coverage_args=CoverageFilterArgs(),
        )
    )
    prompt_zs: str | None = None # fallback zero-shot prompt for rows with no retrieval results
    
    max_examples: int = 10
    tfidf_k: int = 5
    sample_seed: int = 0
    use_tfidf_diversity: bool = True

    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=128,
            cache_flag="validity_scoring",
        )
    )

    # output args
    output_col: str = "exposure:retrieval"
    beta_col: str = "beta:retrieval"

    # misc args
    prompt_col: str = "score_prompt"
    method_col: str = "score_method"


def build_new_ret_prompt_table(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    args: NewRetArgs,
    task_dwa_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_retrieval_prompt_table(
        retrieved_df,
        task_space_df,
        retrieval_args=args.retrieval_args,
        prompt_ret=args.prompt,
        make_examples_str=make_new_examples_str,
        prompt_col=args.prompt_col,
        method_col=args.method_col,
        max_examples=args.max_examples,
        sample_seed=args.sample_seed,
        use_tfidf_diversity=args.use_tfidf_diversity,
        tfidf_k=args.tfidf_k,
        prompt_zs=args.prompt_zs,
        task_dwa_df=task_dwa_df,
        drop_missing_prompt=True,
        other_cols=args.other_cols,
    )


def run_new_ret_scoring(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    args: NewRetArgs,
    task_dwa_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prompt_df = build_new_ret_prompt_table(
        retrieved_df,
        task_space_df,
        args,
        task_dwa_df=task_dwa_df,
    )

    scored_df = run_prompt_column(
        prompt_df,
        prompt_col=args.prompt_col,
        llm_args=args.llm_args,
        output_col=args.output_col,
        parser=parse_new_label,
        pbar_name="Scoring new-ret prompts ({model_name})",
    )

    scored_df = map_score_column(
        scored_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=NEW_BETA_MAP,
    )

    return scored_df
