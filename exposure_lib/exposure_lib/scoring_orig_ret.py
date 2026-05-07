from dataclasses import dataclass, field
from string import Formatter

import pandas as pd
from typing import Callable

from .prompts.orig import (
    ORIG_BETA_MAP,
    PROMPT_ORIG_RET,
    PROMPT_ORIG_ZS,
    make_orig_examples_str,
    parse_orig_label,
)
from .retrieval import (
    RetrievalArgs,
    build_retrieval_prompt_table,
)
from .utils import (
    LLMArgs,
    map_score_column,
    run_convo_column,
    run_prompt_column,
)

FewShotSample = tuple[dict[str, str], str]


def _get_prompt_fields(prompt_template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(prompt_template):
        if field_name is None:
            continue

        # Normalizes possible "{field.attr}" / "{field[idx]}" format keys.
        base_field = field_name.split(".", 1)[0].split("[", 1)[0]
        if base_field:
            fields.add(base_field)

    return fields


def _validate_few_shot_samples(
    few_shot_samples: list[FewShotSample],
    prompt_template: str,
) -> None:
    required_fields = _get_prompt_fields(prompt_template)
    for idx, (sample_vars, sample_output) in enumerate(few_shot_samples):
        if not isinstance(sample_vars, dict):
            raise ValueError(
                f"few_shot_samples[{idx}][0] must be a dict[str, str], "
                f"got {type(sample_vars)}"
            )
        if not isinstance(sample_output, str):
            raise ValueError(
                f"few_shot_samples[{idx}][1] must be a str, got {type(sample_output)}"
            )

        non_str_keys = [k for k in sample_vars.keys() if not isinstance(k, str)]
        if non_str_keys:
            raise ValueError(
                f"few_shot_samples[{idx}] has non-string keys: {non_str_keys}"
            )

        non_str_vals = [
            key
            for key, value in sample_vars.items()
            if not isinstance(value, str)
        ]
        if non_str_vals:
            raise ValueError(
                f"few_shot_samples[{idx}] has non-string values for keys: {non_str_vals}"
            )

        missing_fields = required_fields - set(sample_vars.keys())
        if missing_fields:
            raise ValueError(
                f"few_shot_samples[{idx}] is missing keys required by prompt template: "
                f"{sorted(missing_fields)}"
            )


def _build_few_shot_prefix(
    few_shot_samples: list[FewShotSample],
    prompt_template: str,
) -> list[dict[str, str]]:
    prefix_turns: list[dict[str, str]] = []
    for sample_vars, sample_output in few_shot_samples:
        sample_user_prompt = prompt_template.format(**sample_vars)
        prefix_turns.append({
            "role": "user",
            "content": sample_user_prompt,
        })
        prefix_turns.append({
            "role": "assistant",
            "content": sample_output,
        })
    return prefix_turns


@dataclass
class OrigRetArgs:
    # input args
    other_cols: frozenset[str] = frozenset()

    # method args
    prompt_ret: str = PROMPT_ORIG_RET
    retrieval_args: RetrievalArgs = field(
        default_factory=lambda: RetrievalArgs(
            use_task_retrieval=True,
            use_dwa_retrieval=True,
            coverage_args=None,
        )
    )
    prompt_zs: str = PROMPT_ORIG_ZS # fallback zero-shot prompt for rows with no retrieval results
    prompt_parser: Callable[[str | None], str | None] = parse_orig_label
    beta_map: dict[str, float] = field(
        default_factory=lambda: ORIG_BETA_MAP
    )

    max_examples: int = 10
    tfidf_k: int = 5
    sample_seed: int = 0
    use_tfidf_diversity: bool = False
    few_shot_samples: list[FewShotSample] = field(default_factory=list)

    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=2048,
            num_workers=128,
            cache_flag="conditioned_scoring",
        )
    )

    # output args
    output_col: str = "exposure:retrieval"
    beta_col: str = "beta:retrieval"

    # misc args
    prompt_col: str = "score_prompt"
    method_col: str = "score_method"


def build_orig_ret_prompt_table(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    args: OrigRetArgs,
    task_dwa_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_retrieval_prompt_table(
        retrieved_df,
        task_space_df,
        retrieval_args=args.retrieval_args,
        prompt_ret=args.prompt_ret,
        make_examples_str=make_orig_examples_str,
        prompt_col=args.prompt_col,
        method_col=args.method_col,
        max_examples=args.max_examples,
        sample_seed=args.sample_seed,
        use_tfidf_diversity=args.use_tfidf_diversity,
        tfidf_k=args.tfidf_k,
        prompt_zs=args.prompt_zs,
        task_dwa_df=task_dwa_df,
        drop_missing_prompt=False,
        other_cols=args.other_cols,
    )


def run_orig_ret_scoring(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    args: OrigRetArgs,
    task_dwa_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run original-taxonomy retrieval-conditioned exposure scoring.

    Note: if ``task_dwa_df`` is not provided, DWA fallback retrieval is disabled.
    Rows use task-level retrieval when available, otherwise zero-shot.
    """
    prompt_df = build_orig_ret_prompt_table(
        retrieved_df,
        task_space_df,
        args,
        task_dwa_df=task_dwa_df,
    )

    print('=' * 80)
    print(prompt_df[args.prompt_col].iloc[0])
    print('-' * 80)

    if len(args.few_shot_samples) > 0:
        _validate_few_shot_samples(
            args.few_shot_samples,
            prompt_template=args.prompt_ret,
        )
        few_shot_prefix = _build_few_shot_prefix(
            args.few_shot_samples,
            prompt_template=args.prompt_ret,
        )
        convo_col = f"{args.prompt_col}_convo"
        prompt_df[convo_col] = prompt_df[args.prompt_col].apply(
            lambda prompt: [
                *few_shot_prefix,
                {
                    "role": "user",
                    "content": str(prompt),
                },
            ]
        )

        scored_df = run_convo_column(
            prompt_df,
            convo_col=convo_col,
            llm_args=args.llm_args,
            output_col=args.output_col,
            parser=args.prompt_parser,
            pbar_name="Scoring orig-ret prompts ({model_name})",
        )
    else:
        scored_df = run_prompt_column(
            prompt_df,
            prompt_col=args.prompt_col,
            llm_args=args.llm_args,
            output_col=args.output_col,
            parser=args.prompt_parser,
            pbar_name="Scoring orig-ret prompts ({model_name})",
        )

    scored_df = map_score_column(
        scored_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=args.beta_map,
    )

    print('-' * 80)
    print(scored_df[args.output_col].iloc[0])
    print('=' * 80)

    return scored_df
