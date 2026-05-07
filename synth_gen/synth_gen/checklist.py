from dataclasses import dataclass, field

import pandas as pd

from fast_openai import run_auto

from .prompts.checklist import (
    CHECKLIST_GEN_PROMPT_NOMIN,
    SCORING_PROMPT,
    build_required_scoring_io,
    parse_multiple_tags,
    parse_single_tag,
    parse_yes_no_items,
)
from .utils import LLMArgs, assert_cols, llm_args_suffix


def normalize_required_items(items: list[str] | None) -> list[str]:
    if items is None:
        return []
    return [" ".join(str(x).split()).strip() for x in items if str(x).strip()]


def score_required_items(parsed_item_answers: list[bool | None] | None) -> float | None:
    if not parsed_item_answers:
        return None

    item_bools = [x is True for x in parsed_item_answers]
    return sum(item_bools) / len(item_bools)


@dataclass
class EvaluatedLLMArgs:
    llm_args: LLMArgs
    evaluated_prompt: str = "{query}"


@dataclass
class ChecklistGenArgs:
    rudimentary_args: EvaluatedLLMArgs = field(
        default_factory=lambda: EvaluatedLLMArgs(
            llm_args=LLMArgs(
                model_name="together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                temperature=1.0,
                max_tokens=32000,
                num_workers=16,
                cache_flag="checklist_rudimentary_response",
            ),
            evaluated_prompt="{query}",
        )
    )
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5.2@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="checklist_generation",
        )
    )
    prompt: str = CHECKLIST_GEN_PROMPT_NOMIN
    include_thinking: bool = True


@dataclass
class ChecklistScoreArgs:
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="checklist_scoring",
        )
    )
    prompt: str = SCORING_PROMPT


def generate_checklists(
    df: pd.DataFrame,
    args: ChecklistGenArgs,
    query_col: str = "query",
    rudimentary_output_col: str = "query_out",
    output_col: str = "checklist_gen",
) -> pd.DataFrame:
    """
    Generate per-row checklist items:
    1) Generate a rudimentary response (query_out-like column) from query_col
    2) Generate checklist required items from query + rudimentary response
    Requires columns: query_col
    """
    assert_cols(df, [query_col])

    out_df = df.copy()

    out_df[f"_model_input_{rudimentary_output_col}"] = out_df.apply(
        lambda row: [
            {
                "role": "user",
                "content": args.rudimentary_args.evaluated_prompt.format(
                    **{**row.to_dict(), "query": row[query_col]}
                ),
            }
        ],
        axis=1,
    )

    out_df[f"_model_output_{rudimentary_output_col}"] = run_auto(
        out_df[f"_model_input_{rudimentary_output_col}"],
        full_model_name=args.rudimentary_args.llm_args.model_name,
        max_tokens=args.rudimentary_args.llm_args.max_tokens,
        temperature=args.rudimentary_args.llm_args.temperature,
        num_workers=args.rudimentary_args.llm_args.num_workers,
        request_args=args.rudimentary_args.llm_args.request_args,
        cache_flag=args.rudimentary_args.llm_args.cache_flag,
        pbar_name="Generating rudimentary responses ({model_name})",
    )

    out_df[f"_model_output_str_{rudimentary_output_col}"] = out_df[
        f"_model_output_{rudimentary_output_col}"
    ].apply(lambda x: x.output)
    out_df[rudimentary_output_col] = out_df[f"_model_output_str_{rudimentary_output_col}"]

    out_df[f"_model_input_{output_col}"] = out_df.apply(
        lambda row: [
            {
                "role": "user",
                "content": args.prompt.format(
                    query=row[query_col],
                    query_out=row[rudimentary_output_col],
                ),
            }
        ],
        axis=1,
    )

    out_df[f"_model_output_{output_col}"] = run_auto(
        out_df[f"_model_input_{output_col}"],
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        pbar_name="Generating checklists ({model_name})",
    )

    out_df[f"_model_output_str_{output_col}"] = out_df[f"_model_output_{output_col}"].apply(
        lambda x: x.output
    )

    if args.include_thinking:
        out_df[f"{output_col}_thinking"] = out_df[f"_model_output_str_{output_col}"].apply(
            lambda s: parse_single_tag(s, "thinking")
        )

    out_df[f"{output_col}_itemRequired"] = out_df[f"_model_output_str_{output_col}"].apply(
        lambda s: parse_multiple_tags(s, "itemRequired")
    )

    out_df["checklist_required_items"] = out_df[f"{output_col}_itemRequired"].apply(normalize_required_items)
    out_df["checklist_required_count"] = out_df["checklist_required_items"].apply(len)

    return out_df


def score_checklist(
    df: pd.DataFrame,
    args: ChecklistScoreArgs,
    response_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Score one response column against per-row required checklist items.
    Requires columns: response_col (LM response to score), checklist_required_items (generated checklist items to score against)
    """
    assert_cols(df, [response_col, "checklist_required_items"])

    out_df = df.copy()
    out_df[["checklist_input", "checklist_output"]] = out_df["checklist_required_items"].apply(
        lambda items: pd.Series(build_required_scoring_io(items))
    )
    out_df[f"_model_input_{output_col}"] = out_df.apply(
        lambda row: [
            {
                "role": "user",
                "content": args.prompt.format(
                    checklist_input=row["checklist_input"],
                    checklist_output=row["checklist_output"],
                    query_out=row[response_col],
                ),
            }
        ],
        axis=1,
    )

    out_df[f"_model_output_{output_col}"] = run_auto(
        out_df[f"_model_input_{output_col}"],
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=f"{args.llm_args.cache_flag}_{output_col}",
        pbar_name="Scoring checklists ({model_name})",
    )

    out_df[f"_model_output_str_{output_col}"] = out_df[f"_model_output_{output_col}"].apply(
        lambda x: x.output
    )
    out_df[f"{output_col}_item"] = out_df[f"_model_output_str_{output_col}"].apply(
        parse_yes_no_items
    )

    return out_df


def run_and_score_checklist(
    df: pd.DataFrame,
    args: ChecklistScoreArgs,
    evaluated_args: EvaluatedLLMArgs,
    query_col: str = "query",
    query_output_prefix: str = "query_output",
    output_col_prefix: str = "checklist_eval",
) -> pd.DataFrame:
    """
    Run one evaluated model on query_col, then score its output against checklist_required_items.
    Adds:
    - {query_output_prefix}-{suffix}
    - {output_col_prefix}-{suffix}_item
    - checklist_score_required-{suffix}
    """
    assert_cols(df, [query_col, "checklist_required_items"])

    suffix = llm_args_suffix(evaluated_args.llm_args)
    response_col = f"{query_output_prefix}-{suffix}"
    score_output_col = f"{output_col_prefix}-{suffix}"

    out_df = df.copy()
    out_df[f"_model_input_{response_col}"] = out_df.apply(
        lambda row: [
            {
                "role": "user",
                "content": evaluated_args.evaluated_prompt.format(
                    **{**row.to_dict(), "query": row[query_col]}
                ),
            }
        ],
        axis=1,
    )

    out_df[f"_model_output_{response_col}"] = run_auto(
        out_df[f"_model_input_{response_col}"],
        full_model_name=evaluated_args.llm_args.model_name,
        max_tokens=evaluated_args.llm_args.max_tokens,
        temperature=evaluated_args.llm_args.temperature,
        num_workers=evaluated_args.llm_args.num_workers,
        request_args=evaluated_args.llm_args.request_args,
        cache_flag=evaluated_args.llm_args.cache_flag,
        pbar_name="Generating evaluated responses ({model_name})",
    )

    out_df[f"_model_output_str_{response_col}"] = out_df[f"_model_output_{response_col}"].apply(
        lambda x: x.output
    )
    out_df[response_col] = out_df[f"_model_output_str_{response_col}"]

    out_df = score_checklist(
        out_df,
        args=args,
        response_col=response_col,
        output_col=score_output_col,
    )
    out_df[f"checklist_score_required-{suffix}"] = out_df[f"{score_output_col}_item"].apply(
        score_required_items
    )

    return out_df


def run_and_and_score_checklist(
    df: pd.DataFrame,
    args: ChecklistScoreArgs,
    evaluated_args: EvaluatedLLMArgs,
    query_col: str = "query",
    query_output_prefix: str = "query_output",
    output_col_prefix: str = "checklist_eval",
) -> pd.DataFrame:
    return run_and_score_checklist(
        df=df,
        args=args,
        evaluated_args=evaluated_args,
        query_col=query_col,
        query_output_prefix=query_output_prefix,
        output_col_prefix=output_col_prefix,
    )


def run_and_score_checklist_mult(
    df: pd.DataFrame,
    args: ChecklistScoreArgs,
    all_evaluated_llm_args: list[EvaluatedLLMArgs],
    query_col: str = "query",
    query_output_prefix: str = "query_output",
    output_col_prefix: str = "checklist_eval",
) -> pd.DataFrame:
    out_df = df.copy()
    for evaluated_args in all_evaluated_llm_args:
        out_df = run_and_score_checklist(
            out_df,
            args=args,
            evaluated_args=evaluated_args,
            query_col=query_col,
            query_output_prefix=query_output_prefix,
            output_col_prefix=output_col_prefix,
        )
    return out_df
