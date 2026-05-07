from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from fast_openai import run_auto

from .utils import LLMArgs, assert_cols
from .prompts.verifier import (
    EXPLANATION_DEMO,
    MISSING_INFO_DEMO,
    PERFORM_PROMPT,
    PERFORM_SYSTEM_PROMPT,
    REALISM_EXPLANATION_DEMO,
    REALISM_PROMPT,
    REALISM_SYSTEM_PROMPT,
    parse_perform_output,
    parse_realism_output,
)


@dataclass
class Reference:
    col_name: str
    header: str


def _build_query_with_references(
    query: str,
    row: pd.Series,
    references: Optional[list[Reference]],
    header_prefix: str = "## ",
) -> str:
    if not references:
        return query

    query_parts = [str(query)]
    for ref in references:
        ref_text = row.get(ref.col_name)
        if ref_text is None or pd.isna(ref_text):
            continue

        ref_text_str = str(ref_text).strip()
        if not ref_text_str:
            continue

        query_parts.append(f"{header_prefix}{ref.header}\n{ref_text_str}")

    return "\n\n".join(query_parts).strip()


@dataclass
class BaseVerifierArgs:
    llm_args: LLMArgs
    system_prompt: str
    prompt: str
    answer_col: str
    pass_value: str
    pass_value_col: str
    references_col: Optional[list[Reference]] = None

    @staticmethod
    def check_row_pass(row: pd.Series, args: "BaseVerifierArgs") -> bool:
        return row[args.answer_col] == args.pass_value


@dataclass
class PerformVerifierArgs(BaseVerifierArgs):
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5.2@reasoning_effort=medium",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="verifier_perform",
        )
    )
    system_prompt: str = PERFORM_SYSTEM_PROMPT
    prompt: str = PERFORM_PROMPT
    answer_col: str = "answer_perform"
    pass_value: str = "Possible"
    pass_value_col: str = "pass_perform"
    explanation: bool = False
    predict_missing_info: bool = False


@dataclass
class RealismVerifierArgs(BaseVerifierArgs):
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5.2@reasoning_effort=medium",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            cache_flag="verifier_realism",
        )
    )
    system_prompt: str = REALISM_SYSTEM_PROMPT
    prompt: str = REALISM_PROMPT
    answer_col: str = "answer_realism"
    pass_value: str = "Realistic"
    pass_value_col: str = "pass_realism"
    explanation: bool = False


def apply_perform_prompt(
    data_df: pd.DataFrame,
    args: PerformVerifierArgs,
) -> pd.DataFrame:
    """
    Run the "possible vs impossible" verifier.
    Requires columns: category_occ, query.
    """
    expected_cols = ["category_occ", "query"]
    if args.references_col:
        expected_cols.extend([x.col_name for x in args.references_col])
    assert_cols(data_df, expected_cols)

    out_df = data_df.copy()
    out_df["_model_input_perform"] = out_df.apply(
        lambda row: [
            {
                "role": "system",
                "content": args.system_prompt.format(occupation=row["category_occ"]),
            },
            {
                "role": "user",
                "content": args.prompt.format(
                    occupation=row["category_occ"],
                    query=_build_query_with_references(
                        query=str(row["query"]),
                        row=row,
                        references=args.references_col,
                    ),
                    explanation_demo=EXPLANATION_DEMO if args.explanation else "",
                    missing_info_demo=MISSING_INFO_DEMO if args.predict_missing_info else "",
                ).strip(),
            },
        ],
        axis=1,
    )

    out_df["_model_output_perform"] = run_auto(  # type: ignore
        out_df["_model_input_perform"],
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        pbar_name="Verifier: perform ({model_name})",
    )

    out_df["_model_output_perform_str"] = out_df["_model_output_perform"].apply(
        lambda x: x.output
    )

    parsed_perform = out_df["_model_output_perform_str"].apply(
        lambda output: parse_perform_output(
            output,
            include_explanation=args.explanation,
            include_missing_info=args.predict_missing_info,
        )
    )

    out_df[args.answer_col] = parsed_perform.apply(lambda x: x.answer)

    if args.explanation:
        out_df["explanation_perform"] = parsed_perform.apply(lambda x: x.explanation)

    if args.predict_missing_info:
        out_df["missing_info_perform"] = parsed_perform.apply(lambda x: x.missing_info)

    return out_df


def apply_realism_prompt(
    data_df: pd.DataFrame,
    args: RealismVerifierArgs,
) -> pd.DataFrame:
    """
    Run the "realistic vs unrealistic" verifier.
    Requires columns: category_occ, query.
    """
    expected_cols = ["category_occ", "query"]
    if args.references_col:
        expected_cols.extend([x.col_name for x in args.references_col])
    assert_cols(data_df, expected_cols)

    out_df = data_df.copy()
    out_df["_model_input_realism"] = out_df.apply(
        lambda row: [
            {
                "role": "system",
                "content": args.system_prompt.format(occupation=row["category_occ"]),
            },
            {
                "role": "user",
                "content": args.prompt.format(
                    occupation=row["category_occ"],
                    query=_build_query_with_references(
                        query=str(row["query"]),
                        row=row,
                        references=args.references_col,
                    ),
                    explanation_demo=REALISM_EXPLANATION_DEMO if args.explanation else "",
                ).strip(),
            },
        ],
        axis=1,
    )

    out_df["_model_output_realism"] = run_auto(  # type: ignore
        out_df["_model_input_realism"],
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        pbar_name="Verifier: realism ({model_name})",
    )

    out_df["_model_output_realism_str"] = out_df["_model_output_realism"].apply(
        lambda x: x.output
    )

    parsed_realism = out_df["_model_output_realism_str"].apply(
        lambda output: parse_realism_output(
            output,
            include_explanation=args.explanation,
        )
    )

    out_df[args.answer_col] = parsed_realism.apply(lambda x: x.answer)

    if args.explanation:
        out_df["explanation_realism"] = parsed_realism.apply(lambda x: x.explanation)

    return out_df


def apply_verifier_prompt(
    data_df: pd.DataFrame,
    args: BaseVerifierArgs,
) -> pd.DataFrame:
    if isinstance(args, PerformVerifierArgs):
        return apply_perform_prompt(data_df, args=args)
    if isinstance(args, RealismVerifierArgs):
        return apply_realism_prompt(data_df, args=args)
    raise ValueError(f"Unsupported verifier args type: {type(args).__name__}")
