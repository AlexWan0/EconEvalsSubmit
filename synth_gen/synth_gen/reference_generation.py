from dataclasses import dataclass, field

import pandas as pd

from fast_openai import run_auto_str
from lm_prompts import render_from_df

from .prompts.references import (
    ReferencesPrompt,
    attachments_to_reference_tags,
)
from .utils import LLMArgs, assert_cols


@dataclass
class ReferenceGenArgs:
    llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-5-nano@reasoning_effort=low",
            temperature=1.0,
            max_tokens=64000,
            num_workers=128,
            cache_flag="reference_generation",
        )
    )
    prompt: ReferencesPrompt = field(default_factory=ReferencesPrompt)


def generate_references(
    df: pd.DataFrame,
    args: ReferenceGenArgs,
    query_col: str = "query",
    attachments_col: str = "attachments_raw_xml",
    attachments_replaced_col: str = "attachments_replaced_tags",
    output_col: str = "reference_outputs",
    output_text_col: str = "references_text",
) -> pd.DataFrame:
    """
    Generate full reference text from compact attachment specs.
    """
    assert_cols(df, [query_col, attachments_col])

    out_df = df.copy()
    out_df[attachments_replaced_col] = out_df[attachments_col].apply(attachments_to_reference_tags)

    has_reference_specs = out_df[attachments_replaced_col].apply(
        lambda x: isinstance(x, str) and bool(str(x).strip())
    )

    print(f"Generating references for {has_reference_specs.sum()} out of {len(out_df)} rows.")

    input_col = f"_model_input_{output_col}"
    model_output_col = f"_model_output_{output_col}"
    model_output_str_col = f"_model_output_str_{output_col}"
    
    if has_reference_specs.sum() == 0:
        print("No reference specifications found in the DataFrame. Returning original DataFrame with new columns added but empty.")
        
        out_df[input_col] = None
        out_df[model_output_col] = None
        out_df[model_output_str_col] = None
        out_df[output_col] = None
        out_df[output_text_col] = None
        return out_df

    subset_df = out_df[has_reference_specs].copy()

    model_inputs = list(render_from_df(args.prompt, subset_df))

    model_outputs = run_auto_str(
        model_inputs,
        full_model_name=args.llm_args.model_name,
        max_tokens=args.llm_args.max_tokens,
        temperature=args.llm_args.temperature,
        num_workers=args.llm_args.num_workers,
        request_args=args.llm_args.request_args,
        cache_flag=args.llm_args.cache_flag,
        validation_function=args.prompt.validate_output,
        pbar_name="Generating references ({model_name})",
    )

    model_outputs_str = [output.output for output in model_outputs]

    model_outputs_parsed = list(map(lambda o: args.prompt.parse_output(o.output, verbose=True), model_outputs))

    res_text = list(map(
        lambda refs: "\n\n".join(refs.references) if refs else None,
        model_outputs_parsed
    ))

    # print(f"Example generated references output: {res_text[0] if len(res_text) > 0 else 'None'}")

    subset_df[input_col] = model_inputs
    subset_df[model_output_col] = model_outputs
    subset_df[model_output_str_col] = model_outputs_str
    subset_df[output_col] = model_outputs_parsed
    subset_df[output_text_col] = res_text

    out_df.loc[subset_df.index, input_col] = subset_df[input_col]
    out_df.loc[subset_df.index, model_output_col] = subset_df[model_output_col]
    out_df.loc[subset_df.index, model_output_str_col] = subset_df[model_output_str_col]
    out_df.loc[subset_df.index, output_col] = subset_df[output_col]
    out_df.loc[subset_df.index, output_text_col] = subset_df[output_text_col]

    return out_df
