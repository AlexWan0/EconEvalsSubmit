import pandas as pd
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Callable, TypeVar

from fast_openai import run_auto

from .utils import assert_cols, LLMArgs, format_turns
from .background_generation import generate_backgrounds, BackgroundGenArgs
from .rewrite_generation import RewriteArgs, generate_rewrite
from .prompts.backgrounds import GeneratedBackground
from .prompts.query import GeneratedQuery
from lm_prompts import TurnsPrompt, PromptInputT, cast_to_input, render_from_df
from typing import Generic


GeneratedQueryT = TypeVar("GeneratedQueryT", bound=GeneratedQuery)

@dataclass
class QueryGenArgs(Generic[PromptInputT, GeneratedQueryT]):
    llm_args: LLMArgs
    rewrite_args: list[RewriteArgs] # for rewriting dialogue turns to be more grammatical
    prompt: TurnsPrompt[PromptInputT, GeneratedQueryT]

def generate_queries(
    df: pd.DataFrame,
    bg_args: BackgroundGenArgs,
    query_args: QueryGenArgs[PromptInputT, GeneratedQueryT],
) -> pd.DataFrame:
    """
    Dialogue-based query generation pipeline.
    Requires columns for `df`: category_occ (occupation), task_detailed (task description)
    Other columsn may be required if referenced in the prompt template (e.g., '{time_savings}')
    """

    # check cols for background generation
    required_cols = {"category_occ", "task_detailed"}
    assert_cols(df, list(required_cols))

    # generate backgrounds for each row
    respondents_df_grp = generate_backgrounds(
        df,
        args=bg_args,
    )

    respondents_df = respondents_df_grp.explode("parsed_backgrounds", ignore_index=True)
    respondents_df = respondents_df[respondents_df["parsed_backgrounds"].notnull()].reset_index(
        drop=True
    )

    # convert parsed backgrounds to separate columns
    bg_series = respondents_df["parsed_backgrounds"].apply(asdict)
    bg_df = pd.DataFrame(bg_series.tolist())
    respondents_df = pd.concat(
        [respondents_df.drop(columns=["parsed_backgrounds"]), bg_df], axis=1
    )

    # check cols for query generation
    required_cols_query = query_args.prompt.input_field_names
    assert_cols(respondents_df, list(required_cols_query))

    # make prompts
    model_input_query_turns = list(render_from_df(query_args.prompt, respondents_df))

    # rewrite dialogue turns
    rewrite_df = respondents_df.copy()
    rewrite_df["_model_input_query_turns"] = model_input_query_turns
    for rewrite_arg in query_args.rewrite_args:
        rewrite_df = generate_rewrite(
            rewrite_df,
            args=rewrite_arg,
            rewrite_col='_model_input_query_turns',
        )
    model_input_query_turns = rewrite_df["_model_input_query_turns"].tolist()
    
    # print(model_input_query_turns[0])

    # run LM on prompts and parse outputs into typed local intermediate
    generation_raw_outputs = run_auto(
        model_input_query_turns,
        full_model_name=query_args.llm_args.model_name,
        temperature=query_args.llm_args.temperature,
        max_tokens=query_args.llm_args.max_tokens,
        num_workers=query_args.llm_args.num_workers,
        request_args=query_args.llm_args.request_args,
        cache_flag=query_args.llm_args.cache_flag,
        validation_function=query_args.prompt.validate_output,
        pbar_name='Generating queries ({model_name})'
    )

    generation_outputs = [
        query_args.prompt.parse_output(x.output, verbose=True)
        for x in generation_raw_outputs
    ]

    # print(f"Example generated query output: {generation_outputs[0].email_body if generation_outputs[0] else 'None'}")

    for col_name in query_args.prompt.output_field_names:
        respondents_df[col_name] = [
            getattr(gen_output, col_name) if gen_output else None
            for gen_output in generation_outputs
        ]

    return respondents_df
