import pandas as pd
from dataclasses import dataclass, field
from itertools import chain

from fast_openai import run_auto_str
from lm_prompts import render_from_df

from .utils import assert_cols, LLMArgs
from .prompts.backgrounds import GeneratedBackground, GeneratedBackgroundBatch, BackgroundsPrompt


@dataclass
class BackgroundGenArgs:
    llm_args: LLMArgs
    num_respondents: int = 5 # number of respondent backgrounds to generate per row
    num_loop: int = 1 # number of times to repeat the generation process
    limit_first_k_resp: int | None = None # only use the first k generated backgrounds for each row
    prompt: BackgroundsPrompt = field(default_factory=BackgroundsPrompt)

def generate_backgrounds(
    df: pd.DataFrame,
    args: BackgroundGenArgs,
) -> pd.DataFrame:
    """
    Generates (num_respondents * num_loop) backgrounds for each row in the input DataFrame.
    Requires columns for `df`: category_occ (occupation), task_detailed (task description)
    """
    required_cols = {"category_occ", "task_detailed"}
    assert_cols(df, list(required_cols))

    work_df = df.copy()
    work_df["__row_id"] = work_df.index
    work_df['num_respondents'] = args.num_respondents

    # make prompts
    background_prompts = list(render_from_df(args.prompt, work_df))

    # run prompts
    background_output_strs_by_loop: list[list[str | None]] = []
    background_output_parsed_by_loop: list[list[GeneratedBackgroundBatch | None]] = []
    for loop_idx in range(args.num_loop):
        outputs = run_auto_str(
            background_prompts,
            full_model_name=args.llm_args.model_name,
            temperature=args.llm_args.temperature,
            max_tokens=args.llm_args.max_tokens,
            num_workers=args.llm_args.num_workers,
            request_args=args.llm_args.request_args,
            cache_flag=args.llm_args.cache_flag + f"_{loop_idx}",
            validation_function=args.prompt.validate_output,
            pbar_name='Generating backgrounds ({model_name})'
        )
        background_output_strs_by_loop.append([x.output for x in outputs])
        background_output_parsed_by_loop.append([
            args.prompt.parse_output(x.output, verbose=True) for x in outputs
        ])

    # save to dataframe
    work_df['background_prompt'] = background_prompts
    
    work_df["background_output_strs"] = [
        [
            background_output_strs_by_loop[loop_idx][row_idx]
            for loop_idx in range(args.num_loop)
        ]
        for row_idx in range(len(work_df))
    ]
    
    def _unpack(batch: GeneratedBackgroundBatch | None) -> list[GeneratedBackground]:
        if batch is None:
            return []
        return batch.backgrounds

    work_df["parsed_backgrounds"] = [
        list(chain(*[
            _unpack(background_output_parsed_by_loop[loop_idx][row_idx])
            for loop_idx in range(args.num_loop)
        ]))
        for row_idx in range(len(work_df))
    ]

    if args.limit_first_k_resp is not None:
        work_df["parsed_backgrounds"] = work_df["parsed_backgrounds"].apply(
            lambda x: x[:args.limit_first_k_resp] if isinstance(x, list) else x
        )

    return work_df
