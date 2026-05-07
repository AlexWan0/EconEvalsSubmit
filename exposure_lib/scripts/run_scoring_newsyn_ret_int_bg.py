import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from exposure_lib.scoring_newsyn_ret_int import InterviewArgs, run_interview_scoring
from exposure_lib.utils import (
    add_response_column_from_convo,
    assert_cols,
    LLMArgs,
    read_llm_args_json,
    save_score_count_plot,
)
from exposure_lib.scripts_utils import (
    _apply_categorize_setting,
    _apply_occupations_filter,
    _ensure_query_convo_subset,
    _load_interview_input,
    _load_llm_args_list,
    _select_prompt,
)
from synth_gen.background_generation import BackgroundGenArgs, generate_backgrounds
from synth_gen.utils import LLMArgs as SynthLLMArgs


def _generate_background_df(
    base_df: pd.DataFrame,
    num_respondents: int,
    num_loop: int,
    limit_first_k_resp: int | None,
) -> pd.DataFrame:
    assert_cols(
        base_df,
        [
            "category_occ",
            "task_detailed",
        ],
    )

    bg_llm_args = SynthLLMArgs(
        model_name='openai/gpt-4.1-mini',
        temperature=1.0,
        max_tokens=8192,
        num_workers=256,
    )

    base_df = base_df.copy()
    bg_rows = generate_backgrounds(
        base_df,
        args=BackgroundGenArgs(
            llm_args=bg_llm_args,
            num_respondents=num_respondents,
            num_loop=num_loop,
            limit_first_k_resp=limit_first_k_resp,
        ),
    )

    # explode generated backgrounds
    bg_rows = bg_rows.explode("parsed_backgrounds", ignore_index=True).copy()
    if bg_rows["parsed_backgrounds"].isna().any():
        raise ValueError(
            "Background generation failed for one or more rows: parsed_backgrounds is missing."
        )
    bg_rows = bg_rows.reset_index(drop=True)

    # add backgrounds as cols
    bg_struct = pd.DataFrame(
        bg_rows["parsed_backgrounds"].apply(asdict).tolist()
    )
    bg_rows = pd.concat(
        [bg_rows.drop(columns=["parsed_backgrounds"]).reset_index(drop=True), bg_struct],
        axis=1,
    )

    return bg_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/search_results.pkl.gz",
        help="Path to DataFrame or dict pickle (with verified_df) input.",
    )
    parser.add_argument(
        "--occupations-path",
        default="",
        help="Optional path to text file with one occupation title per line; if omitted, no occupation filtering is applied.",
    )
    parser.add_argument(
        "--response-llm-config-dir",
        default="scripts/model_configs",
        help="Directory containing response LLM config JSON files.",
    )
    parser.add_argument(
        "response_llm_config_fns",
        nargs="+",
        help="One or more response LLM config filenames (or paths).",
    )
    parser.add_argument(
        "--background-num-respondents",
        type=int,
        default=5,
        help="Number of backgrounds (respondents) to generate per input row for each background config.",
    )
    parser.add_argument(
        "--background-num-loop",
        type=int,
        default=1,
        help="Number of generation loops to run per background config.",
    )
    parser.add_argument(
        "--background-limit-first-k-resp",
        type=int,
        default=None,
        help="Optional cap on number of generated respondents to keep per row after background generation.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
    )
    parser.add_argument(
        "--interview-model-name",
        default="openai/gpt-4.1-mini",
    )
    parser.add_argument(
        "--categorize-setting",
        default="binary",
        choices=["binary", "cat5", "cat10"],
        help="Set categorization mode for interview scoring.",
    )
    parser.add_argument(
        "--output-path",
        default="data/scored_interview.pkl",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--plot-score-col",
        default="beta:interview",
        help="Score column to plot as a count plot.",
    )
    parser.add_argument(
        "--plot-path",
        default="data/scored_interview_countplot.png",
        help="Path to output score count plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_df_base = _load_interview_input(args.input_path)
    input_df_base = input_df_base[[
        "category_occ",
        "category_task",
        "task_detailed",
        "query",
    ]].copy()

    input_df_base = _apply_occupations_filter(input_df_base, args.occupations_path)

    response_cfgs = _load_llm_args_list(
        args.response_llm_config_fns,
        args.response_llm_config_dir,
    )

    scored_dfs: list[pd.DataFrame] = []
    background_df = _generate_background_df(
        input_df_base,
        num_respondents=args.background_num_respondents,
        num_loop=args.background_num_loop,
        limit_first_k_resp=args.background_limit_first_k_resp,
    )

    for _, response_cfg_path in response_cfgs:
        response_llm_args = read_llm_args_json(response_cfg_path)
        response_llm_args.cache_flag = ""

        background_df_with_response_input = _ensure_query_convo_subset(background_df)
        background_df_with_response, response_output_col = add_response_column_from_convo(
            background_df_with_response_input,
            llm_args=response_llm_args,
        )
        background_df_with_response["response"] = background_df_with_response[
            response_output_col
        ]

        interview_args = InterviewArgs(
            messages=_select_prompt(args.prompt_version),
            interview_llm_args=LLMArgs(
                model_name=args.interview_model_name,
                temperature=1.0,
                max_tokens=32_000,
                num_workers=128,
                cache_flag="",
            ),
        )
        _apply_categorize_setting(interview_args, args.categorize_setting)

        scored_df = run_interview_scoring(
            background_df_with_response,
            args=interview_args,
        )
        scored_df["response_model_name"] = response_llm_args.model_name

        scored_dfs.append(scored_df)

    scored_df = pd.concat(scored_dfs, ignore_index=True)
    scored_df.to_pickle(args.output_path)

    save_score_count_plot(
        scored_df,
        score_col=args.plot_score_col,
        output_path=args.plot_path,
        model_col="response_model_name",
    )
    print(f"Saved {len(scored_df)} rows to {args.output_path}")
    print(f"Saved count plot to {args.plot_path}")


if __name__ == "__main__":
    main()
