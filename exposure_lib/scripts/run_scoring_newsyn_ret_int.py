import pandas as pd
from pathlib import Path

import argparse

from exposure_lib.scoring_newsyn_ret_int import InterviewArgs, run_interview_scoring
from exposure_lib.utils import (
    assert_cols,
    add_response_column_from_convo,
    read_llm_args_json,
    read_pickle_gzip,
    save_score_count_plot,
    LLMArgs,
)
from exposure_lib.scripts_utils import (
    _apply_categorize_setting,
    _apply_occupations_filter,
    _ensure_query_convo_subset,
    _load_interview_input,
    _load_llm_args_list,
    _select_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # input args
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
    
    # model response args (LM to be scored)
    parser.add_argument(
        "--sample-size",
        default=None,
        type=int,
        help="When set, randomly sample this many rows from the tasks df before running."
    )
    parser.add_argument(
        "--sample-seed",
        default=42,
        type=int,
        help="Random seed to use when sampling rows from the tasks df."
    )
    
    # model response args (LM to be scored)
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

    # run config args
    parser.add_argument(
        "--prompt-version",
        default='v1',
    )
    parser.add_argument(
        "--interview-model-name",
        default='openai/gpt-4.1-mini',
    )
    parser.add_argument(
        "--categorize-setting",
        default="binary",
        choices=["binary", "cat5", "cat10", "span5"],
        help="Set categorization mode for interview scoring.",
    )

    # for restarting from the middle of the pipeline
    parser.add_argument(
        "--restart-output-path",
        default=None,
        help="Path to intermediate output pickle file to restart from; must be set if any `restart-*` args are set.",
    )
    parser.add_argument(
        "--restart-interview-from",
        default=None,
        help="Set the turn index to start interview scoring from, or 'skip' to skip.",
    )

    # output args
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

    if args.restart_output_path is not None:
        print(f"Restarting from intermediate output at {args.restart_output_path}")
        input_df_base = pd.read_pickle(args.restart_output_path)
        print(f"Loaded {len(input_df_base)} rows from {args.restart_output_path} with columns {input_df_base.columns}")
    
    else:
        input_df_base = _load_interview_input(args.input_path)
        input_df_base = _apply_occupations_filter(input_df_base, args.occupations_path)

        # maybe sample a subset of rows to run on
        if args.sample_size is not None:
            input_df_base = input_df_base.sample(n=args.sample_size, random_state=args.sample_seed).copy()

    # input_df_base.to_pickle('data/_sampled_input_df_base.pkl.zst')

    response_cfgs = _load_llm_args_list(
        args.response_llm_config_fns,
        args.response_llm_config_dir,
    )
    scored_dfs: list[pd.DataFrame] = []
    for _, response_cfg_path in response_cfgs:
        response_llm_args = read_llm_args_json(response_cfg_path)
        response_llm_args.cache_flag = ""

        input_df_with_convo = _ensure_query_convo_subset(input_df_base)
        response_df, response_output_col = add_response_column_from_convo(
            input_df_with_convo,
            llm_args=response_llm_args,
            skip_if_exists=True
        )
        response_df["response"] = response_df[response_output_col]

        interview_args = InterviewArgs(
            messages=_select_prompt(args.prompt_version),
            interview_llm_args=LLMArgs(
                model_name=args.interview_model_name,
                temperature=1.0,
                max_tokens=32_000,
                num_workers=128,
                cache_flag="",
            ),
            interview_start_from=args.restart_interview_from,
        )
        _apply_categorize_setting(interview_args, args.categorize_setting)

        scored_df = run_interview_scoring(
            response_df,
            args=interview_args,
        )
        scored_df["response_model_name"] = response_llm_args.model_name

        scored_dfs.append(scored_df)

    scored_df = pd.concat(scored_dfs, ignore_index=True)

    # print(scored_df[scored_df['category_task'] == 'Modify existing software to correct errors, adapt it to new hardware, or upgrade interfaces and improve performance.']['_result_15'].iloc[0])
    # print('-' * 80)
    # print(scored_df[scored_df['category_task'] == 'Signal truck driver to position truck to facilitate pouring concrete, and move chute to direct concrete on forms.']['_result_15'].iloc[0])
    
    # print('=' * 80)
    
    for j, (grp_name, group_rows) in enumerate(scored_df.groupby("category_occ")):
        if j >= 5:
            break

        for i, row in group_rows.sort_values('beta:interview', ascending=False).iterrows():
            print(f'{row["category_occ"]}\t{row["category_task"]}\t{row["beta:interview"]}')
        
        print('-' * 80)

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
