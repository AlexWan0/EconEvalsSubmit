import argparse

import pandas as pd

from synth_gen.search import search_time_savings, SearchArgs
from synth_gen.postprocess_generation import QueryPostArgs
from synth_gen.verifier import PerformVerifierArgs, RealismVerifierArgs
from synth_gen.query_generation import RewriteArgs, BackgroundGenArgs, QueryGenArgs, LLMArgs
from synth_gen.utils import save_pickle_gzip, RequestArgs

from synth_gen.prompts.query import BasicEmailPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # input args
    parser.add_argument(
        "--tasks-path",
        default="data/detailed_tasks.csv",
        help="Path to O*NET task dataframe with detailed task descriptions.",
    )
    parser.add_argument(
        "--occupations-path",
        default="data/time_data_occupations.txt",
        help="Path to text file with one occupation title per line.",
    )
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

    # output args
    parser.add_argument(
        "--output-path",
        default="data/search_results.pkl.gz",
        help="Path to output pickle file.",
    )

    # search args
    parser.add_argument(
        "--included-respondents",
        default=1,
        type=int,
        help='Number of respondents to include per task. Should be <= total-respondents.',
    )
    parser.add_argument(
        "--total-respondents",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--time-savings",
        nargs="+",
        default=[
            '90%-100%',
            '80%',
            '70%',
            '60%',
            '50%',
            '40%',
            '30%',
            '20%',
            '10%'
        ],
        help="List of time savings to search over.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # original O*NET tasks with an extra col for detailed task description
    tasks_dwas_df = pd.read_csv(args.tasks_path)

    input_df = tasks_dwas_df.rename(columns={
        'Title': 'category_occ',
        'Task': 'category_task',
        'Task_detailed': 'task_detailed'
    })[['category_occ', 'category_task', 'task_detailed']].drop_duplicates().copy()

    # limit to subset of occupations listed in the input text file
    if args.occupations_path:
        with open(args.occupations_path, "r", encoding="utf-8") as f:
            time_occupations = [line.strip() for line in f if line.strip()]
        
        input_df_samp = input_df[
            input_df['category_occ'].isin(time_occupations)
        ].copy()
    else:
        input_df_samp = input_df.copy()

    # maybe sample a subset of rows to run on
    if args.sample_size is not None:
        input_df_samp = input_df_samp.sample(n=args.sample_size, random_state=args.sample_seed).copy()

    # time savings to search over
    time_savings = args.time_savings

    assert args.included_respondents <= args.total_respondents, "Included respondents should be less than or equal to total respondents."

    search_result = search_time_savings(
        input_df_samp=input_df_samp,
        time_savings=time_savings,
        bg_args=BackgroundGenArgs(
            llm_args=LLMArgs(
                model_name='openai/gpt-4.1-mini',
                temperature=1.0,
                max_tokens=8192,
                num_workers=256,
            ),
            num_respondents=args.total_respondents,
            num_loop=1,
            limit_first_k_resp=args.included_respondents,
        ),
        query_args=QueryGenArgs(
            llm_args=LLMArgs(
                model_name='openai/gpt-5-mini@reasoning_effort=low',
                temperature=1.0,
                max_tokens=8192,
                num_workers=256,
                request_args=RequestArgs(
                    use_cache=True,
                    hash_keys=True,
                    num_retries=3,
                    post_timeout=300,
                    total_timeout=300,
                )
            ),
            rewrite_args=[
                RewriteArgs(turn_idx=2),
                RewriteArgs(turn_idx=4),
            ],
            prompt=BasicEmailPrompt(),
        ),
        verifier_args=[
            PerformVerifierArgs(
                llm_args=LLMArgs(
                    model_name='openai/gpt-5.2@reasoning_effort=low',
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag='verifier_perform',
                ),
            ),
            RealismVerifierArgs(
                llm_args=LLMArgs(
                    model_name='openai/gpt-5.2@reasoning_effort=low',
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag='verifier_realism',
                ),
            ),
        ],
        search_args=SearchArgs(
            postprocess_query=True,
            query_post_args=QueryPostArgs(
                llm_args=LLMArgs(
                    model_name='openai/gpt-5-nano@reasoning_effort=low',
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag='postprocess_query',
                ),
            ),
        ),
    )

    search_result_dict = {
        "verified_df": search_result.verified_df,
        "not_found_df": search_result.not_found_df,
        "intermediate_df": search_result.intermediate_df,
    }

    save_pickle_gzip(search_result_dict, args.output_path)


if __name__ == "__main__":
    main()
