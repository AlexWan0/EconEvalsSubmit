import argparse

import pandas as pd

from synth_gen.postprocess_generation import QueryPostArgs
from synth_gen.prompts.query_references import (
    AttachmentsEmailPrompt,
)
from synth_gen.query_generation import (
    BackgroundGenArgs,
    LLMArgs,
    QueryGenArgs,
    RewriteArgs,
)
from synth_gen.reference_generation import ReferenceGenArgs
from synth_gen.search import SearchArgs, search_time_savings
from synth_gen.utils import save_pickle_gzip
from synth_gen.verifier import PerformVerifierArgs, RealismVerifierArgs, Reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--output-path",
        default="data/search_results_references.pkl.gz",
        help="Path to output pickle file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks_dwas_df = pd.read_csv(args.tasks_path)

    input_df = tasks_dwas_df.rename(columns={
        "Title": "category_occ",
        "Task": "category_task",
        "Task_detailed": "task_detailed",
    })[["category_occ", "category_task", "task_detailed"]].drop_duplicates().copy()

    if args.occupations_path:
        with open(args.occupations_path, "r", encoding="utf-8") as f:
            time_occupations = [line.strip() for line in f if line.strip()]

        input_df_samp = input_df[
            input_df["category_occ"].isin(time_occupations)
        ].copy()
    else:
        input_df_samp = input_df.copy()

    time_savings = [
        "90%-100%",
        "80%",
        # "70%",
        # "60%",
        # "50%",
        # "40%",
        # "30%",
        # "20%",
        # "10%",
    ]

    references_cfg = [Reference(col_name="references_text", header="References")]

    search_result = search_time_savings(
        input_df_samp=input_df_samp,
        time_savings=time_savings,
        bg_args=BackgroundGenArgs(
            llm_args=LLMArgs(
                model_name="openai/gpt-4.1-mini",
                temperature=1.0,
                max_tokens=8192,
                num_workers=256,
            ),
            num_respondents=5,
            num_loop=1,
            limit_first_k_resp=1,
        ),
        query_args=QueryGenArgs(
            llm_args=LLMArgs(
                model_name="openai/gpt-5-mini@reasoning_effort=low",
                temperature=1.0,
                max_tokens=8192,
                num_workers=256,
            ),
            rewrite_args=[
                RewriteArgs(turn_idx=2),
                RewriteArgs(turn_idx=4),
            ],
            prompt=AttachmentsEmailPrompt(),
        ),
        verifier_args=[
            PerformVerifierArgs(
                llm_args=LLMArgs(
                    model_name="openai/gpt-5.2@reasoning_effort=low",
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag="verifier_perform_refs",
                ),
                references_col=references_cfg,
            ),
            RealismVerifierArgs(
                llm_args=LLMArgs(
                    model_name="openai/gpt-5.2@reasoning_effort=low",
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag="verifier_realism_refs",
                ),
                references_col=references_cfg,
            ),
        ],
        search_args=SearchArgs(
            postprocess_query=True,
            query_post_args=QueryPostArgs(
                llm_args=LLMArgs(
                    model_name="openai/gpt-5-nano@reasoning_effort=low",
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=256,
                    cache_flag="postprocess_query_refs",
                ),
            ),
            generate_references=True,
            reference_args=ReferenceGenArgs(
                llm_args=LLMArgs(
                    model_name="openai/gpt-5-mini@reasoning_effort=low",
                    temperature=1.0,
                    max_tokens=64000,
                    num_workers=128,
                    cache_flag="reference_generation",
                )
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
