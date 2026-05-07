import argparse

from fast_openai import RequestArgs
import pandas as pd

from synth_gen.checklist import (
    ChecklistGenArgs,
    ChecklistScoreArgs,
    EvaluatedLLMArgs,
    generate_checklists,
    run_and_score_checklist_mult,
)
from synth_gen.utils import LLMArgs, read_pickle_gzip, save_pickle_gzip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/search_results.pkl.gz",
        help="Path to pickle output from run_search.py (dict with DataFrame values).",
    )
    parser.add_argument(
        "--output-path",
        default="data/checklist_results.pkl.zst",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--occupations",
        default="",
        help="Optional comma-separated occupation names to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # load data
    loaded_obj = read_pickle_gzip(args.input_path)
    assert isinstance(loaded_obj, dict), f"Expected dict object in {args.input_path}, but got {type(loaded_obj)}"
    assert "verified_df" in loaded_obj, f"Expected key 'verified_df' in {args.input_path}; got keys={list(loaded_obj.keys())}"
    data_df = loaded_obj["verified_df"]
    assert isinstance(data_df, pd.DataFrame), "Expected verified_df to be a pandas DataFrame"
    data_df = data_df.copy()

    # maybe filter by occupations
    occupations = [x.strip() for x in args.occupations.split(",") if x.strip()]
    if occupations:
        data_df = data_df[data_df["category_occ"].isin(occupations)].copy()

    default_request_args = RequestArgs(
        use_cache=True,
        hash_keys=True,
        total_timeout=300,
        post_timeout=300,
    )
    together_request_args = RequestArgs(
        use_cache=True,
        hash_keys=True,
        total_timeout=300,
        post_timeout=300,
        rate_limit_rpm=60,
        num_retries=5,
    )

    checklist_gen_args = ChecklistGenArgs(
        rudimentary_args=EvaluatedLLMArgs(
            llm_args=LLMArgs(
                model_name="together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                temperature=1.0,
                max_tokens=32000,
                num_workers=16,
                request_args=together_request_args,
                cache_flag="checklist_rudimentary_response",
            ),
            evaluated_prompt="{query}",
        ),
        llm_args=LLMArgs(
            model_name="openai/gpt-5.2@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            request_args=default_request_args,
            cache_flag="checklist_generation",
        ),
    )

    checklist_score_args = ChecklistScoreArgs(
        llm_args=LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=low",
            temperature=1.0,
            max_tokens=8192,
            num_workers=256,
            request_args=default_request_args,
            cache_flag="checklist_scoring",
        ),
    )

    all_evaluated_args = [
        EvaluatedLLMArgs(
            llm_args=LLMArgs(
                model_name=model_name,
                temperature=1.0,
                max_tokens=32000,
                num_workers=16 if model_name.startswith("together/") else 256,
                request_args=together_request_args if model_name.startswith("together/") else default_request_args,
                cache_flag="checklist_evaluated_response",
            ),
            evaluated_prompt="{query}",
        )
        for model_name in [
            "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "openai/gpt-4.1-nano",
            "openai/gpt-5-nano@reasoning_effort=low",
            "openai/gpt-5.2@reasoning_effort=low",
        ]
    ]

    data_df = generate_checklists(
        data_df,
        args=checklist_gen_args,
        query_col="query",
        rudimentary_output_col="query_out",
        output_col="checklist_gen",
    )

    data_df = run_and_score_checklist_mult(
        data_df,
        args=checklist_score_args,
        all_evaluated_llm_args=all_evaluated_args,
        query_col="query",
        query_output_prefix="query_output",
        output_col_prefix="checklist_eval",
    )

    save_pickle_gzip(data_df, args.output_path)
    print(f"Saved {len(data_df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
