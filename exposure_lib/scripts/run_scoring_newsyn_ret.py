import argparse
from pathlib import Path

import pandas as pd

from exposure_lib.scoring_newsyn_ret import NewsynRetArgs, run_newsyn_ret_scoring
from exposure_lib.utils import (
    add_response_column_from_convo,
    read_llm_args_json,
    read_pickle_gzip,
    save_score_count_plot,
)


def _search_result_dict_to_newsyn_input_df(search_result_obj: object) -> pd.DataFrame:
    if not isinstance(search_result_obj, dict):
        raise ValueError(
            "Input must be a dict with DataFrame values "
            "(e.g., output from synth_gen/scripts/run_search.py)."
        )
    if "verified_df" not in search_result_obj:
        raise ValueError(
            "Input dict is missing key 'verified_df'. "
            f"Found keys: {list(search_result_obj.keys())}"
        )

    verified_df = search_result_obj["verified_df"]
    if not isinstance(verified_df, pd.DataFrame):
        raise ValueError("search result dict['verified_df'] must be a DataFrame")
    if verified_df.empty:
        raise ValueError("search_result['verified_df'] is empty; no passing rows to score")

    required_cols = {"category_occ", "category_task", "query"}
    missing_cols = required_cols - set(verified_df.columns)
    if missing_cols:
        raise ValueError(
            "search_result.verified_df is missing required columns for newsyn scoring: "
            f"{sorted(missing_cols)}"
        )

    out_df = verified_df.dropna(subset=["query"]).copy()
    out_df["convo_subset"] = out_df["query"].apply(
        lambda q: [{"role": "user", "content": str(q)}]
    )

    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/search_results.pkl.gz",
        help="Path to dict pickle output by synth_gen/scripts/run_search.py.",
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
        "--output-path",
        default="data/scored_newsyn_ret.pkl",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--plot-score-col",
        default="beta:retrieval",
        help="Score column to plot as a count plot.",
    )
    parser.add_argument(
        "--plot-path",
        default="data/scored_newsyn_ret_countplot.png",
        help="Path to output score count plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loaded_obj = read_pickle_gzip(args.input_path)
    input_df = _search_result_dict_to_newsyn_input_df(loaded_obj)

    if args.occupations_path:
        with open(args.occupations_path, "r", encoding="utf-8") as f:
            occupations = [line.strip() for line in f if line.strip()]
        if occupations:
            input_df = input_df[input_df["category_occ"].isin(occupations)].copy()

    input_df_base = input_df
    scored_dfs: list[pd.DataFrame] = []
    for config_fn in args.response_llm_config_fns:
        config_path = Path(config_fn)
        if config_path.parent == Path("."):
            config_path = Path(args.response_llm_config_dir) / config_path

        response_llm_args = read_llm_args_json(str(config_path))
        input_df_model, response_col = add_response_column_from_convo(
            input_df_base,
            llm_args=response_llm_args,
        )
        input_df_model["response"] = input_df_model[response_col]

        scored_df = run_newsyn_ret_scoring(
            df=input_df_model,
            args=NewsynRetArgs(),
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
