import argparse
from pathlib import Path

import pandas as pd

from exposure_lib.scoring_orig_ret import OrigRetArgs, run_orig_ret_scoring
from exposure_lib.utils import (
    add_response_column_from_convo,
    read_llm_args_json,
    save_score_count_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved-path",
        default="data/retrieved_samples.pkl.zst",
        help="Path to retrieved samples DataFrame with category_occ/category_task/convo_subset and optional dwa.",
    )
    parser.add_argument(
        "--task-space-path",
        default="data/task_space.pkl.zst",
        help="Path to task-space DataFrame with category_occ/category_task.",
    )
    parser.add_argument(
        "--occupations-path",
        default="",
        help="Optional path to text file with one occupation title per line; if omitted, no occupation filtering is applied.",
    )
    parser.add_argument(
        "--task-dwa-path",
        default="data/task_dwa.pkl.zst",
        help="Optional path to task<->dwa mapping DataFrame with category_occ/category_task/dwa. If omitted, DWA fallback is skipped (task retrieval first, else zero-shot).",
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
        default="data/scored_orig_ret.pkl",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--plot-score-col",
        default="beta:retrieval",
        help="Score column to plot as a count plot.",
    )
    parser.add_argument(
        "--plot-path",
        default="data/scored_orig_ret_countplot.png",
        help="Path to output score count plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retrieved_df = pd.read_pickle(args.retrieved_path)
    assert isinstance(retrieved_df, pd.DataFrame), f"Expected DataFrame at {args.retrieved_path}, got {type(retrieved_df)}"
    retrieved_df_base = retrieved_df

    task_space_df = pd.read_pickle(args.task_space_path)
    assert isinstance(task_space_df, pd.DataFrame), f"Expected DataFrame at {args.task_space_path}, got {type(task_space_df)}"

    occupations: list[str] = []
    if args.occupations_path:
        with open(args.occupations_path, "r", encoding="utf-8") as f:
            occupations = [line.strip() for line in f if line.strip()]

    if occupations:
        retrieved_df = retrieved_df[retrieved_df["category_occ"].isin(occupations)].copy()
        task_space_df = task_space_df[task_space_df["category_occ"].isin(occupations)].copy()

    task_dwa_df: pd.DataFrame | None = None
    if args.task_dwa_path:
        loaded_task_dwa = pd.read_pickle(args.task_dwa_path)
        assert isinstance(loaded_task_dwa, pd.DataFrame), f"Expected DataFrame at {args.task_dwa_path}, got {type(loaded_task_dwa)}"
        task_dwa_df = loaded_task_dwa
        if occupations:
            task_dwa_df = task_dwa_df[task_dwa_df["category_occ"].isin(occupations)].copy()

    scored_dfs: list[pd.DataFrame] = []
    for config_fn in args.response_llm_config_fns:
        config_path = Path(config_fn)
        if config_path.parent == Path("."):
            config_path = Path(args.response_llm_config_dir) / config_path

        response_llm_args = read_llm_args_json(str(config_path))
        retrieved_df_model, response_col = add_response_column_from_convo(
            retrieved_df_base,
            llm_args=response_llm_args,
        )
        retrieved_df_model["response"] = retrieved_df_model[response_col]

        scored_df = run_orig_ret_scoring(
            retrieved_df=retrieved_df_model,
            task_space_df=task_space_df,
            args=OrigRetArgs(),
            task_dwa_df=task_dwa_df,
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
