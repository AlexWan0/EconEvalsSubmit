import argparse

import pandas as pd

from exposure_lib.scoring_orig_zs import OrigZSArgs, run_orig_zs_scoring
from exposure_lib.utils import save_score_count_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--output-path",
        default="data/scored_orig_zs.pkl",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--plot-score-col",
        default="beta:zs",
        help="Score column to plot as a count plot.",
    )
    parser.add_argument(
        "--plot-path",
        default="data/scored_orig_zs_countplot.png",
        help="Path to output score count plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_space_df = pd.read_pickle(args.task_space_path)
    assert isinstance(task_space_df, pd.DataFrame), f"Expected DataFrame at {args.task_space_path}, got {type(task_space_df)}"

    if args.occupations_path:
        with open(args.occupations_path, "r", encoding="utf-8") as f:
            occupations = [line.strip() for line in f if line.strip()]
        if occupations:
            task_space_df = task_space_df[task_space_df["category_occ"].isin(occupations)].copy()

    scored_df = run_orig_zs_scoring(
        task_space_df=task_space_df,
        args=OrigZSArgs(),
    )

    scored_df.to_pickle(args.output_path)
    save_score_count_plot(
        scored_df,
        score_col=args.plot_score_col,
        output_path=args.plot_path,
    )
    print(f"Saved {len(scored_df)} rows to {args.output_path}")
    print(f"Saved count plot to {args.plot_path}")


if __name__ == "__main__":
    main()
