"""
Display saved exposure scores by occupation.

Loads the scored interview DataFrame and prints each occupation with one line per
task: task ID and exposure score.
"""

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_SCORE_PATH = (
    Path(__file__).parent
    / "data"
    / "scored_interview_sampleocc3_30_2_span5_5mini_v4.pkl.zst"
)
DEFAULT_EXPOSURE_COL = "beta:interview"
TASK_ID_COL_CANDIDATES = ("Task ID", "task_id", "task id")


def parse_args():
    """
    Parse display options.

    Assumes relative paths should be resolved from the current working directory.
    Returns argparse's namespace of CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Pretty-print saved exposure scores by occupation.",
    )
    parser.add_argument(
        "--scores-path",
        default=str(DEFAULT_SCORE_PATH),
        help="Path to the saved scored interview DataFrame.",
    )
    parser.add_argument(
        "--task-id-col",
        default="",
        help="Column containing task IDs. If omitted, common task ID columns are tried.",
    )
    parser.add_argument(
        "--exposure-col",
        default=DEFAULT_EXPOSURE_COL,
        help="Column containing exposure scores.",
    )
    return parser.parse_args()


def require_columns(df, columns):
    """
    Validate that required columns are present.

    Assumes df is a pandas DataFrame. Raises ValueError listing available columns
    when an expected column is missing.
    """
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )


def load_scores(scores_path):
    """
    Load the saved scored interview DataFrame.

    Assumes the path points to a pandas pickle, including pandas-supported
    compression such as .pkl.zst. Returns the loaded DataFrame.
    """
    scores_df = pd.read_pickle(scores_path)
    if not isinstance(scores_df, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame at {scores_path}")
    return scores_df


def resolve_task_id_col(df, requested_col):
    """
    Choose the task ID column to print.

    Assumes either requested_col is present or the scored DataFrame preserved one
    of TASK_ID_COL_CANDIDATES. Returns the selected column name.
    """
    if requested_col:
        require_columns(df, [requested_col])
        return requested_col

    matches = [col for col in TASK_ID_COL_CANDIDATES if col in df.columns]
    if not matches:
        raise ValueError(
            "Could not find a task ID column. Pass --task-id-col explicitly. "
            f"Tried: {list(TASK_ID_COL_CANDIDATES)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return matches[0]


def pretty_print_scores(scores_df, task_id_col, exposure_col):
    """
    Print task IDs and exposures grouped by occupation.

    Assumes scores_df has category_occ, task_id_col, and exposure_col. Emits rows
    sorted by occupation and task ID for stable, auditable output.
    """
    require_columns(scores_df, ["category_occ", task_id_col, exposure_col])

    output_cols = ["category_occ", task_id_col, exposure_col]
    print_df = (
        scores_df[output_cols]
        .drop_duplicates()
        .sort_values(["category_occ", task_id_col])
    )

    for occupation, occupation_df in print_df.groupby("category_occ", sort=False):
        print()
        print(occupation)
        print("-" * len(str(occupation)))
        for _, row in occupation_df.iterrows():
            print(f"{row[task_id_col]}\t{row[exposure_col]}")


def main():
    args = parse_args()
    scores_df = load_scores(args.scores_path)
    task_id_col = resolve_task_id_col(scores_df, args.task_id_col)
    pretty_print_scores(scores_df, task_id_col, args.exposure_col)


if __name__ == "__main__":
    main()
