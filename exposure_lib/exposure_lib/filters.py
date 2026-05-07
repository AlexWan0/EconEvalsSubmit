from dataclasses import dataclass

import pandas as pd

from .utils import assert_cols


@dataclass
class CoverageFilterArgs:
    min_coverage: float = 0.3
    evidence_col: str = "model_inputs"


def add_coverage_columns(
    df: pd.DataFrame,
    args: CoverageFilterArgs,
) -> pd.DataFrame:
    assert_cols(df, ["category_occ", args.evidence_col])

    out_df = df.copy()

    out_df["occ_coverage"] = (
        out_df.groupby("category_occ")[args.evidence_col].transform(
            lambda col: col.notna().mean()
        )
    )
    out_df["occ_count"] = (
        out_df.groupby("category_occ")[args.evidence_col].transform(
            lambda col: col.notna().sum()
        )
    )
    out_df["occ_total"] = (
        out_df.groupby("category_occ")[args.evidence_col].transform(
            lambda col: len(col)
        )
    )

    return out_df


def filter_by_coverage(
    df: pd.DataFrame,
    args: CoverageFilterArgs,
) -> pd.DataFrame:
    with_coverage_df = add_coverage_columns(df, args)
    return with_coverage_df[with_coverage_df["occ_coverage"] >= args.min_coverage].copy()
