from dataclasses import dataclass, field

import pandas as pd

from .background_generation import BackgroundGenArgs
from .postprocess_generation import QueryPostArgs, postprocess_queries
from .query_generation import QueryGenArgs, generate_queries
from .reference_generation import ReferenceGenArgs, generate_references
from .utils import assert_cols
from .verifier import (
    BaseVerifierArgs,
    apply_verifier_prompt,
)


@dataclass
class SearchArgs:
    postprocess_query: bool = True
    query_post_args: QueryPostArgs = field(default_factory=QueryPostArgs)
    generate_references: bool = False
    reference_args: ReferenceGenArgs = field(default_factory=ReferenceGenArgs)


@dataclass
class SearchResult:
    verified_df: pd.DataFrame
    not_found_df: pd.DataFrame
    intermediate_df: pd.DataFrame


def search_time_savings(
    input_df_samp: pd.DataFrame,
    time_savings: list[str],
    bg_args: BackgroundGenArgs,
    query_args: QueryGenArgs,
    verifier_args: list[BaseVerifierArgs],
    search_args: SearchArgs,
) -> SearchResult:
    """
    Iteratively search for queries in the order of `time_savings`. On each iteration, generate queries for the current time saving, run verifiers on them, and keep the first passing query for each input row. Remove passing rows from the pool for the next iteration.
    """
    assert_cols(input_df_samp, ["category_occ", "task_detailed"])
    if not verifier_args:
        raise ValueError("verifier_args must contain at least one verifier config")

    search_base_df = input_df_samp.reset_index(drop=True).copy()
    search_base_df["__row_id"] = search_base_df.index

    # on each iter, remove rows from pending_df that have passed verifiers
    pending_df = search_base_df.copy()
    selected_rows: list[pd.DataFrame] = [] # ...and add them to selected_rows to concat at the end

    intermediate_rows: list[pd.DataFrame] = [] # for debugging: keep track of all generated rows & verifications

    for ts_idx, ts in enumerate(time_savings):
        if pending_df.empty:
            break

        print(f"Searching for time saving: {ts} (pending rows: {len(pending_df)})")

        iter_input_df = pending_df.copy()
        iter_input_df["time_savings"] = ts

        # generate initial queries
        generated_df = generate_queries(
            iter_input_df,
            bg_args=bg_args,
            query_args=query_args,
        )
        assert "query" in generated_df.columns, "generate_queries must return a DataFrame with a 'query' column"
        generated_df = generated_df.dropna(subset=["query"]).reset_index(drop=True)
        if generated_df.empty:
            continue

        # possibly rewrite queries w/ an LM
        if search_args.postprocess_query:
            generated_df = postprocess_queries(
                generated_df,
                args=search_args.query_post_args,
            )

        if search_args.generate_references:
            generated_df = generate_references(
                generated_df,
                args=search_args.reference_args,
            )

        # run verifiers
        verified_df = generated_df.copy()
        pass_cols: list[str] = []
        for verifier_arg in verifier_args:
            verified_df = apply_verifier_prompt(verified_df, args=verifier_arg)
            verified_df[verifier_arg.pass_value_col] = verified_df.apply(
                lambda row: BaseVerifierArgs.check_row_pass(row, verifier_arg),
                axis=1,
            )
            pass_cols.append(verifier_arg.pass_value_col)

        verified_df["passed_both"] = verified_df[pass_cols].all(axis=1)

        intermediate_rows.append(verified_df)

        passed_df = verified_df[verified_df["passed_both"]].copy()
        if passed_df.empty:
            continue
        
        # keep only the first passing query per input row
        selected_iter_df = passed_df.sort_values(["__row_id"]).drop_duplicates(
            subset=["__row_id"],
            keep="first",
        )
        selected_rows.append(selected_iter_df)

        # remove passed rows from pending_df for the next iter
        passed_row_ids = selected_iter_df["__row_id"].dropna().unique().tolist()
        pending_df = pending_df[~pending_df["__row_id"].isin(passed_row_ids)].copy()

    # collate passed rows across all iters
    selected_verified_df = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else pd.DataFrame()
    )
    intermediate_df = (
        pd.concat(intermediate_rows, ignore_index=True)
        if intermediate_rows
        else pd.DataFrame()
    )

    if not selected_verified_df.empty:
        selected_verified_df = (
            selected_verified_df
            .sort_values(["__row_id"])
            .drop_duplicates(subset=["__row_id"], keep="first")
            .reset_index(drop=True)
        )

    return SearchResult(
        verified_df=selected_verified_df,
        not_found_df=pending_df,
        intermediate_df=intermediate_df,
    )
