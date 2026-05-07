from dataclasses import dataclass
from itertools import chain
from typing import Callable, Literal

import pandas as pd

from .filters import CoverageFilterArgs, filter_by_coverage
from .utils import (
    assert_cols,
    build_task_examples,
    lint_convo_column,
    sample_list_column,
    zip_inputs_outputs,
)


@dataclass
class RetrievalArgs:
    use_task_retrieval: bool = True
    use_dwa_retrieval: bool = False
    coverage_args: CoverageFilterArgs | None = None


def retrieval_convo_to_user_text(convo: list[dict[str, str]]) -> str:
    return "\n".join(
        str(turn.get("content", ""))
        for turn in convo
        if turn.get("role") == "user"
    )


def _pick_diverse_indices(
    idxs: list[int],
    *,
    k: int,
    matrix: object,
) -> list[int]:
    if len(idxs) <= k:
        return idxs

    import numpy as np
    from sklearn.metrics import pairwise_distances

    # matrix is the global TF-IDF matrix; mirror validity.ipynb behavior.
    dist = pairwise_distances(matrix[idxs], metric="cosine", n_jobs=1) # type: ignore

    first = int(dist.mean(axis=1).argmax())
    selected = [first]

    while len(selected) < k:
        remaining = [i for i in range(len(idxs)) if i not in selected]
        min_dists = dist[np.ix_(remaining, selected)].min(axis=1)
        next_idx = remaining[int(min_dists.argmax())]
        selected.append(next_idx)

    return [idxs[i] for i in selected]


def select_tfidf_retrieval_rows(
    retrieved_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    retrieved_df = lint_convo_column(retrieved_df, convo_col="convo_subset")
    assert_cols(retrieved_df, ["category_occ", "category_task", "convo_subset"])
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    text_col = retrieved_df["convo_subset"].apply(retrieval_convo_to_user_text)
    tfidf_matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
        text_col.fillna("")
    )

    selected_indices: list[int] = []
    for _, group in retrieved_df.groupby(["category_occ", "category_task"]):
        idxs = group.index.to_list()
        selected_indices.extend(
            _pick_diverse_indices(idxs, k=k, matrix=tfidf_matrix)
        )

    return retrieved_df.loc[selected_indices].reset_index(drop=True)


def build_prompt_from_retrieval_row(
    row: pd.Series,
    *,
    prompt_ret: str,
    make_examples_str: Callable[[list[tuple[list[dict[str, str]], str]]], str],
    prompt_zs: str | None = None,
    other_cols: frozenset[str] = frozenset(),
) -> tuple[str | None, str]:
    if isinstance(row["full_convo_bytask"], list):
        examples_str = make_examples_str(row["full_convo_bytask"])
        return (
            prompt_ret.format(
                examples_strs=examples_str,
                occupation=row["category_occ"],
                task=row["category_task"],
                **{col: row[col] for col in other_cols},
            ),
            "task",
        )

    if isinstance(row["full_convo_bydwas"], list):
        examples_str = make_examples_str(row["full_convo_bydwas"])
        return (
            prompt_ret.format(
                examples_strs=examples_str,
                occupation=row["category_occ"],
                task=row["category_task"],
                **{col: row[col] for col in other_cols},
            ),
            "dwa",
        )

    if prompt_zs is not None:
        return (
            prompt_zs.format(
                occupation=row["category_occ"],
                task=row["category_task"],
                **{col: row[col] for col in other_cols},
            ),
            "zs",
        )

    return (None, "none")


def build_retrieval_table(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    args: RetrievalArgs,
    task_dwa_df: pd.DataFrame | None = None,
    other_cols: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    assert_cols(retrieved_df, ["category_occ", "category_task", "convo_subset", "response"])
    assert_cols(task_space_df, ["category_occ", "category_task"])

    print(other_cols)

    # other cols may be in retrieved_df or task_space_df; check and separate them here
    ret_other_cols = [col for col in other_cols if col in retrieved_df.columns]
    ts_other_cols = [col for col in other_cols if col in task_space_df.columns]
    if (len(ret_other_cols) + len(ts_other_cols)) < len(other_cols):
        missing_cols = other_cols - set(ret_other_cols) - set(ts_other_cols)
        raise ValueError(
            f"Other cols {sorted(missing_cols)} not found in either retrieved_df or task_space_df. "
            f"Found cols in retrieved_df: {sorted(retrieved_df.columns)}, "
            f"found cols in task_space_df: {sorted(task_space_df.columns)}."
        )

    # TODO: other cols in retrieved_df will need to be aggregated; kind've annoying to handle
    if len(ret_other_cols) > 0:
        raise NotImplementedError(
            "Other cols in retrieved_df not yet supported; found these cols in retrieved_df: "
            f"{sorted(ret_other_cols)}. Only other cols in task_space_df are currently supported."
        )

    base_df = task_space_df[["category_occ", "category_task", *ts_other_cols]].drop_duplicates().copy()

    print(ts_other_cols, base_df.columns)

    if args.use_task_retrieval:
        task_to_ret_df = build_task_examples(retrieved_df)
        task_to_ret_df["full_convo_bytask"] = task_to_ret_df.apply(
            zip_inputs_outputs,
            axis=1,
        )
        task_to_ret_df = task_to_ret_df[
            [
                "category_occ",
                "category_task",
                "model_inputs",
                "model_outputs",
                "full_convo_bytask",
            ]
        ]

        out_df = base_df.merge(
            task_to_ret_df,
            on=["category_occ", "category_task"],
            how="left",
        )

        if args.coverage_args is not None:
            out_df = filter_by_coverage(out_df, args.coverage_args)
    else:
        if args.coverage_args is not None:
            raise ValueError("coverage_args requires use_task_retrieval=True")

        out_df = base_df.copy()
        out_df["model_inputs"] = None
        out_df["model_outputs"] = None
        out_df["full_convo_bytask"] = None

    if args.use_dwa_retrieval and task_dwa_df is not None:
        assert_cols(retrieved_df, ["dwa"])
        assert_cols(task_dwa_df, ["category_occ", "category_task", "dwa"])

        task_dwa_map_df = task_dwa_df[
            ["category_occ", "category_task", "dwa"]
        ].drop_duplicates()

        dwa_to_ret_df = (
            retrieved_df.groupby(["dwa"])
            .agg(model_inputs=("convo_subset", list), model_outputs=("response", list))
            .reset_index()
        )
        dwa_to_ret_df["full_convo"] = dwa_to_ret_df.apply(zip_inputs_outputs, axis=1)
        dwa_to_ret_df = dwa_to_ret_df[["dwa", "full_convo"]]

        task_to_ret_df_bydwas = (
            task_dwa_map_df.merge(dwa_to_ret_df, on="dwa", how="inner")
            .groupby(["category_occ", "category_task"], as_index=False)
            .agg(full_convo_bydwas=("full_convo", lambda x: list(chain(*x))))
        )

        out_df = out_df.merge(
            task_to_ret_df_bydwas,
            on=["category_occ", "category_task"],
            how="left",
        )
    else:
        out_df["full_convo_bydwas"] = None

    print(out_df.columns)

    return out_df


def build_retrieval_prompt_table(
    retrieved_df: pd.DataFrame,
    task_space_df: pd.DataFrame,
    retrieval_args: RetrievalArgs,
    *,
    prompt_ret: str,
    make_examples_str: Callable[[list[tuple[list[dict[str, str]], str]]], str],
    prompt_col: str,
    method_col: str,
    max_examples: int,
    sample_seed: int,
    use_tfidf_diversity: bool,
    tfidf_k: int,
    prompt_zs: str | None = None,
    task_dwa_df: pd.DataFrame | None = None,
    drop_missing_prompt: bool = False,
    other_cols: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    retrieved_for_build = lint_convo_column(
        retrieved_df,
        convo_col="convo_subset",
    )
    if use_tfidf_diversity:
        retrieved_for_build = select_tfidf_retrieval_rows(
            retrieved_df,
            k=tfidf_k,
        )

    prompt_df = build_retrieval_table(
        retrieved_for_build,
        task_space_df,
        args=retrieval_args,
        task_dwa_df=task_dwa_df,
        other_cols=other_cols,
    )

    prompt_df = sample_list_column(
        prompt_df,
        col="full_convo_bytask",
        n=max_examples,
        seed=sample_seed,
    )
    prompt_df = sample_list_column(
        prompt_df,
        col="full_convo_bydwas",
        n=max_examples,
        seed=sample_seed,
    )

    prompt_df[[prompt_col, method_col]] = prompt_df.apply(
        lambda row: pd.Series(
            build_prompt_from_retrieval_row(
                row,
                prompt_ret=prompt_ret,
                make_examples_str=make_examples_str,
                prompt_zs=prompt_zs,
                other_cols=other_cols,
            )
        ),
        axis=1,
    )

    if drop_missing_prompt:
        prompt_df = prompt_df.dropna(subset=[prompt_col]).copy()

    return prompt_df
