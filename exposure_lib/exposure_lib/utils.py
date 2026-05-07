import random
import gzip
import pickle
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, TypeVar, TypedDict
import re
import os

import pandas as pd
from fast_openai import RequestArgs, run_auto, run_auto_str
from typeguard import check_type


def assert_cols(df: pd.DataFrame, expected_cols: list[str]) -> None:
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing expected columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )


def lint_convo(convo: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for turn in convo:
        assert "role" in turn and "content" in turn, f"Each turn must have 'role' and 'content' keys. Invalid turn: {turn}"
        assert isinstance(turn["role"], str) and isinstance(turn["content"], str), f"'role' and 'content' must be strings. Invalid turn: {turn}"
        result.append({
            "role": turn["role"],
            "content": turn["content"],
        })
    return result


def lint_convo_column(
    df: pd.DataFrame,
    convo_col: str = "convo_subset",
) -> pd.DataFrame:
    assert_cols(df, [convo_col])

    out_df = df.copy()
    out_df[convo_col] = out_df[convo_col].apply(
        lambda convo: lint_convo(convo) if isinstance(convo, list) else convo
    )

    return out_df


@dataclass
class LLMArgs:
    model_name: str
    temperature: float
    max_tokens: int
    num_workers: int
    request_args: RequestArgs = field(
        default_factory=lambda: RequestArgs(
            use_cache=True,
            hash_keys=True,
            num_retries=5,
            post_timeout=300,
            total_timeout=300,
        )
    )
    cache_flag: str = ""


_LLM_ARGS_HASH_NUM_CHARS = 4


def llm_args_to_dict(llm_args: LLMArgs) -> dict[str, object]:
    return {
        "model_name": llm_args.model_name,
        "temperature": llm_args.temperature,
        "max_tokens": llm_args.max_tokens,
        "num_workers": llm_args.num_workers,
        "cache_flag": llm_args.cache_flag,
        "request_args": llm_args.request_args.__dict__,
    }


def llm_args_from_dict(data: dict[str, object]) -> LLMArgs:
    request_args_data = data.get("request_args", {})
    if not isinstance(request_args_data, dict):
        raise ValueError("LLMArgs JSON must contain request_args as an object")

    return LLMArgs(
        model_name=str(data["model_name"]),
        temperature=float(data["temperature"]), # type: ignore
        max_tokens=int(data["max_tokens"]), # type: ignore
        num_workers=int(data["num_workers"]), # type: ignore
        cache_flag=str(data.get("cache_flag", "")),
        request_args=RequestArgs(**request_args_data),
    )


def llm_args_to_pretty_json(llm_args: LLMArgs) -> str:
    return json.dumps(
        llm_args_to_dict(llm_args),
        indent=2,
        sort_keys=True,
    )


def llm_args_from_json_str(data: str) -> LLMArgs:
    return llm_args_from_dict(json.loads(data))


def save_llm_args_json(llm_args: LLMArgs, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(llm_args_to_pretty_json(llm_args))


def read_llm_args_json(filepath: str) -> LLMArgs:
    with open(filepath, "r", encoding="utf-8") as f:
        return llm_args_from_json_str(f.read())


def normalize_model_str(model_str: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in model_str])


def llm_args_hash(llm_args: LLMArgs) -> str:
    payload = json.dumps(llm_args_to_dict(llm_args), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_LLM_ARGS_HASH_NUM_CHARS]


def llm_args_suffix(llm_args: LLMArgs) -> str:
    return f"{normalize_model_str(llm_args.model_name)}-{llm_args_hash(llm_args)}"


def run_prompt_column(
    df: pd.DataFrame,
    prompt_col: str,
    llm_args: LLMArgs,
    output_col: str,
    parser: Callable[[str | None], object],
    pbar_name: str = 'Running prompts',
) -> pd.DataFrame:
    assert_cols(df, [prompt_col])

    out_df = df.copy()

    out_df[f"_model_output_{output_col}"] = run_auto_str(
        out_df[prompt_col],
        full_model_name=llm_args.model_name,
        temperature=llm_args.temperature,
        max_tokens=llm_args.max_tokens,
        num_workers=llm_args.num_workers,
        request_args=llm_args.request_args,
        cache_flag=llm_args.cache_flag,
        pbar_name=pbar_name,
    )

    out_df[f"_model_output_str_{output_col}"] = out_df[
        f"_model_output_{output_col}"
    ].apply(lambda x: x.output)

    out_df[output_col] = out_df[f"_model_output_str_{output_col}"].apply(parser)

    return out_df


def run_convo_column(
    df: pd.DataFrame,
    convo_col: str,
    llm_args: LLMArgs,
    output_col: str,
    parser: Callable[[str | None], object],
    pbar_name: str,
) -> pd.DataFrame:
    assert_cols(df, [convo_col])

    out_df = lint_convo_column(df, convo_col=convo_col)

    out_df[f"_model_output_{output_col}"] = run_auto( # type: ignore
        out_df[convo_col],
        full_model_name=llm_args.model_name,
        temperature=llm_args.temperature,
        max_tokens=llm_args.max_tokens,
        num_workers=llm_args.num_workers,
        request_args=llm_args.request_args,
        cache_flag=llm_args.cache_flag,
        pbar_name=pbar_name,
    )

    out_df[f"_model_output_str_{output_col}"] = out_df[
        f"_model_output_{output_col}"
    ].apply(lambda x: x.output)

    out_df[output_col] = out_df[f"_model_output_str_{output_col}"].apply(parser) # type: ignore

    return out_df


def parse_xml_tags(
    xml_text: str,
    tags: list[str] | None = None,
    require_code_fence: bool = True,
) -> dict[str, str | list[str]] | None:
    text = xml_text.strip()

    fence_pat = re.compile(r"(?s)^\s*```[^\n]*\n(.*?)\n\s*```\s*$")

    if require_code_fence:
        m = fence_pat.match(text)
        if m is None:
            return None
        text = m.group(1).strip()
    else:
        if text.startswith("```"):
            m = fence_pat.match(text)
            if m is not None:
                text = m.group(1).strip()

    tag_pattern = re.compile(
        r"<(?P<tag>[A-Za-z_][\w\-.]*)(?:\s+[^>]*)?>(?P<value>.*?)</(?P=tag)>",
        re.DOTALL,
    )

    include = set(tags) if tags is not None else None
    out: dict[str, str | list[str]] = {}

    for m in tag_pattern.finditer(text):
        tag = m.group("tag")
        if include is not None and tag not in include:
            continue

        val = m.group("value").strip()
        if tag in out:
            existing = out[tag]
            out[tag] = [*([existing] if not isinstance(existing, list) else existing), val]
        else:
            out[tag] = val

    return out


def detect_multiturn_response_indices(messages: list[dict]) -> list[int]:
    return [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), dict)
        and "system" in message["content"]
    ]


def build_multiturn_prompt_input(
    row: pd.Series,
    messages: list[dict],
    resp_idx: int,
) -> list[dict[str, str]]:
    if resp_idx < 0 or resp_idx >= len(messages):
        raise ValueError(f"resp_idx {resp_idx} is out of range for messages")

    response_turn = messages[resp_idx]

    if response_turn["role"] != "assistant":
        raise ValueError(
            f"Expected assistant response turn at messages[{resp_idx}], got role: {response_turn['role']}"
        )
    system_prompt = response_turn["content"]["system"]

    full_prompt = [
        {"role": "system", "content": system_prompt},
        *[
            (
                message
                if isinstance(message.get("content"), str)
                else {"role": message.get("role"), "content": "{_result_" + str(idx) + "}"}
            )
            for idx, message in enumerate(messages[:resp_idx])
        ]
    ]

    return [
        {
            "role": turn["role"],
            "content": turn["content"].format(**row),
        }
        for turn in full_prompt
    ]

def run_multiturn_prompts(
    df: pd.DataFrame,
    messages: list[dict],
    llm_args: LLMArgs,
    parser: Callable[[str | None], object] | None = None,
    pbar_name: str = "Interview multi-turn prompts ({model_name})",
    start_from_turn: int | None = None,
) -> pd.DataFrame:
    out_df = df.copy()

    response_turns = detect_multiturn_response_indices(messages)
    if not response_turns:
        raise ValueError("No assistant response turns found in message template")

    response_turns = list(sorted(response_turns))

    if start_from_turn is not None:
        assert start_from_turn in response_turns, f"start_from_turn {start_from_turn} not found in detected response turns {response_turns}"
        response_turns = [idx for idx in response_turns if idx >= start_from_turn]
        print(f'starting multi-turn prompts from turn {start_from_turn}, remaining response turns: {response_turns}')

    for idx in response_turns:
        if idx < 0 or idx >= len(messages):
            raise ValueError(f"response turn index {idx} out of range for messages")

        model_input_col = f"_model_input_{idx}"
        model_output_col = f"_model_output_{idx}"
        model_output_str_col = f"_model_output_str_{idx}"
        result_col = f"_result_{idx}"

        out_df[model_input_col] = out_df.apply(
            lambda row: build_multiturn_prompt_input(row, messages, idx),
            axis=1,
        )
        out_df[model_output_col] = run_auto( # type: ignore
            out_df[model_input_col],
            full_model_name=llm_args.model_name,
            max_tokens=llm_args.max_tokens,
            temperature=llm_args.temperature,
            num_workers=llm_args.num_workers,
            request_args=llm_args.request_args,
            cache_flag=llm_args.cache_flag,
            pbar_name=pbar_name,
        )
        out_df[model_output_str_col] = out_df[model_output_col].apply(lambda x: x.output)

        if parser is None:
            out_df[result_col] = out_df[model_output_str_col]
        else:
            out_df[result_col] = out_df[model_output_str_col].apply(parser)

    return out_df


def add_response_column_from_convo(
    df: pd.DataFrame,
    llm_args: LLMArgs,
    convo_col: str = "convo_subset",
    output_col_prefix: str = "response",
    skip_if_exists: bool = False
) -> tuple[pd.DataFrame, str]:
    assert_cols(df, [convo_col])

    output_col = f"{output_col_prefix}-{llm_args_suffix(llm_args)}"
    out_df = lint_convo_column(df.copy(), convo_col=convo_col)

    if skip_if_exists and output_col in out_df.columns:
        print(f"Output column {output_col} already exists, skipping response generation")
        return out_df, output_col

    out_df[f"_model_output_{output_col}"] = run_auto( # type: ignore
        out_df[convo_col],
        full_model_name=llm_args.model_name,
        max_tokens=llm_args.max_tokens,
        temperature=llm_args.temperature,
        num_workers=llm_args.num_workers,
        request_args=llm_args.request_args,
        cache_flag=llm_args.cache_flag,
        pbar_name="Generating response column ({model_name})",
    )

    out_df[f"_model_output_str_{output_col}"] = out_df[f"_model_output_{output_col}"].apply(
        lambda x: x.output
    )
    out_df[output_col] = out_df[f"_model_output_str_{output_col}"]

    return out_df, output_col


def zip_inputs_outputs(
    row: pd.Series,
    model_inputs_col: str = "model_inputs",
    model_outputs_col: str = "model_outputs",
) -> list[tuple[list[dict[str, str]], str]] | None:
    pairs = []

    for model_input, model_output in zip(
        row[model_inputs_col],
        row[model_outputs_col],
    ):
        if model_output is None:
            continue

        pairs.append((model_input, model_output))

    return pairs if pairs else None


def sample_list_column(
    df: pd.DataFrame,
    col: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    assert_cols(df, [col])

    out_df = df.copy()
    rng = random.Random(seed)

    out_df[col] = out_df[col].apply(
        lambda x: rng.sample(x, n) if isinstance(x, list) and len(x) > n else x
    )

    return out_df


T = TypeVar("T")


def select_tfidf_diverse_items(
    items: list[T],
    n: int,
    item_to_text: Callable[[T], str],
) -> list[T]:
    if len(items) <= n:
        return items

    texts = [item_to_text(item) for item in items]
    if len(set(texts)) <= 1:
        return items[:n]

    # local import to avoid forcing sklearn dependency when this utility is unused
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    dist = pairwise_distances(tfidf_matrix, metric="cosine", n_jobs=1)

    first_idx = int(dist.mean(axis=1).argmax())
    selected = [first_idx]

    while len(selected) < n:
        remaining = [i for i in range(len(items)) if i not in selected]
        min_dists = dist[np.ix_(remaining, selected)].min(axis=1)
        next_idx = remaining[int(min_dists.argmax())]
        selected.append(next_idx)

    return [items[i] for i in selected]


def sample_list_column_tfidf_diverse(
    df: pd.DataFrame,
    col: str,
    n: int,
    item_to_text: Callable[[T], str],
) -> pd.DataFrame:
    assert_cols(df, [col])

    out_df = df.copy()
    out_df[col] = out_df[col].apply(
        lambda items: (
            select_tfidf_diverse_items(items, n=n, item_to_text=item_to_text)
            if isinstance(items, list) and len(items) > n
            else items
        )
    )
    return out_df


def build_task_examples(
    df: pd.DataFrame,
) -> pd.DataFrame:
    assert_cols(df, ["category_occ", "category_task", "convo_subset", "response"])

    return (
        df.groupby(["category_occ", "category_task"])
        .agg(model_inputs=("convo_subset", list), model_outputs=("response", list))
        .reset_index()
    )


def map_score_column(
    df: pd.DataFrame,
    exposure_col: str,
    output_col: str,
    score_map: dict[str, float],
) -> pd.DataFrame:
    assert_cols(df, [exposure_col])

    out_df = df.copy()
    out_df[output_col] = out_df[exposure_col].map(score_map)

    return out_df


def save_score_count_plot(
    df: pd.DataFrame,
    score_col: str,
    output_path: str,
    model_col: str | None = None,
) -> None:
    expected_cols = [score_col]
    if model_col is not None:
        expected_cols.append(model_col)
    assert_cols(df, expected_cols)

    # local import to avoid forcing matplotlib dependency when plotting is unused
    import matplotlib.pyplot as plt

    def _to_label(val: object) -> str:
        return "None" if pd.isna(val) else str(val) # type: ignore

    def _sort_key(label: str) -> tuple[int, float | str]:
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    score_labels = df[score_col].apply(_to_label)
    ordered_scores = sorted(score_labels.unique().tolist(), key=_sort_key)

    fig, ax = plt.subplots(figsize=(10, 6))
    if model_col is None:
        counts = score_labels.value_counts()
        counts = counts.reindex(ordered_scores, fill_value=0)
        ax.bar(counts.index.tolist(), counts.values.tolist()) # type: ignore
    else:
        model_labels = df[model_col].apply(_to_label)
        counts_df = (
            pd.DataFrame({"score": score_labels, "model": model_labels})
            .groupby(["score", "model"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        pivot_df = counts_df.pivot(index="score", columns="model", values="count").fillna(0)
        pivot_df = pivot_df.reindex(ordered_scores).fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        pivot_df.plot(kind="bar", ax=ax)
        ax.legend(title=model_col)

    ax.set_xlabel(score_col)
    ax.set_ylabel("count")
    ax.set_title(f"Count Plot: {score_col}")
    fig.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def save_pickle_gzip(obj: object, filepath: str) -> None:
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_pickle_gzip(filepath: str) -> object:
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)

def cached_xlsx_df(url: str, **kwargs) -> pd.DataFrame:
    cache_dir = kwargs.pop("cache_dir", ".cache/")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Create a deterministic filename based on URL + kwargs
    key = url + str(sorted(kwargs.items()))
    filename = hashlib.md5(key.encode("utf-8")).hexdigest() + ".xlsx"
    filepath = os.path.join(cache_dir, filename)

    # Download and cache if not already present
    if not os.path.exists(filepath):
        df = pd.read_excel(url, **kwargs)
        df.to_excel(filepath, index=False)
        return df

    # Load from cache
    return pd.read_excel(filepath, **kwargs)
