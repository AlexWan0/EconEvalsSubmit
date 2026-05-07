import gzip
import hashlib
import json
import pickle
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, TypedDict

from fast_openai import RequestArgs


def assert_cols(df: pd.DataFrame, expected_cols: list[str]) -> None:
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing expected columns: {missing_cols}. Available columns: {df.columns.tolist()}")


@dataclass
class LLMArgs:
    model_name: str
    temperature: float
    max_tokens: int
    num_workers: int
    request_args: RequestArgs = field(default_factory=lambda: RequestArgs(
        use_cache=True,
        hash_keys=True,
        num_retries=5,
        post_timeout=300,
        total_timeout=300,
    ))
    cache_flag: str = ''


_LLM_ARGS_HASH_NUM_CHARS = 4


def normalize_model_str(model_str: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in model_str])


def llm_args_hash(llm_args: LLMArgs) -> str:
    args_dict = {
        "model_name": llm_args.model_name,
        "temperature": llm_args.temperature,
        "max_tokens": llm_args.max_tokens,
        "num_workers": llm_args.num_workers,
        "cache_flag": llm_args.cache_flag,
        "request_args": llm_args.request_args.__dict__,
    }
    payload = json.dumps(args_dict, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_LLM_ARGS_HASH_NUM_CHARS]


def llm_args_suffix(llm_args: LLMArgs) -> str:
    return f"{normalize_model_str(llm_args.model_name)}-{llm_args_hash(llm_args)}"

class Turn(TypedDict):
    role: str
    content: str

def format_turns(row: pd.Series | dict[str, str], turns_prompt: list[Turn]) -> list[Turn]:
    return [
        {
            'role': turn['role'],
            'content': turn['content'].format(**row)
        }
        for turn in turns_prompt
    ]


def print_turns(turns: list[dict[str, str]]) -> None:
    for turn in turns:
        print('=' * 80)
        print(f'Role: {turn["role"]}')
        print('-' * 80)
        print(turn["content"])
        print('=' * 80)


def save_pickle_gzip(obj: Any, filepath: str) -> None:
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_pickle_gzip(filepath: str) -> Any:
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)
