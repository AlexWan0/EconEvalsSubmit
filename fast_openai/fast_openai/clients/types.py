from dataclasses import dataclass
from typing import Generic, TypeVar, Tuple, Callable
from .logprobs import Logprobs
import os

from ..base import InputType, ResultType, Result


DEFAULT_RPM_LIMIT = 1_000_000_000 # (practically) no limit by default
DEFAULT_TPM_LIMIT = 1_000_000_000_000 # (practically) no limit by default
OPENAI_EMBEDDING_LIMIT = 10_000 // 2

@dataclass
class _WrappedReturn(Generic[ResultType]):
    """
    Adds some metadata for API responses.
    Unlike `..base.Result`, `_WrappedResult` is meant to be used internally only.
    """
    content: ResultType
    tokens_used: int
    logprobs: Logprobs | None  = None
    reasoning_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

@dataclass
class RequestArgs:
    """
    Parameters for how the request is made.
    """
    use_cache: bool = True
    hash_keys: bool = False
    compress_values: bool = False

    num_retries: int = 3
    retries_delay_start: int = 5
    post_timeout: int = 60
    
    total_timeout: int = 2 * 60
    connect_timeout: int = 5

    rate_limit_rpm: int = int(os.getenv('FAST_OPENAI_MAX_RPM', DEFAULT_RPM_LIMIT))
    rate_limit_tpm: int = int(os.getenv('FAST_OPENAI_MAX_TPM', DEFAULT_TPM_LIMIT))

