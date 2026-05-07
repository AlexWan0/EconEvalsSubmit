from enum import Enum
from typing import Generic, TypeVar, TypeAlias
from abc import ABC, abstractmethod

from dataclasses import dataclass
from .utils import fast_json_dumps, JSONType
from .clients.logprobs import Logprobs

class Serializable(ABC):
    @abstractmethod
    def to_json(self) -> JSONType:
        raise NotImplementedError()

    @property
    def serialized(self) -> str:
        data = self.to_json()

        return fast_json_dumps(data)

InputType = TypeVar("InputType", bound=Serializable)
ResultType = TypeVar("ResultType")

@dataclass
class Success(Generic[ResultType]):
    output: ResultType
    tokens_used: int
    from_cache: bool
    logprobs: Logprobs | None

@dataclass
class Failure(Generic[ResultType]):
    error: Exception

MaybeResult = Success[ResultType] | Failure[ResultType]

@dataclass
class Result(Generic[ResultType]):
    """
    The model output plus some metadata/errors if the request fails
    """
    output: ResultType | None # if error is not None then output has to be not None
    error: Exception | None
    tokens_used: int | None
    from_cache: bool # if error then is False
    logprobs: Logprobs | None = None
    reasoning_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def is_success(self) -> bool:
        return (
            (self.output is not None) and
            (self.error is None) and
            (self.tokens_used is not None)
            # (self.from_cache) can be either True or False
            # (self.logprobs) can be either None or not None
        )

    @property
    def is_failure(self) -> bool:
        return (
            (self.output is None) and
            (self.error is not None) and
            (self.tokens_used is None) and
            (not self.from_cache) and
            (self.logprobs is None)
        )

    def __post_init__(self):
        if (self.is_success + self.is_failure) != 1:
            raise ValueError(f'Invalid result values: {self.__dict__}; is_success={self.is_success} and is_failure={self.is_failure}')

    def as_maybe(self) -> MaybeResult[ResultType]:
        """
        After init we alreacy check that we either have a set of values that either
        conforms to Success or Failure (__post_init__), so we can just convert it.
        TODO: just return MaybeResult to begin with?
        """

        if self.is_success:
            # must be exclusively success or fail
            if self.is_failure:
                raise ValueError(f'Invalid result values: {self.__dict__}')

            # to make type checker happy; unnecessary?
            if self.output is None or self.tokens_used is None:
                raise ValueError(f'Invalid result values: {self.__dict__}')

            return Success[ResultType](
                output=self.output,
                tokens_used=self.tokens_used,
                from_cache=self.from_cache,
                logprobs=self.logprobs
            )
        
        # must be exclusively success or fail
        if not self.is_failure:
            raise ValueError(f'Invalid result values: {self.__dict__}')

        # to make type checker happy; unnecessary?
        if not self.error:
            raise ValueError(f'Invalid result values: {self.__dict__}')
        
        return Failure[ResultType](
            self.error
        )
