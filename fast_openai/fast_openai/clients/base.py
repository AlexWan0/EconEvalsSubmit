from typing import Generic, TypeVar, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from aiohttp import ClientSession
from tqdm import tqdm
from contextlib import nullcontext, AsyncExitStack
import traceback
import asyncio
from aiolimiter import AsyncLimiter
import os

from ..base import InputType, ResultType, Result
from ..cashews_cache import cache
from .exceptions import TokenLimitException, RetriesExceededException, OutputValidationException
from .logprobs import Logprobs
from .types import RequestArgs, _WrappedReturn
from .rate_limits import RateLimiter

ValidationFunctionType = Callable[[ResultType], bool]
VALIDATION_FUNCTION_DEFAULT = lambda x: True

@dataclass
class OutputValidation(Generic[ResultType]):
    """
    Define a `verify_function` which gets run on `ResultType`.
    If it returns False, then `APIClient` will throw a OutputValidationException
    and retry with no delay.
    """
    verify_function: ValidationFunctionType[ResultType] = VALIDATION_FUNCTION_DEFAULT


CACHE_KEY_TEMPLATE: str = "{self.__class__.__name__}.call:{payload.serialized},retry_idx={retry_idx}:hash_keys={request_args.hash_keys},compress_values={request_args.compress_values}"

class APIClient(ABC, Generic[InputType, ResultType]):
    @abstractmethod
    async def _call(
            self,
            payload: InputType,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[ResultType]:
        """
        Just does the API call and nothing else.
        Throws an error if something goes wrong.
        Implementation note: we should try to make it s.t. _call is as close as possible to a
        "vanilla" API call as possible. e.g., moving token tracking outside.
        """
        raise NotImplementedError()

    @cache(ttl="100d", key=CACHE_KEY_TEMPLATE)
    async def _cached_call(
            self,
            payload: InputType, # part of cache key
            session: ClientSession, # not part of cache key
            request_args: RequestArgs, # not part of cache key
            retry_idx: int, # part of cache key
            rate_limiters: tuple[RateLimiter, ...] = tuple() # not part of cache key
        ) -> _WrappedReturn[ResultType]:
        """
        Wrapper to call with caching and rate limiting.
        Throws an error if something goes wrong.

        All the middleware args (i.e., hash_keys=true/false, compress_values=true/false) must
        be at the end
        TODO: use a better way to pass these args
        """

        # update/check rate limits
        for rl_before in rate_limiters:
            await rl_before.acquire_before(payload)

        # actual call
        res = await self._call(payload, session, request_args)

        # update/check rate limits
        for rl_after in rate_limiters:
            await rl_after.acquire_after(res)

        return res

    async def process_input(
            self,
            payload: InputType,
            session: ClientSession,
            request_args: RequestArgs,
            output_validation: OutputValidation[ResultType] = OutputValidation[ResultType](),
            rate_limiters: tuple[RateLimiter, ...] = tuple()
        ) -> Result[ResultType]:
        """
        Performs the API call along with the everything else (e.g., error handling, tracking etc.).
        Intended to be used to process many inputs (hence the `idx` parameter).
        Will not throw an error if something goes wrong, instead will just return one in `Result.error`.
        """
        # TODO: would be nice just to just have some standalone function like 'run_model(input)' for simple experiments

        seen_exceptions: list[Exception] = [] # store all exceptions seen so far
        delay = request_args.retries_delay_start
        res = None
        for attempt in range(request_args.num_retries):
            try:
                maybe_disable = cache.disabling() if not request_args.use_cache else nullcontext()
                
                # used to track cache hits
                with cache.detect as detector: # type: ignore
                    with maybe_disable:
                        # ok finally try making the call
                        res = await self._cached_call(
                            payload=payload,
                            session=session,
                            retry_idx=attempt,
                            request_args=request_args,
                            rate_limiters=rate_limiters
                        )
                    
                    # report cache hit
                    used_cache = (len(detector.calls) > 0)

                # try to validate request output
                if not output_validation.verify_function(res.content):
                    raise OutputValidationException("Unable to validate model response")

                # return successful request
                return Result(
                    output=res.content,
                    error=None,
                    tokens_used=res.tokens_used,
                    from_cache=used_cache,
                    logprobs=res.logprobs,
                    reasoning_tokens=res.reasoning_tokens,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens
                )

            except Exception as e:
                # print info
                print(f"Error processing input: {payload.to_json()}\n  Attempt {attempt + 1}/{request_args.num_retries} failed with error: {str(e)}")
                print(traceback.format_exc()) # in 3.11 will be affected if you modify e

                traceback.clear_frames(e.__traceback__) # traceback contains refs to local vars
                seen_exceptions.append(e)

                # if the error was b/c of token limits then shut everything down
                if isinstance(e, TokenLimitException):
                    raise e

                # if the error was because of an OutputValidationException don't delay
                # before retrying
                if isinstance(e, OutputValidationException):
                    print(f"Model output:\n{res.content if res is not None else None}")
                    print("Retrying immediately")
                else:
                    # timeout before retrying
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

                    delay *= 2

        # reaching here means we've used up all our retries
        return Result(
            output=None,
            error=RetriesExceededException(
                seen_exceptions=seen_exceptions,
                num_retries=request_args.num_retries
            ),
            tokens_used=None,
            from_cache=False, # TODO: well... we could retrieve all our error inputs from cache
            logprobs=None
        )
