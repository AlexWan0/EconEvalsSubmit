from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import json
from dataclasses import dataclass
from cashews.key import get_cache_key, get_cache_key_template

from ..base import Serializable, JSONType, Result
from ..cashews_cache import cache
from .logprobs import collect_openai_logprobs
from .base import APIClient, _WrappedReturn, RequestArgs, InputType, ResultType, OutputValidation, CACHE_KEY_TEMPLATE
from .auth import OPENAI_API_KEY
from .exceptions import RequestException, TokenLimitException, InvalidResponseError, CacheMissException
from .token_tracking import oai_tokens_tracker
from .rate_limits import RateLimiter


class CacheOnlyClient(APIClient[InputType, ResultType]):
    async def _call(
            self,
            payload: InputType,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[ResultType]:
        """
        Should never reach here.
        """
        raise Exception("Reached _call in CacheOnlyClient")

    async def _cached_call(
            self,
            payload: InputType, # part of cache key
            session: ClientSession, # not part of cache key
            request_args: RequestArgs, # not part of cache key
            retry_idx: int, # part of cache key
            rate_limiters: tuple[RateLimiter, ...] = tuple() # not part of cache key
        ) -> _WrappedReturn[ResultType]:
        """
        Should never reach here.
        """
        raise Exception("Reached _cached_call in CacheOnlyClient")

    async def process_input(
            self,
            payload: InputType,
            session: ClientSession,
            request_args: RequestArgs,
            output_validation: OutputValidation[ResultType] = OutputValidation[ResultType](),
            rate_limiters: tuple[RateLimiter, ...] = tuple()
        ) -> Result[ResultType]:
        """
        Tried to retrieve the target payload from cache, and returns a failed Result if
        the key isn't found.
        """

        if not request_args.use_cache:
            raise ValueError('request_args.use_cache set to False for CacheOnlyClient')
        
        key_template = get_cache_key_template(
            self._cached_call,
            key=CACHE_KEY_TEMPLATE,
            prefix=""
        )

        for retry_idx in range(request_args.num_retries):
            args = (
                payload,
                session,
                retry_idx,
                request_args,
            )

            kwargs = {
                'rate_limiters': rate_limiters,
                'self': self
            }

            cache_key = get_cache_key(self._cached_call, key_template, args, kwargs)

            print(cache_key)

            res: _WrappedReturn[ResultType] | None = await cache.get(cache_key)

            if res is not None:
                return Result(
                    output=res.content,
                    error=None,
                    tokens_used=res.tokens_used,
                    from_cache=True,
                    logprobs=res.logprobs
                )

        return Result(
            output=None,
            error=CacheMissException(),
            tokens_used=None,
            from_cache=False,
            logprobs=None
        )
