from ..base import InputType, ResultType
from typing import Generic
from abc import ABC, abstractmethod
from aiolimiter import AsyncLimiter

from .types import _WrappedReturn


class RateLimiter(ABC, Generic[InputType, ResultType]):
    @abstractmethod
    async def acquire_before(self, payload: InputType):
        raise NotImplementedError()
    
    @abstractmethod
    async def acquire_after(self, result: _WrappedReturn[ResultType]):
        raise NotImplementedError()

class RPMLimiter(RateLimiter[InputType, ResultType]):
    def __init__(self, max_rate: float, time_period: int = 60):
        self.max_rate = max_rate
        self.time_period = time_period

        self.limiter = AsyncLimiter(
            max_rate=max_rate,
            time_period=time_period
        )

    async def acquire_before(self, payload: InputType):
        await self.limiter.acquire()

    async def acquire_after(self, result: _WrappedReturn[ResultType]):
        pass

    def __repr__(self):
        return f'RPMLimiter(max_rate={self.max_rate}, time_period={self.time_period})'

class TPMLimiter(RateLimiter[InputType, ResultType]):
    def __init__(self, max_rate: float, time_period: int = 60):
        self.max_rate = max_rate
        self.time_period = time_period

        self.limiter = AsyncLimiter(
            max_rate=max_rate,
            time_period=time_period
        )

    async def acquire_before(self, payload: InputType):
        pass

    async def acquire_after(self, result: _WrappedReturn[ResultType]):
        await self.limiter.acquire(result.tokens_used)

    def __repr__(self):
        return f'RPMLimiter(max_rate={self.max_rate}, time_period={self.time_period})'
