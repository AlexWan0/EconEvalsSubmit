from typing import Generic
from aiohttp import ClientSession
import asyncio
from dataclasses import dataclass
from tqdm import tqdm
from aiolimiter import AsyncLimiter

from .clients import APIClient, RequestArgs, OutputValidation, RateLimiter
from .base import Result, InputType, ResultType



@dataclass
class WorkerInput(Generic[InputType]):
    idx: int
    payload: InputType
    request_args: RequestArgs
    output_validation: OutputValidation

@dataclass
class WorkerOutput(Generic[ResultType]):
    idx: int
    content: Result[ResultType]

class Worker(Generic[InputType, ResultType]):
    def __init__(
            self,
            input_q: asyncio.Queue[WorkerInput[InputType] | None],
            result_q: asyncio.Queue[WorkerOutput[ResultType] | None],
            client: APIClient[InputType, ResultType],
            session: ClientSession,
            pbar: tqdm,
            pbar_name: str = 'running',
            rate_limiters: tuple[RateLimiter, ...] = tuple(),
        ):
        self.input_q = input_q
        self.result_q = result_q
        self.client = client
        self.session = session
        self.pbar = pbar
        self.pbar_name = pbar_name
        self.rate_limiters = rate_limiters
    
    async def __call__(self):
        while True:
            task = await self.input_q.get()

            # handle stop sentinal
            if task is None:
                self.input_q.task_done()
                break

            # make api call
            res = await self.client.process_input(
                task.payload,
                self.session,
                task.request_args,
                output_validation=task.output_validation,
                rate_limiters=self.rate_limiters
            )

            if res.from_cache:
                self.pbar.set_description(f'{self.pbar_name} (from cache)')
            else:
                self.pbar.set_description(self.pbar_name)

            await self.result_q.put(WorkerOutput(
                idx=task.idx,
                content=res
            ))
            self.input_q.task_done()

            self.pbar.update(1)
        
        await self.result_q.put(None)


