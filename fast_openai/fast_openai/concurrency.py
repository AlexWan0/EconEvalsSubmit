import asyncio
from typing import Collection, Generic
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from tqdm import tqdm
from dataclasses import dataclass

from .clients import APIClient, RequestArgs, oai_tokens_tracker, OutputValidation, RateLimiter, RPMLimiter, TPMLimiter
from .base import Result, InputType, ResultType
from .sized_iterable import MaybeSizedIter, SizedIterable
from .worker import WorkerInput, WorkerOutput, Worker
from .results_collector import (
    ResultsCollector,
    ResultsCollectorArgs,
    DiskResultsCollector
)


def _exit_on_error(t: asyncio.Task):
    '''
    Callback to shut everything down on error.
    From: https://stackoverflow.com/a/62589864
    '''
    if not t.cancelled() and t.exception() is not None:
        import traceback, sys
        e = t.exception()
        traceback.print_exception(
            None,
            e,
            e.__traceback__ if e is not None else None
        )
        sys.exit(1)


class ConcurrentRunner(Generic[InputType, ResultType]):
    def __init__(
            self,
            client: APIClient[InputType, ResultType],
            n_workers: int,
            request_args: RequestArgs,
            pbar_name: str = 'running'
        ):
        # None sentinal used to signal no more tasks (retrieved by worker)
        self.input_q: asyncio.Queue[WorkerInput[InputType] | None] = asyncio.Queue(maxsize=n_workers * 2)
        
        # None sentinal used to signal worker is exiting (retrieved by results collector)
        self.result_q: asyncio.Queue[WorkerOutput[ResultType] | None] = asyncio.Queue(maxsize=n_workers * 2)
        
        self.client = client
        self.n_workers = n_workers
        self.request_args = request_args
        self.pbar_name = pbar_name

    async def run(
            self,
            inputs: MaybeSizedIter[InputType],
            collector_args: ResultsCollectorArgs = ResultsCollectorArgs(),
            output_validation: OutputValidation[ResultType] = OutputValidation[ResultType]()
        ) -> ResultsCollector[ResultType]: # TODO: set type depending on args

        # init collector
        collector = ResultsCollector.from_args(collector_args)

        # init networking stuff
        connector = TCPConnector(limit=max(1024, self.n_workers))
        timeout = ClientTimeout(
            total=self.request_args.total_timeout,
            connect=self.request_args.connect_timeout
        )

        # init rate limiting
        rate_limiters: tuple[RateLimiter, RateLimiter] = (
            RPMLimiter(
                max_rate=self.request_args.rate_limit_rpm / 30,
                time_period=2 # seconds
            ),
            TPMLimiter(
                max_rate=self.request_args.rate_limit_tpm / 30,
                time_period=2 # seconds
            )
        )

        # TODO: originally did the MaybeSizedIter stuff for this, but just going with this fix for now...
        pbar = tqdm(total=len(inputs) if hasattr(inputs, '__len__') else None) # type: ignore
        async with ClientSession(connector=connector, timeout=timeout) as session:
            # init results collector
            collector_task = asyncio.create_task(
                collector.start_collect(self.result_q, self.n_workers)
            )
            collector_task.add_done_callback(_exit_on_error)

            # init workers
            workers = [
                Worker(
                    self.input_q,
                    self.result_q,
                    self.client,
                    session,
                    pbar,
                    self.pbar_name,
                    rate_limiters=rate_limiters
                )
                for _ in range(self.n_workers)
            ]

            worker_tasks = []
            for w in workers:
                task = asyncio.create_task(w())
                task.add_done_callback(_exit_on_error)
                worker_tasks.append(task)

            # send tasks
            for idx, payload in enumerate(inputs):
                # if we're resuming from a broken "to_disk" run, then
                # we should skip the already covered inputs
                if isinstance(collector, DiskResultsCollector) and collector.already_processed(idx):
                    pbar.update(1)
                    pbar.set_description(self.pbar_name + ' (already found on disk)')
                    continue

                pbar.set_description(self.pbar_name)

                # put in input queue
                await self.input_q.put(
                    WorkerInput(
                        idx=idx,
                        payload=payload,
                        request_args=self.request_args,
                        output_validation=output_validation
                    )
                )

            # send stop signal
            for _ in range(self.n_workers):
                await self.input_q.put(None)

            # wait until workers finished
            await self.input_q.join()
            await asyncio.gather(*worker_tasks)

            # wait until results collector finished
            await self.result_q.join()
            await collector_task

            pbar.close()
            collector.close()

            # print token usage
            oai_tokens_tracker.display_tokens_used()

            return collector

    async def run_to_list(
            self,
            inputs: MaybeSizedIter[InputType],
            collector_args: ResultsCollectorArgs = ResultsCollectorArgs(),
            output_validation: OutputValidation[ResultType] = OutputValidation[ResultType]()
        ) -> list[Result[ResultType]]:

        return (await self.run(inputs, collector_args, output_validation)).to_list()
