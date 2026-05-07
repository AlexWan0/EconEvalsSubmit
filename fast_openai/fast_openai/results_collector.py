from typing import Generic, Literal, Any, Iterable
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import mmap
import threading
import pickle
import os
import io
from tqdm import tqdm, trange
import numpy as np

from .worker import WorkerOutput
from .base import Result, ResultType
from .backward_compatibility import PatchedUnpickler

@dataclass
class ResultsCollectorArgs:
    # TODO: make separate args for each class maybe? at the very least use typedict for kwargs
    how: Literal['memory', 'disk'] = 'memory'
    kwargs: dict[str, Any] = field(default_factory=dict)


class ResultsCollector(ABC, Generic[ResultType]):
    @abstractmethod
    async def _collect_single(
            self,
            res: WorkerOutput[ResultType]
        ):
        raise NotImplementedError()

    async def start_collect(
            self,
            result_input_q: asyncio.Queue[WorkerOutput[ResultType] | None],
            n_workers: int
        ):
        seen_sentinals = 0

        while True:
            task = await result_input_q.get()

            # handle stop sentinal
            if task is None:
                seen_sentinals += 1
                result_input_q.task_done()

                if seen_sentinals >= n_workers:
                    break

                continue
            
            # handle real input
            assert task is not None
            await self._collect_single(task)

            result_input_q.task_done()

    @abstractmethod
    def keys(self) -> list[int]:
        raise NotImplementedError()
    
    def sorted_keys(self) -> list[int]:
        return sorted(self.keys())

    @abstractmethod
    def __getitem__(self, idx: int) -> Result[ResultType]:
        raise NotImplementedError()
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    def close(self):
        return

    @classmethod
    def load(cls, fp: str) -> 'ResultsCollector':
        with open(fp, 'rb') as f_in:
            data: SavedResultsCollector = PatchedUnpickler(f_in).load()
            assert isinstance(data, SavedResultsCollector), f'loaded data from {fp} has class {data.__class__}, expected SavedResultsCollector'

            return data.CollectorClass(**data.init_kwargs)

    @abstractmethod
    def _serialize(self) -> 'SavedResultsCollector':
        raise NotImplementedError()

    def save(self, fp: str):
        with open(fp, 'wb') as f_out:
            pickle.dump(self._serialize(), f_out)

    def to_list(self, enable_pbar: bool = False) -> list[Result[ResultType]]:
        # collect results (in orig order)
        collector_keys = self.sorted_keys()
        results = [
            self[idx]
            for idx in tqdm(collector_keys, disable=not enable_pbar)
        ]

        # check to make sure that there aren't gaps in the indices
        # (the returned list needs to be 1:1 wrt the input tasks)
        # TODO: how to ensure that the input tasks have contiguous indices?
        for expected_idx, found_idx in zip(range(len(collector_keys)), collector_keys):
            assert expected_idx == found_idx
        
        return results

    def iter_list(self, enable_pbar: bool = False) -> Iterable[Result[ResultType]]:
        collector_keys = self.sorted_keys()

        idx_iter = zip(
            trange(len(collector_keys), disable=not enable_pbar),
            collector_keys
        )

        # collect results (in orig order)
        for expected_idx, idx in idx_iter:
            assert expected_idx == idx

            yield self[idx]

    def to_array(self, enable_pbar: bool = False, arr_slice: tuple[int, int] | None = None) -> np.ndarray:
        """
        Loads data directly to numpy array. Specify slice to only load a subset of indices.
        """
        collector_keys = self.sorted_keys()
        item_dim = np.array(self[0].output).shape

        n = len(collector_keys) if arr_slice is None else (arr_slice[1] - arr_slice[0])
        result = np.zeros((n, *item_dim))

        if arr_slice is None:
            idx_iter = zip(
                trange(len(collector_keys), disable=not enable_pbar),
                collector_keys
            )
        else:
            idx_iter = zip(
                trange(*arr_slice, disable=not enable_pbar),
                collector_keys[arr_slice[0]:arr_slice[1]]
            )

        for expected_idx, idx in idx_iter:
            assert expected_idx == idx

            item = np.array(self[idx].output)
            result[idx] = item

        return result

    def iter_array(self, enable_pbar: bool = False, arr_slice: tuple[int, int] | None = None) -> Iterable[np.ndarray]:
        """
        Loads data directly to numpy array. Specify slice to only load a subset of indices.
        """
        collector_keys = self.sorted_keys()

        if arr_slice is None:
            idx_iter = zip(
                trange(len(collector_keys), disable=not enable_pbar),
                collector_keys
            )
        else:
            idx_iter = zip(
                trange(*arr_slice, disable=not enable_pbar),
                collector_keys[arr_slice[0]:arr_slice[1]]
            )

        for expected_idx, idx in idx_iter:
            assert expected_idx == idx

            yield np.array(self[idx].output)

    @classmethod
    def from_args(cls, args: ResultsCollectorArgs) -> 'ResultsCollector':
        if args.how == 'memory':
            return MemResultsCollector(**args.kwargs)
        elif args.how == 'disk':
            # try to load previous incomplete collector checkpoint
            # TODO: probably not good to load here? in_progress_fp is a kwarg of a different class...
            if ('in_progress_fp' in args.kwargs) and (args.kwargs['in_progress_fp'] is not None) and os.path.isfile(args.kwargs['in_progress_fp']):
                print(f"collector in-progress checkpoint found: {args.kwargs['in_progress_fp']}, loading from checkpoint")
                return DiskResultsCollector.load(args.kwargs['in_progress_fp'])

            return DiskResultsCollector(**args.kwargs)
        
        raise ValueError(f'how={args.how} is an invalid collector method for ResultsCollector')
    

@dataclass
class SavedResultsCollector:
    CollectorClass: type[ResultsCollector]
    init_kwargs: dict[str, Any]


class MemResultsCollector(ResultsCollector, Generic[ResultType]):
    def __init__(
            self,
            unordered_data: list[WorkerOutput] = [],
            real_to_stored_keys: dict[int, int] = {}
        ):
        self.unordered_data: list[WorkerOutput] = list(unordered_data)
        
        self.real_to_stored_keys: dict[int, int] = dict.copy(real_to_stored_keys)

    async def _collect_single(
            self,
            res: WorkerOutput[ResultType]
        ):

        real_idx = res.idx
        stored_idx = len(self.unordered_data)
        self.unordered_data.append(res)

        self.real_to_stored_keys[real_idx] = stored_idx
    
    def keys(self) -> list[int]:
        return list(self.real_to_stored_keys.keys())

    def __getitem__(self, idx: int) -> Result[ResultType]:
        if idx not in self.real_to_stored_keys:
            raise KeyError(f"{idx} not found")

        stored_idx = self.real_to_stored_keys[idx]

        res = self.unordered_data[stored_idx]
        
        assert res.idx == idx

        return res.content

    def __len__(self) -> int:
        return len(self.real_to_stored_keys)

    def _serialize(self) -> SavedResultsCollector:
       return SavedResultsCollector(
           CollectorClass=self.__class__,
           init_kwargs={
                'unordered_data': self.unordered_data,
                'real_to_stored_keys': self.real_to_stored_keys
            }
       )


class DiskResultsCollector(ResultsCollector, Generic[ResultType]):
    def __init__(
            self,
            file_path: str | Path = '.cache/results.bin',
            real_idx_to_offset: dict[int, int] = {},
            in_progress_fp: str | None = None,
            in_progress_interval: int = 100_000
        ):

        self.file_path = Path(os.path.abspath(file_path))
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # disk stuff
        self.real_idx_to_offset: dict[int, int] = dict.copy(real_idx_to_offset)
        self.file = open(self.file_path, 'a+b')

        # for writing to disk
        self.write_lock = threading.Lock()

        # for random reads from disk
        self.read_mmap: mmap.mmap | None = None

        # for saving intermediate checkpoints
        self.in_progress_fp = in_progress_fp
        self.in_progress_interval = in_progress_interval
        self.iter_counter = 0

    def set_in_progress_args(
            self,
            in_progress_fp: str | None,
            in_progress_interval: int = 100_000
        ):
        '''
        We don't serialize this, so we should be able to to set it after init.
        '''
        self.in_progress_fp = in_progress_fp
        self.in_progress_interval  = in_progress_interval

    def _records_to_bytes(self, res: WorkerOutput[ResultType]) -> bytes:
        return pickle.dumps(res)

    def _bytes_to_records(self, data: bytes) -> WorkerOutput[ResultType]:
        buf = io.BytesIO(data)
        return PatchedUnpickler(buf).load()

    def _write_record(self, res: WorkerOutput[ResultType]):
        data = self._records_to_bytes(res)

        length = len(data)
        with self.write_lock:
            # get end
            self.file.seek(0, 2)
            offset = self.file.tell()

            # prefix record with length
            self.file.write(length.to_bytes(8, 'big'))

            # write record itself
            self.file.write(data)
            self.file.flush()

            # save offset
            # important that we do this at the very end!
            # because we need to know that any idx recorded here is guaranteed
            # to be readable from disk
            self.real_idx_to_offset[res.idx] = offset

            # checkpointing
            self.iter_counter += 1

            if (self.in_progress_fp is not None) and (self.iter_counter % self.in_progress_interval == 0):
                self.save(self.in_progress_fp)

    async def _collect_single(
            self,
            res: WorkerOutput[ResultType]
        ):

        # run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_record, res)

    def get_mmap(self) -> mmap.mmap:
        # get file length
        self.file.seek(0, 2)
        length = self.file.tell()

        # load mmap
        return mmap.mmap(
            self.file.fileno(),
            length,
            access=mmap.ACCESS_READ
        )

    def keys(self) -> list[int]:
        return list(self.real_idx_to_offset.keys())

    def __getitem__(self, idx: int) -> Result[ResultType]:
        # initialize mmap if we haven't already
        if self.read_mmap is None:
            print('[DiskResultsCollector] initializing mmap')
            self.read_mmap = self.get_mmap()

        if idx not in self.real_idx_to_offset:
            raise KeyError(f"{idx} not found")

        # find offset
        offset = self.real_idx_to_offset[idx]

        # get length prefix
        length_bytes = self.read_mmap[offset:offset + 8]
        length = int.from_bytes(length_bytes, 'big')

        # read data
        start = offset + 8
        data = self.read_mmap[start:start+length]
        
        retrieved_obj = self._bytes_to_records(data)
        assert retrieved_obj.idx == idx

        return retrieved_obj.content

    def __len__(self) -> int:
        return len(self.real_idx_to_offset)

    def close(self):
        if self.read_mmap is not None:
            self.read_mmap.close()

        self.file.close()

    def _serialize(self) -> SavedResultsCollector:
       return SavedResultsCollector(
           CollectorClass=self.__class__,
           init_kwargs={
                'file_path': self.file_path,
                'real_idx_to_offset': self.real_idx_to_offset
            }
       )

    def already_processed(self, idx: int) -> bool:
        return idx in self.real_idx_to_offset
