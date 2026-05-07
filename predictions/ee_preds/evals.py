from dataclasses import dataclass, field, asdict, replace
from abc import ABC, abstractmethod
from fast_openai import run_auto, RequestArgs
from fast_openai.base import Result
import pandas as pd
import os
from typing import Generic, TypedDict
import orjson
from hashlib import sha256
import random
from dacite import from_dict
import time

import logging
logger = logging.getLogger(__name__)

from .types import Turn, OutputType
from .utils import df_to_convos, normalize_model_str, convo_to_str


TURN_SUBSET_FN = 'with_turns.pkl.zst'
PREDICTIONS_DIR = 'predictions'
PREDICTIONS_OUT_FORMAT = '{model_name_norm}.jsonl'
PREDICTIONS_ARGS_FORMAT = '{model_name_norm}.args.json'
TOKENS_OUT_FORMAT = '{model_name_norm}_tkns.jsonl'


class Runnable(ABC, Generic[OutputType]):
    @abstractmethod
    def run(self, *args, **kwargs) -> OutputType:
        raise NotImplementedError()

_DRY_RUN_OUTPUT_INIT = lambda: Result('dry run output', None, 0, False, None)

@dataclass
class ModelArgs(Runnable[list[Result[str]]]):
    model_str: str
    max_tokens: int = 16384
    temperature: float = 0.0
    num_workers: int = 128
    request_args: RequestArgs = field(default_factory=lambda: RequestArgs(
        use_cache=True,
        hash_keys=True,
        num_retries=3,
        total_timeout=10 * 60,
        post_timeout=10 * 60,
        connect_timeout=60
    ))

    def run(self, convos: list[list[Turn]], dry_run: bool = False) -> list[Result[str]]:
        if dry_run:
            return [
                _DRY_RUN_OUTPUT_INIT()
                for _ in convos
            ]

        results = run_auto(
            model_inputs=convos,
            full_model_name=self.model_str,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            num_workers=self.num_workers,
            request_args=self.request_args
        )

        return results

class PredRow(TypedDict):
    idx: int
    convo_hash: str
    request_hash: str
    response: str | None
    error: str | None

@dataclass
class SamplingArgs(Runnable[pd.DataFrame]):
    random_state: int = 0
    num_samples: int = 50
    drop_deficient: bool = True
    
    instance_dedup: bool = False

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'category_idx' in df.columns and 'instance_idx' in df.columns, df.columns

        assert df.index.is_unique, 'df.index isn\'t unique!'

        rng = random.Random(self.random_state)
        def _sample(rows: pd.DataFrame) -> pd.DataFrame:
            if len(rows) < self.num_samples:
                if self.drop_deficient:
                    return rows.iloc[0:0]

                logger.warning(f'sampling: {len(rows)} < {self.num_samples}; using smaller')

            to_sample = min(len(rows), self.num_samples)
            return rows.sample(
                n=to_sample,
                random_state=rng.randint(0, 1_000_000)
            )

        def _sample_with_dedup(rows: pd.DataFrame) -> pd.DataFrame:
            _rows = rows.copy()
            _rows['convo_subset_str'] = _rows['convo_subset'].apply(convo_to_str)
            _rows = _rows.drop_duplicates(subset=['convo_subset_str'])

            if len(_rows) < self.num_samples:
                if self.drop_deficient:
                    return rows.iloc[0:0]
                logger.warning(f'sampling after dedup: {len(_rows)} < {self.num_samples}; using smaller')

            to_sample = min(len(_rows), self.num_samples)
            return _rows.sample(
                n=to_sample,
                random_state=rng.randint(0, 1_000_000)
            )

        # df = df.sort_values(by=('category_idx', 'instance_idx'))

        if self.instance_dedup:
            df_sampled = df.groupby('category_idx', group_keys=False).apply(_sample_with_dedup)
        else:
            df_sampled = df.groupby('category_idx', group_keys=False).apply(_sample)

        return df_sampled

TURNS_DF_EXPECT_COLS = [
    'convo_subset',
    'category_idx',
    'instance_idx'
]

@dataclass
class RunArgs(Runnable[None]):
    model_args: ModelArgs
    sampling_args: SamplingArgs = field(default_factory=SamplingArgs)
    dry_run: bool = False
    predictions_folder_name: str = PREDICTIONS_DIR
    base_dir: str = '/PLACEHOLDER'
    df_fn: str = TURN_SUBSET_FN

    @staticmethod
    def hash_convo(convo: list[Turn]) -> str:
        return sha256(orjson.dumps(
            convo,
            option=orjson.OPT_SORT_KEYS
        )).hexdigest()

    def _make_row(
            self,
            idx: int,
            convo: list[Turn],
            res: Result[str],
            req_hash: str
        ) -> PredRow:
        convo_hash = self.hash_convo(convo)

        if res.output is not None:
            return {
                'idx': idx,
                'convo_hash': convo_hash,
                'request_hash': req_hash,
                'error': None,
                'response': res.output,
            }

        else:
            return {
                'idx': idx,
                'convo_hash': convo_hash,
                'request_hash': req_hash,
                'error': str(res.error),
                'response': None,
            }

    def run(self) -> None:
        # setup paths
        assert os.path.isdir(self.base_dir), self.base_dir
        out_dir = os.path.join(
            self.base_dir,
            self.predictions_folder_name,
        )
        out_path = os.path.join(
            out_dir,
            PREDICTIONS_OUT_FORMAT.format(
                model_name_norm=normalize_model_str(self.model_args.model_str)
            )
        )
        args_out_path = os.path.join(
            out_dir,
            PREDICTIONS_ARGS_FORMAT.format(
                model_name_norm=normalize_model_str(self.model_args.model_str)
            )
        )
        tokens_out_path = os.path.join(
            out_dir,
            TOKENS_OUT_FORMAT.format(
                model_name_norm=normalize_model_str(self.model_args.model_str)
            )
        )
        os.makedirs(out_dir, exist_ok=True)

        if os.path.isfile(out_path):
            logger.info(f'{out_path} already exists, skipping')
            # logger.warning(f'{out_path} already exists, going to overwrite')
            return

        # load data
        turns_subset_df: pd.DataFrame = pd.read_pickle(
            os.path.join(self.base_dir, self.df_fn)
        )

        for col in TURNS_DF_EXPECT_COLS:
            assert col in turns_subset_df.columns, (col, turns_subset_df.columns)

        # subsample
        turns_subset_df['_run_index'] = list(range(len(turns_subset_df))) # make sure we have a unique index we can use
        turns_subset_samp_df = self.sampling_args.run(turns_subset_df)
        sampled_indices = turns_subset_samp_df['_run_index']

        convos = df_to_convos(turns_subset_samp_df)

        # run
        run_start_time = time.time()
        result = self.model_args.run(convos)
        run_end_time = time.time()
        run_time = run_end_time - run_start_time
        logger.info(f'model_args.run took {run_time:.3f} seconds')
        assert len(convos) == len(result)

        # get total tokens
        # tokens = sum(
        #     res.tokens_used
        #     for res in result
        #     if res.tokens_used is not None
        # )

        # logger.info(f'{"input" if self.dry_run else "total"} tokens found: {tokens}')

        # export
        req_data = orjson.dumps(
            asdict(self),
            option=orjson.OPT_SORT_KEYS
        )
        with open(args_out_path, 'w') as f_out:
            f_out.write(
                req_data.decode('utf-8')
            )
        
        with open(tokens_out_path, 'w') as f_out:
            for idx, r in zip(sampled_indices, result, strict=True):
                f_out.write(
                    orjson.dumps({
                        'total_tokens': r.tokens_used,
                        'prompt_tokens': r.prompt_tokens,
                        'completion_tokens': r.completion_tokens,
                        'reasoning_tokens': r.reasoning_tokens,
                        'total_time': run_time,
                    }).decode('utf-8') + '\n'
                )

        with open(out_path, 'w') as f_out:
            for idx, r, c in zip(sampled_indices, result, convos, strict=True):
                f_out.write(
                    orjson.dumps(self._make_row(
                        idx=idx,
                        convo=c,
                        res=r,
                        req_hash=sha256(req_data).hexdigest()
                    )).decode('utf-8') + '\n'
                )

@dataclass
class RunFromFileArgs(Runnable[None]):
    filename: str

    # override certain fields through cli
    dry_run: bool | None = field(default=None)
    base_dir: str | None = field(default=None)
    df_fn: str | None = field(default=None)
    predictions_folder_name: str | None = field(default=None)
    max_tokens: int | None = field(default=None)
    num_workers: int | None = field(default=None)
    num_samples: int | None  = field(default=None) # override through cli the number of samples per group
    drop_deficient: bool | None = field(default=None)
    use_cache: bool | None = field(default=None)
    num_retries: int | None = field(default=None)
    total_timeout: int | None = field(default=None)
    post_timeout: int | None = field(default=None)
    connect_timeout: int | None = field(default=None)
    rate_limit_rpm: int | None = field(default=None)
    rate_limit_tpm: int | None = field(default=None)

    def run(self) -> None:
        with open(self.filename, 'r') as f_in:
            data = orjson.loads(f_in.read())

        # load run args from disk
        disk_args = from_dict(data_class=RunArgs, data=data)
        
        override_fields: list[str] = [
            'dry_run',
            'base_dir',
            'df_fn',
            'predictions_folder_name'
        ]

        override_fields_model: list[str] = [
            'max_tokens',
            'num_workers',
        ]

        override_fields_model_requests: list[str] = [
            'use_cache',
            'num_retries',
            'total_timeout',
            'post_timeout',
            'connect_timeout',
            'rate_limit_rpm',
            'rate_limit_tpm',
        ]

        override_fields_sampling: list[str] = [
            'num_samples',
            'drop_deficient'
        ]

        # replace outer run args
        final_args = replace(
            disk_args,
            **{
                k: getattr(self, k)
                for k in override_fields
                if getattr(self, k) is not None
            }
        )

        # replace model args
        final_args = replace(
            final_args,
            model_args=replace(
                final_args.model_args,
                **{
                    k: getattr(self, k)
                    for k in override_fields_model
                    if getattr(self, k) is not None
                }
            )
        )

        # replace model request args
        final_args = replace(
            final_args,
            model_args=replace(
                final_args.model_args,
                request_args=replace(
                    final_args.model_args.request_args,
                    **{
                        k: getattr(self, k)
                        for k in override_fields_model_requests
                        if getattr(self, k) is not None
                    }
                )
            )
        )

        # replace sampling run args
        final_args = replace(
            final_args,
            sampling_args=replace(
                final_args.sampling_args,
                **{
                    k: getattr(self, k)
                    for k in override_fields_sampling
                    if getattr(self, k) is not None
                }
            )
        )

        logger.info(f'running {final_args}')

        return final_args.run()
