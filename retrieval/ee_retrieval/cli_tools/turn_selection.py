from dataclasses import dataclass
from tyro.conf import OmitArgPrefixes
import pandas as pd
import os
import json
from typing import ClassVar
import logging
logger = logging.getLogger(__name__)

from .base import DataArgs
from .classify import Args as ClassifyArgs
from ..turn_selection import get_turn_subset_selection
from ..utils import full_repr


@dataclass
class SelectTurns:
    '''
    Build turn-selection data from retrieved category-conversation pairs.

    Typical input is the output of:
    `python retrieval/cli/misc.py load-result ...`
    '''
    data_args: OmitArgPrefixes[DataArgs]

    in_fn: str = 'result.pkl'
    out_fn: str = 'with_turns.pkl.zst'

    filter_by_keep: bool = True
    keep_col: str = f'{ClassifyArgs.PRED_KEEP_COL}:so-far'

    overwrite: bool = False

    CALLS_FN: ClassVar[str] = 'calls.log'

    def run(self):
        logger.info(f'called with args: {self.__dict__}')

        in_fp = self.data_args.build_path(self.in_fn, is_folder=False)
        out_fp = self.data_args.build_path(self.out_fn, is_folder=False)

        calls_fp = self.data_args.build_path(self.CALLS_FN)
        with open(calls_fp, 'a+') as f_out:
            f_out.write(json.dumps(
                full_repr(self),
            ) + '\n')

        if os.path.isfile(out_fp) and not self.overwrite:
            logger.info(f'{out_fp} already exists, exiting')
            return

        df = pd.read_pickle(in_fp)

        for req_col in ('instance_idx', 'dwa'):
            assert req_col in df.columns, (req_col, df.columns)

        if self.filter_by_keep:
            if self.keep_col not in df.columns:
                raise ValueError(
                    f'filter_by_keep=True but keep column not found: {self.keep_col}'
                )

            orig_len = len(df)
            df = df[df[self.keep_col]].copy()
            logger.info(f'filtered by {self.keep_col} == True, dropped: {orig_len - len(df)}')

        orig_cols = list(df.columns)
        new_cols = ['category_idx', 'picked_turn', 'convo_subset']

        dwa_summ_map = self.data_args.get_dwa_summ_map()

        turns_df = get_turn_subset_selection(
            df=df,
            dwa_summ_map=dwa_summ_map,
            dset_ptrs_path=self.data_args.dataset_ptrs_path
        )
        logger.info(f'saving output to {out_fp}')
        turns_df[[*orig_cols, *new_cols]].to_pickle(out_fp)
