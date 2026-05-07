from dataclasses import dataclass, field
import tyro
from tyro.conf import OmitArgPrefixes
import zstandard as zstd
import pickle
import pandas as pd
from functools import reduce
import logging
logger = logging.getLogger(__name__)

from .base import DataArgs
from .classify import filter_keep, Args


@dataclass
class LoadClassifiedDWA:
    '''
    Exports the classified DWAs to out_fn.
    '''

    out_fn: str

    data_args: OmitArgPrefixes[DataArgs]

    def run(self):
        out_path = self.data_args.build_path(self.out_fn)

        self.data_args.get_classified_dwa_df().to_pickle(
            out_path
        )

@dataclass
class LoadDWASummMap:
    '''
    Exports the detailed DWA data to `out_fn`.
    '''

    out_fn: str

    data_args: OmitArgPrefixes[DataArgs]

    def run(self):
        out_path = self.data_args.build_path(self.out_fn)

        res = self.data_args.get_dwa_summ_map()

        with zstd.open(out_path, 'wb') as f_out:
            pickle.dump(res, f_out)

@dataclass
class LoadIWASummMap:
    '''
    Exports the detailed IWA data to `out_fn`.
    '''

    out_fn: str

    data_args: OmitArgPrefixes[DataArgs]

    def run(self):
        out_path = self.data_args.build_path(self.out_fn)

        res = self.data_args.get_iwa_summ_map()

        with zstd.open(out_path, 'wb') as f_out:
            pickle.dump(res, f_out)


@dataclass
class LoadResult:
    '''
    Exports the kept category-conversation pairs across all rounds up to `max_round_idx`.
    '''

    out_fn: str
    max_round_idx: int

    data_args: OmitArgPrefixes[DataArgs]

    filter_only_files: list[str] | None = None # whitelist files to include
    filter_exclude_files: list[str] = field(default_factory=list) # blacklist files to include

    filename_order: list[str] | None = None # order to load files
    
    def run(self):
        # output path
        out_fp = self.data_args.build_path(self.out_fn, is_folder=False)

        # load full retrieved_df
        ret_df = pd.read_pickle(
            self.data_args.build_path(self.data_args.retrieved_df_fn)
        )

        all_rounds: list[pd.DataFrame] = []
        for round_idx in range(self.max_round_idx + 1):
            round_folder_path = self.data_args.build_path(str(round_idx), is_folder=True)

            ret_ft_df = filter_keep(
                round_folder_path,
                ret_df,
                include_only=frozenset(self.filter_only_files) if self.filter_only_files is not None else None,
                dont_include=frozenset(self.filter_exclude_files),
                filtered_output=False,
                fn_order=self.filename_order
            )

            # get prediction columns
            pred_cols = [
                c
                for c in ret_ft_df.columns
                if c.startswith(Args.PRED_KEEP_COL + ':')
            ]

            keep_cols = [*DataArgs.RETRIEVED_DF_COLS, *pred_cols]

            # validate that cols exist
            for col in keep_cols:
                assert col in ret_ft_df.columns, (col, ret_ft_df.columns)

            # remove rows for which no predictions have been made
            has_na_df = ret_ft_df.isna()
            is_all_na = reduce(
                lambda x, y: x & y,
                [
                    has_na_df[c]
                    for c in pred_cols
                    if not c.endswith(':so-far')
                ]
            )
            
            logger.info(f'dropping {sum(is_all_na)} rows')
            ret_ft_df = ret_ft_df[~is_all_na]

            # add
            all_rounds.append(
                ret_ft_df[keep_cols]
            )

        all_ret_ft_df = pd.concat(all_rounds)

        all_ret_ft_df.to_pickle(out_fp)
