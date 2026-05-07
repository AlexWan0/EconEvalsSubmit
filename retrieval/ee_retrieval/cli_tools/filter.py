from dataclasses import dataclass, asdict
from tyro.conf import OmitArgPrefixes
import pandas as pd
from pathlib import Path
import tyro
from typing import ClassVar, Any, Literal
from tqdm import tqdm
import json
import logging
logger = logging.getLogger(__name__)

from .base import DataFrameFilter, DataArgs
from .classify import filter_keep, Args
from ..utils import full_repr
from ..conversations import get_instance_dedup_size_lst


@dataclass
class DWADeficient:
    '''
    Produces dataframe filter to remove category-conversation pairs where we've already retrieved the target number of samples for each category. In this case, it filters out the pairs where the DWA already has at least `min_samples` samples that the entire subsequent classification pipeline has decided to keep.
    '''

    max_round_idx: int
    out_fn: str
    data_args: OmitArgPrefixes[DataArgs]

    min_samples: int = 50
    concat_df_fn: str = 'concat_ret_df.pkl.zst'
    filter_only_files: list[str] | None = None
    n_samples_col: Literal['n', 'n_inst_dedup'] = 'n'
    
    CALLS_FN: ClassVar[str] = 'calls.log'

def run_dwa_deficient(cli_args: DWADeficient):
    logger.info(f'called with args: {cli_args.__dict__}')
    
    # output path
    out_fp = cli_args.data_args.build_path(cli_args.out_fn, is_folder=False)
    concat_out_fp = cli_args.data_args.build_path(cli_args.concat_df_fn, is_folder=False)

    # logs
    calls_fp = cli_args.data_args.build_path(
        DWADeficient.CALLS_FN
    )
    with open(calls_fp, 'a+') as f_out:
        f_out.write(json.dumps(
            full_repr(cli_args),
        ) + '\n')

    # load full retrieved_df
    ret_df = pd.read_pickle(
        cli_args.data_args.build_path(cli_args.data_args.retrieved_df_fn)
    )
    orig_dwas = set(ret_df['dwa'].tolist()) # need for filling in empty rows later

    all_rounds = []
    for round_idx in range(cli_args.max_round_idx + 1):
        round_folder_path = cli_args.data_args.build_path(str(round_idx), is_folder=True)

        ret_ft_df = filter_keep(
            round_folder_path,
            ret_df,
            include_only=frozenset(cli_args.filter_only_files) if cli_args.filter_only_files is not None else None
        )
        
        for col in DataArgs.RETRIEVED_DF_COLS:
            assert col in ret_df.columns, (col, ret_df.columns)

        all_rounds.append(
            ret_ft_df[DataArgs.RETRIEVED_DF_COLS]
        )

    all_ret_ft_df = pd.concat(all_rounds)

    # aggregate across dwa
    agg_cols = [
        c
        for c in DataArgs.RETRIEVED_DF_COLS
        if c != 'dwa'
    ]
    agg_ret_ft_df = all_ret_ft_df.groupby('dwa').agg({
        c: list
        for c in agg_cols
    }).reset_index()

    # fill empty DWAs
    logger.info(f'num orig dwas: {len(orig_dwas)}')
    empty_rows = pd.DataFrame([
        {
            'dwa': title,
            **{
                k: list()
                for k in agg_cols
            }
        }
        for title in (orig_dwas - set(agg_ret_ft_df['dwa']))
    ])
    agg_ret_ft_df = pd.concat([
        agg_ret_ft_df,
        empty_rows
    ])
    assert set(agg_ret_ft_df['dwa']) == orig_dwas, (len(set(agg_ret_ft_df['dwa'])), len(orig_dwas))
    agg_ret_ft_df = agg_ret_ft_df.reset_index(drop=True)

    # add metadata & format
    agg_ret_ft_df['n'] = agg_ret_ft_df['instance_idx'].apply(len)
    agg_ret_ft_df['n_inst_dedup'] = agg_ret_ft_df['instance_idx'].apply(
        lambda indices: get_instance_dedup_size_lst(indices, dset_ptrs_path=cli_args.data_args.dataset_ptrs_path)
    )
    
    agg_ret_ft_df['cat_keep'] = agg_ret_ft_df[cli_args.n_samples_col] < cli_args.min_samples

    logger.info(f'num <{cli_args.min_samples} dwas: {agg_ret_ft_df["cat_keep"].sum()}')

    logger.info(f'saving all clasified to {concat_out_fp}')
    agg_ret_ft_df.to_pickle(concat_out_fp)

    out_filter = DataFrameFilter(
        col_names=('dwa',),
        col_values=frozenset([
            (x,)
            for x in agg_ret_ft_df[agg_ret_ft_df['cat_keep']]['dwa']
        ])
    )

    out_filter.save(out_fp)

def main(cli_args: DWADeficient):
    if isinstance(cli_args, DWADeficient):
        run_dwa_deficient(cli_args)
    
    else:
        raise ValueError('invalid command')
