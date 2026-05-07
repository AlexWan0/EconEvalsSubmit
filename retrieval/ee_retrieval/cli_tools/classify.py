from dataclasses import dataclass, asdict, field
from tyro.conf import OmitArgPrefixes
import pandas as pd
from pathlib import Path
from typing import Literal, ClassVar
import glob
import os
from functools import reduce
import json
import logging
logger = logging.getLogger(__name__)

from .base import DataFrameFilter, DataArgs
from ..utils import full_repr
from ..pipeline import apply_categorization, apply_hw_pipeline, apply_general_qual_pipeline


@dataclass
class Args:
    '''
    Arguments for all classification steps.
    '''
    k_start: int # only performs classification from the k_start to k_stop most semantically similar conversation for each category
    k_stop: int # only performs classification from the k_start to k_stop most semantically similar conversation for each category
    round_idx: int # sequential id of the round number
    df_filters: list[str] = field(default_factory=list) # filename(s) of dataframe filters to use
    
    PRED_KEEP_COL: ClassVar[str] = 'pred_keep'
    CALLS_FN: ClassVar[str] = 'calls.log'

@dataclass
class Categorization:
    '''
    Arguments for subcommand: categorization.
    '''
    method_id: int # numerical id of classification method to use (see: ee_retrieval/pipeline/categorization/pipeline.py)
    
    args: OmitArgPrefixes[Args]
    data_args: OmitArgPrefixes[DataArgs]

    prompts_version: str = 'v10' # version of prompts to use (see: ee_retrieval/pipeline/categorization/prompt_versions)
    keep_method: Literal['ABC', 'ABCD', 'A'] = 'ABC'

    @property
    def out_fn(self) -> str:
        return f'method_id-{self.method_id}.pkl.zst'

@dataclass
class Quality:
    '''
    Arguments for subcommand: quality classification.
    '''
    filter_name: Literal['hw', 'general'] # whether to perform homework classification or general classification
    
    args: OmitArgPrefixes[Args]
    data_args: OmitArgPrefixes[DataArgs]

    qual_prompts_version: str = 'v5' # version of prompts to use (see: ee_retrieval/pipeline/qual_general/prompts.py)
    hw_prompts_version: str = 'v6' # version of prompts ot use (see: ee_retrieval/pipeline/qual_homework/prompts_versions)

    @property
    def out_fn(self) -> str:
        return f'filter_name-{self.filter_name}.pkl.zst'


def filter_keep(
        pred_folder: str,
        pairs_df: pd.DataFrame,
        include_only: frozenset[str] | None = None,
        dont_include: frozenset[str] = frozenset(),
        filtered_output: bool = True,
        fn_order: list[str] | None = None
    ) -> pd.DataFrame:
    '''
    Load all the predictions from the previous rounds (by looking for *.pkl.zst files in `pred_folder`) and remove the pairs (the rows) that have been dropped in previous round (i.e., only keeps the ones that haven't been predicted on yet OR all previous rounds have resulted in a KEEP prediction).

    pred_folder: path to search for data from rounds.
    pairs_df: each row corresponds to a conversation-category pair; result is a subset of this dataframe
    include_only; dont_include: whitelist/blacklist filepaths (`dont_include` is checked after `include_only`)
    filtered_output: if set to false, only add the prediction columns & don't do the filtering of rows (pairs)
    fn_order: order to load the previous round outputs
    '''
   
    fps = glob.glob(os.path.join(pred_folder, '*.pkl.zst'))

    if len(fps) == 0:
        return pairs_df
    
    # filter whitelist
    if include_only is not None:
        fps = list(filter(
            lambda _fp: os.path.basename(_fp) in include_only,
            fps,
        ))
        logger.info(f'include_only filtered to {fps}')
    
    # filter blacklist
    fps = list(filter(
        lambda _fp: os.path.basename(_fp) not in dont_include,
        fps,
    ))
    logger.info(f'dont_include filtered to {fps}')

    # reorder
    if fn_order is not None:
        fps = list(sorted(
            fps,
            key=lambda _fp: fn_order.index(os.path.basename(_fp))
        ))
        logger.info(f'reordered fps to: {fps}')

    # aggregate
    pred_cols: list[str] = []
    for i, fp in enumerate(fps):
        fn = os.path.basename(fp)
    
        logger.info(f'aggregating on {fn}')

        p_df = pd.read_pickle(fp)
        new_col = f'{Args.PRED_KEEP_COL}:{i}'

        assert new_col not in pairs_df.columns, pairs_df.columns

        pairs_df = pairs_df.merge(
            p_df[[*DataArgs.RETRIEVED_DF_COLS_IDENT, Args.PRED_KEEP_COL]],
            on=[*DataArgs.RETRIEVED_DF_COLS_IDENT],
            how='left'
        ).rename(
            columns={
                Args.PRED_KEEP_COL: new_col
            }
        ) # TODO: add validation

        if filtered_output:
            orig_len = len(pairs_df)
            pairs_df = pairs_df.dropna(subset=new_col)
            logger.info(f'df dropped na ({fp}): {orig_len - len(pairs_df)}')

        pred_cols.append(new_col)

    pairs_df[f'{Args.PRED_KEEP_COL}:so-far'] = reduce(
        lambda x, y: x & y,
        [
            pairs_df[c]
            for c in pred_cols
        ]
    )

    if filtered_output:
        return pairs_df[pairs_df[f'{Args.PRED_KEEP_COL}:so-far']]
    
    return pairs_df


def load_df(args: Args, data_args: DataArgs, out_folder_path: str) -> pd.DataFrame:
    '''
    Loads the actual category-conversation pairs that we want to perform prediction over (based on CLI args).
    Also "populates" the columns with actual data (e.g., adds conversation text, adds dwa_detailed, etc.)
    '''
    # load data
    ret_df = pd.read_pickle(
        data_args.build_path(data_args.retrieved_df_fn)
    )
    for col in DataArgs.RETRIEVED_DF_COLS:
        assert col in ret_df.columns, (col, ret_df.columns)

    ret_ft_df = ret_df

    # maybe filter categories; # TODO: this doesn't nest the filters correctly
    for filt_idx, filt_fn in enumerate(args.df_filters):
        orig_len = len(ret_ft_df)
        df_filter = DataFrameFilter.load(
            data_args.build_path(filt_fn)
        )
        
        ret_ft_df = df_filter.filter_df(ret_df)

        logger.info(f'filter by df filter ({filt_idx}; {filt_fn}), droppped: {orig_len - len(ret_ft_df)}')

    # filter k range
    orig_len = len(ret_ft_df)
    ret_ft_df = (
        ret_ft_df
        .groupby(list(DataArgs.CATEGORY_DF_COLS_IDENT), group_keys=False)
        .apply(
            lambda g: g.sort_values(by='rank_idx').iloc[args.k_start : args.k_stop]
        )
        .reset_index(drop=True)
    )
    logger.info(f'filter by k range, dropped: {orig_len - len(ret_ft_df)}')

    ret_ft_df = ret_ft_df.copy()

    # filter by prev preds
    orig_len = len(ret_ft_df)
    ret_ft_df = filter_keep(
        out_folder_path,
        ret_ft_df
    ).copy()
    logger.info(f'filter by prev preds, dropped: {orig_len - len(ret_ft_df)}')

    data_args.add_data_cols(ret_ft_df)

    return ret_ft_df

def run_categorization(ret_ft_df: pd.DataFrame, cli_args: Categorization) -> pd.DataFrame:
    '''
    Performs categorization on conversation-cateogry pairs.
    '''
    args, data_args = cli_args.args, cli_args.data_args

    # run
    classif_col = apply_categorization(
        ret_ft_df,
        prompts_version=cli_args.prompts_version,
        run_cfg_id=cli_args.method_id,
        keep_method=cli_args.keep_method
    )

    ret_ft_df[args.PRED_KEEP_COL] = ret_ft_df[classif_col]

    return ret_ft_df

def run_quality(ret_ft_df: pd.DataFrame, cli_args: Quality) -> pd.DataFrame:
    '''
    Performs quality classification on conversation-cateogry pairs.
    '''
    args = cli_args.args
    
    if cli_args.filter_name == 'hw':
        hw_col = apply_hw_pipeline(
            ret_ft_df,
            prompts_version=cli_args.hw_prompts_version
        )

        ret_ft_df[args.PRED_KEEP_COL] = ~ret_ft_df[hw_col]
    
    elif cli_args.filter_name == 'general':
        gen_col = apply_general_qual_pipeline(
            ret_ft_df,
            key=cli_args.qual_prompts_version
        )

        ret_ft_df[args.PRED_KEEP_COL] = ~ret_ft_df[gen_col]
    
    else:
        raise ValueError(f'invalid quality filter_name: {cli_args.filter_name}')

    return ret_ft_df

def run_null(ret_ft_df: pd.DataFrame, cli_args: Quality | Categorization) -> pd.DataFrame:
    '''
    When the number of conversation-categories pairs is zero, this function is used to add the expected prediction column but without any actual values.
    '''

    logger.info(f'running null classif')

    args = cli_args.args

    if len(ret_ft_df) > 0:
        raise ValueError(f'input df has {len(ret_ft_df)} rows, but expected zero')

    ret_ft_df[args.PRED_KEEP_COL] = []

    return ret_ft_df

def main(cli_args: Categorization | Quality):
    logger.info(f'called with args: {cli_args.__dict__}')

    args, data_args = cli_args.args, cli_args.data_args

    # setup output path
    out_folder_path = Path(
        data_args.build_path(str(args.round_idx), is_folder=True)
    )

    # logs
    with open(out_folder_path / Args.CALLS_FN, 'a+') as f_out:
        f_out.write(json.dumps(
            full_repr(cli_args),
        ) + '\n')

    # early exit
    if (out_folder_path / cli_args.out_fn).exists():
        logger.info(f"{out_folder_path / cli_args.out_fn} already exists, exiting")
        exit()

    ret_ft_df = load_df(args, data_args, str(out_folder_path))

    logger.info(f'{len(ret_ft_df["dwa"].unique())} dwas loaded')

    if len(ret_ft_df) == 0:
        ret_ft_df = run_null(ret_ft_df, cli_args)
        ret_ft_df.to_pickle(out_folder_path / cli_args.out_fn)
        exit()

    # run command
    if isinstance(cli_args, Categorization):
        ret_ft_df = run_categorization(ret_ft_df, cli_args)
        ret_ft_df.to_pickle(out_folder_path / cli_args.out_fn)
    
    elif isinstance(cli_args, Quality):
        ret_ft_df = run_quality(ret_ft_df, cli_args)
        ret_ft_df.to_pickle(out_folder_path / cli_args.out_fn)

    else:
        raise ValueError(f'invalid command')

