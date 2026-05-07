from dataclasses import dataclass, asdict, field
from tyro.conf import OmitArgPrefixes
import pandas as pd
import os
from fast_openai import run_openai_str, RequestArgs
from typeguard import check_type
import numpy as np
import tyro
import json
from typing import Annotated, ClassVar, Literal
import logging
logger = logging.getLogger(__name__)

from ..embeds.category_embeds import EmbedArgs, get_ensemble_embeds
from ..embeds.faiss_index import MultiIndex, cached_retrieve
from .base import DataArgs
from ..utils import full_repr, cached_func


DATASETS: dict[str, list[str]] = {
    'chat_v1': ['arena_55k', 'arena_100k', 'lmsys_1m', 'wildchat'],
    'chat_v2': ['arena_55k', 'arena_100k', 'lmsys_1m', 'wildchat_4_8', 'arena_140k_fixed'],
}

@dataclass
class RetrievalArgs:
    '''
    Arguments for retrieving from the embedding index.
    '''
    top_k: int = 1000 # retrieve the top-k most semantically similar conversations for each category; you should probably just set this as a high number b/c it's not much more computationally expensive to retrieve more samples per category
    dataset: str = 'chat_v2' # collection of datasets to use (specified by `DATASETS` in `embed_retrieval.py`)

    CALLS_FN: ClassVar[str] = 'calls.log'


@dataclass
class DWA:
    '''
    Arguments for subcommand: retrieving conversations for a random sample of DWAs; uses all task statements for each DWA: we perform semantic-similarity retrieval at the task statement level (i.e., retrieves the top-k conversations for all task statements corresponding to all of the selected DWAs).
    '''
    embed_args: OmitArgPrefixes[EmbedArgs]
    ret_args: OmitArgPrefixes[RetrievalArgs]
    data_args: OmitArgPrefixes[DataArgs]

    n_dwas: int = 100 # number of DWAs to sample, if this is set higher than the number of available DWAs, then all DWAs will be used (i.e., if you want to retrieve over all DWAs, then set this to a high number)
    seed: int = 6 # random seed for sampling

    samp_dwa_fn: str = 'sampled_dwa_df.pkl.zst' # file to save the sampled DWAs
    load_existing_samp_dwa: bool = False # use existing file at `samp_dwa_fn` if it exists

    modif: Literal['none'] = 'none' # (deprecated) modifies the categories before embedding


@dataclass
class Sampling:
    '''
    Arguments for subcommand: (not implemented yet) more flexible sampling (e.g., flat sampling across task statements)
    '''
    embed_args: OmitArgPrefixes[EmbedArgs]
    ret_args: OmitArgPrefixes[RetrievalArgs]
    data_args: OmitArgPrefixes[DataArgs]

    # TODO: not implemented

def build_categories_dwa(cli_args: DWA) -> pd.DataFrame:
    '''
    Gets categories as specified by the `DWA` subcommand cli args. Returns a dataframe where each row is a category (defined by the task statement/occupation and DWA).
    '''
    data_args = cli_args.data_args

    samp_dwa_fp = data_args.build_path(cli_args.samp_dwa_fn, is_folder=False)
    if cli_args.load_existing_samp_dwa and os.path.isfile(samp_dwa_fp):
        return pd.read_pickle(samp_dwa_fp)

    classified_dwa_df = data_args.get_classified_dwa_df()

    # do sampling
    n_sample = min(cli_args.n_dwas, len(classified_dwa_df['DWA Title'].unique()))

    if n_sample < cli_args.n_dwas:
        logger.info(f'WARNING: found only {n_sample} DWAS when target num to sample is {cli_args.n_dwas}; shuffling instead')

    sampled_dwas: list[str] = classified_dwa_df['DWA Title'].drop_duplicates().sample(
        n=n_sample,
        random_state=cli_args.seed
    ).tolist()
    check_type(sampled_dwas, list[str])

    dwa_df = data_args.get_dwa_df()
    sampled_dwa_df = dwa_df[dwa_df['DWA Title'].isin(sampled_dwas)].copy()
    
    # rename cols
    sampled_dwa_df = sampled_dwa_df.rename(columns={
        'Title': 'category_occ',
        'Task': 'category_task',
        'DWA Title': 'dwa',
    })

    # export sampled dwas
    sampled_dwa_df.to_pickle(samp_dwa_fp)

    return sampled_dwa_df


def main(cli_args: DWA | Sampling):
    logger.info(f'called with args: {cli_args.__dict__}')
    
    # output path
    out_fp = cli_args.data_args.build_path(
        cli_args.data_args.retrieved_df_fn,
        is_folder=False
    )

    # logs
    calls_fp = cli_args.data_args.build_path(
        RetrievalArgs.CALLS_FN
    )
    with open(calls_fp, 'a+') as f_out:
        f_out.write(json.dumps(
            full_repr(cli_args),
        ) + '\n')

    # early exit
    if os.path.isfile(out_fp):
        logger.info(f'{out_fp} already exists')
        exit()

    # load index data
    multi_index = MultiIndex(cli_args.data_args.index_fp)

    # load categories df
    if isinstance(cli_args, DWA):
        dwa_df = build_categories_dwa(cli_args)
    
    elif isinstance(cli_args, Sampling):
        raise NotImplementedError()
    
    else:
        raise ValueError(f'invalid subcommand')

    # dwa_df requires category cols
    for col in DataArgs.CATEGORY_DF_COLS_IDENT:
        assert col in dwa_df.columns, (col, dwa_df.columns)

    # col containing the task statement/occupation
    col_task_statement = 'category_task'
    col_occupation = 'category_occ'
    cols_included = set([*DataArgs.CATEGORY_DF_COLS_IDENT, col_task_statement, col_occupation])

    # maybe add iwa data first
    if 'iwa' in cli_args.embed_args.embed_methods:
        dwa_args = cli_args.data_args.onet_args
        iwa_df = dwa_args.get_iwa_df()

        dwa_to_iwa_map = iwa_df.set_index('DWA Title')['IWA Title'] # TODO: WARNING; maybe need both dwa title and id as index
        dwa_df['iwa'] = dwa_df['dwa'].map(dwa_to_iwa_map)

        logger.info(f"added iwa sample: {dwa_df['dwa'].values[0], dwa_df['iwa'].values[0]}")

        cols_included.add('iwa')

    # maybe contrast the task statements
    if isinstance(cli_args, DWA):
        if cli_args.modif == 'none':
            logger.info('no modifications')
        
        else:
            raise ValueError(f'invalid modification: {cli_args.modif}')

    # make embedding data input
    dwa_data: list[dict[str, str]] = dwa_df.apply(
        lambda row: {
            k: row[k]
            for k in cols_included
        },
        axis=1
    ).tolist()

    logger.info(f"sample dwa_data: {dwa_data[0]}")

    # get embeds
    embeds = get_ensemble_embeds(
        dwa_data,
        embed_args=cli_args.embed_args,
        task_col=col_task_statement,
        occ_col=col_occupation
    )
    assert embeds.shape[0] == len(dwa_data), (embeds.shape[0], len(dwa_data))
    assert len(embeds.shape) == 2, embeds.shape

    # do retrieval
    ret_res = cached_retrieve(
        multi_index,
        embeds,
        cli_args.ret_args.top_k,
        DATASETS[cli_args.ret_args.dataset]
    )

    # build dataframe; indices
    sim_val, retrieved = ret_res.sim_mat, ret_res.agg_idx_mat

    ret_rows = []
    for c_idx in range(len(dwa_data)):
        cat_data = dwa_data[c_idx]
        for rank_idx, (s, r_idx) in enumerate(zip(sim_val[c_idx], retrieved[c_idx], strict=True)):
            ret_rows.append({
                'instance_idx': r_idx,
                'rank_idx': rank_idx,
                'sim': s,
                'cat_data': cat_data,
            })
    ret_df = pd.DataFrame(ret_rows)
    
    # build dataframe; category data
    ret_df['category_occ'] = ret_df['cat_data'].apply(lambda x: x['category_occ'])
    ret_df['category_task'] = ret_df['cat_data'].apply(lambda x: x['category_task'])
    ret_df['dwa'] = ret_df['cat_data'].apply(lambda x: x['dwa'])

    # make sure we have the correct cols
    for col in DataArgs.RETRIEVED_DF_COLS:
        assert col in ret_df.columns, (col, ret_df.columns)

    # output
    ret_df.to_pickle(out_fp)
