from dataclasses import dataclass, field
import pandas as pd
import gzip
import pickle
import os
from typing import ClassVar, Literal
import logging
logger = logging.getLogger(__name__)

from typing import Union

from ..conversations import DatasetPointers
from ..utils import convo_to_str, convo_to_str_human_only

from ..onet import (
    ONETDataV1_292,
    ONETWorkbankDesires
)


@dataclass
class DataArgs:
    '''
    General arguments for configuring input/output data.
    '''

    experiment_name: str # name of subfolder in base_path to load/save data for run
    base_path: str = '.runs/' # parent dir of all experiments
    retrieved_df_fn: str = 'retrieved_df.pkl.zst' # fn to store embed-retrieval results
    dataset_ptrs_path: str = 'data/chat_corpora/dataset_pointers.pkl' # location of mapping from conversation ids to datasets
    index_fp: str = 'data/chat_corpora/index' # currently used during retrieval only; location of embedding index data

    data_cols: list[str] = field(
        default_factory=lambda: [
            'instance_text_human_only',
            'category_task_detailed',
            'dwa_detailed',
            'iwa_detailed'
        ]
    ) # currently used during classification only; columns required by prompts in subsequent classification steps

    dwa_summ_method: Literal['disk', 'v1', 'v2', 'v2_nocontrast', 'v2_posonly'] = 'v2' # currently used during classification only; versioning for adding detail to O*NET categories
    iwa_summ_method: Literal['v1'] = 'v1' # currently used during classification only; versioning for adding detail to O*NET categories
    task_summ_method: Literal['v1'] = 'v1' # currently used during classification only; versioning for adding detail to O*NET categories

    onet_args: Union[
        ONETWorkbankDesires,
        ONETDataV1_292
    ] = ONETDataV1_292() # arguments for loading O*NET data


    RETRIEVED_DF_COLS: ClassVar[frozenset[str]] = frozenset([
        'category_task',
        'category_occ',
        'dwa',
        'instance_idx',
        'rank_idx',
    ]) # expected columns for the output of embedding retrieval

    RETRIEVED_DF_COLS_IDENT: ClassVar[frozenset[str]] = frozenset([
        'category_task',
        'category_occ',
        'dwa',
        'instance_idx',
    ]) # expected columns for identifying retrieved category-conversation pair

    CATEGORY_DF_COLS_IDENT: ClassVar[frozenset[str]] = frozenset([
        'category_task',
        'category_occ',
        'dwa',
    ]) # expected columns for identifying a category (both the task statement, occupation, and DWA)

    def build_path(self, subpath: str, is_folder: bool = False) -> str:
        '''
        Given subpath, returns the full path in the experiment directory.
        '''
        fp = os.path.join(self.base_path, self.experiment_name)
        full_path = os.path.join(fp, subpath)

        if is_folder:
            os.makedirs(full_path, exist_ok=True)
        else:
            os.makedirs(fp, exist_ok=True)

        return full_path

    '''
    Getting (possibly processed) O*NET data
    '''
    def get_dwa_df(self) -> pd.DataFrame:
        return self.onet_args.get_dwa_df()

    def get_classified_dwa_df(self) -> pd.DataFrame:
        return self.onet_args.get_classified_dwa_df()

    def get_iwa_summ_map(self) -> dict[str, str]:
        return self.onet_args.get_iwa_summ_map(self.iwa_summ_method)

    def get_dwa_summ_map(self) -> dict[str, str]:
        return self.onet_args.get_dwa_summ_map(self.dwa_summ_method)

    def get_task_summ_map(self) -> dict[tuple[str, str], str]:
        return self.onet_args.get_task_summ_map(self.task_summ_method)

    '''
    Transformations on dataframes
    '''
    def add_instance_text(self, df: pd.DataFrame):
        '''
        Given dataframe, adds column in-place containing the actual conversation data for each row.

        Expects instance_idx column.
        '''
        data = DatasetPointers.load(self.dataset_ptrs_path)

        if 'inst_datum' not in df:
            df['inst_datum'] = data.get_instances_ordered_batch(
                df['instance_idx'].tolist()
            )

        if 'instance_text_human_only' in self.data_cols:
            df['instance_text_human_only'] = df['inst_datum'].apply(
                lambda x: convo_to_str_human_only(x.convo)
            )
        
        if 'instance_text' in self.data_cols:
            df['instance_text'] = df['inst_datum'].apply(
                lambda x: convo_to_str(x.convo)
            )

        df.drop(columns=['inst_datum'], inplace=True)
    
    def add_data_cols(
            self,
            ret_ft_df: pd.DataFrame
        ):
        '''
        Adds additional data columns based on self.data_cols setting. Always adds instance_text and instance_text_human_only cols.

        Modifies dataframe in-place.

        ret_ft_df must always have an instance_idx column (conversation id).
        If dwa_detailed is included in self.data_cols, then dwa column must be present.
        If iwa_detailed is included in self.data_cols, then dwa column must be present (*dwa* is not a typo here).
        If category_task_detailed is included in self.data_cols, then category_occ and category_task must be present.
        '''

        # load data cols; instance text
        self.add_instance_text(ret_ft_df)

        # load data cols; dwa_detailed
        if 'dwa_detailed' in self.data_cols:
            logger.info(f'adding dwa_detailed')
            dwa_summ_map = self.get_dwa_summ_map()
            ret_ft_df['dwa_detailed'] = ret_ft_df['dwa'].apply(
                lambda x: dwa_summ_map[x]
            )

        # load data cols; iwa_detailed
        if 'iwa_detailed' in self.data_cols:
            logger.info(f'adding iwa_detailed')
            iwa_summ_map = self.get_iwa_summ_map()
            ret_ft_df['iwa_detailed'] = ret_ft_df['dwa'].map(iwa_summ_map)
            assert not any(ret_ft_df['iwa_detailed'].isna())

        # load data cols; category_task_detailed
        if 'category_task_detailed' in self.data_cols:
            logger.info(f'adding category_task_detailed')
            task_summ_map = self.get_task_summ_map()
            ret_ft_df['category_task_detailed'] = ret_ft_df.apply(
                lambda row: task_summ_map[(row['category_occ'], row['category_task'])],
                axis=1
            )

        return ret_ft_df


@dataclass
class DataFrameFilter:
    '''
    Dataframe filter for specifying subsets of dataframes based on columns values.
    Looking at only the columns in `col_names`, only the rows with values in `col_values` will be kept in the dataframe. For example, if we set `col_names` as ("ColA",) and `col_values` as {("X",),("Y",)}, then this filter will select for rows where ColA equals to X or Y.

    The purpose is to have a modular and compact way to serialize subsets of dataframes on-disk. It's not the most compact, but if your columns are categories/natural language then it doesn't use too much storage b/c of compression.
    '''
    col_names: tuple[str, ...]
    col_values: frozenset[tuple]

    def __post_init__(self):
        for row in self.col_values:
            assert len(row) == len(self.col_names)

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Use the filter to select the subset of a dataframe.
        '''
        return df[df[list(self.col_names)].apply(
            lambda row: tuple(row) in self.col_values,
            axis=1
        )]

    def save(self, fp: str):
        with gzip.open(fp, 'wb') as f_out:
            pickle.dump(self, f_out)
    
    @classmethod
    def load(cls, fp) -> 'DataFrameFilter':
        with gzip.open(fp, 'rb') as f_in:
            return pickle.load(f_in) # type: ignore

    def __add__(self, other: 'DataFrameFilter') -> 'DataFrameFilter':
        '''
        Add two `DataFrameFilter` objects together. Concatenates `col_values`; needs col_names to be the same (ordering does not need to be the same).

        Reorders to account for differences in ordering.
        '''
        assert set(self.col_names) == set(other.col_names), (self.col_names, other.col_names)
        
        other_indices = [
            self.col_names.index(o_c)
            for o_c in other.col_names
        ]
        target2opos: dict[int, int] = {
            pos: i
            for i, pos in enumerate(other_indices)
        }
        
        return DataFrameFilter(
            col_names=self.col_names,
            col_values=frozenset([
                *self.col_values,
                *[
                    tuple(o_row[target2opos[i]] for i in range(len(o_row)))
                    for o_row in other.col_values
                ]
            ])
        )

    def __len__(self) -> int:
        return len(self.col_values)
