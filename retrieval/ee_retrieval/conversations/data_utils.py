import pandas as pd
import os
import glob

from ..utils.data import convo_to_str, convo_to_str_human_only
from .dataset_ranges import DatasetPointers


def add_instance_text_col(
        df: pd.DataFrame,
        cols: tuple[str, ...] = ('instance_text', 'instance_text_human_only'),
        dset_ptrs_path: str = 'data/chat_corpora/dataset_pointers.pkl'
    ):
    """Retrieves instances (individual conversations in HF datasets) by their conversation id (`instance_idx`) by adding an additional column to the input dataframe in-place.
    `instance_idx` is how each conversation is identified in the faiss index; this function is used to get the actual content of the conversation.

    Args:
        df (pd.DataFrame): Input dataframe containing conversation id (instance_idx column)
        cols (tuple[str, ...]): Data to add (full conversations or just conversations just containing human turns)
        dset_ptrs_path (str): Filepath for mapping from conversation ids (instance_idx) to actual conversation data (see: ``conversations/dataset_ranges``).
    """

    assert 'instance_idx' in df.columns, df.columns

    data = DatasetPointers.load(dset_ptrs_path)

    df['instance_data'] = data.get_instances_ordered_batch(df['instance_idx'].tolist())

    for c in cols:
        if c == 'instance_text':
            df['instance_text'] = df['instance_data'].apply(
                lambda x: convo_to_str(x.convo)
            )

        elif c == 'instance_text_human_only':
            df['instance_text_human_only'] = df['instance_data'].apply(
                lambda x: convo_to_str_human_only(x.convo)
            )

        else:
            raise ValueError(f'invalid col: {c}')

def get_instance_dedup_size(
        df: pd.DataFrame,
        dset_ptrs_path: str
    ) -> int:
    """Given a dataframe with column `instance_idx` get the number of conversations (instances) while removing the ones that are the same (wrt the conversation text).

    Args:
        df (pd.DataFrame): Input dataframe containing conversation id (instance_idx column)
        cols (tuple[str, ...]): Data to add (full conversations or just conversations just containing human turns)
        dset_ptrs_path (str): Filepath for mapping from conversation ids (instance_idx) to actual conversation data (see: ``conversations/dataset_ranges``).
    
    Returns:
        int: deduplicated size
    """

    assert 'instance_idx' in df.columns, df.columns

    df = df.drop_duplicates(subset=['instance_idx'])

    df = df.copy()

    add_instance_text_col(
        df,
        cols=('instance_text',),
        dset_ptrs_path=dset_ptrs_path
    )

    return len(df['instance_text'].unique())

def get_instance_dedup_size_lst(
        instance_indices: list[int],
        dset_ptrs_path: str
    ) -> int:
    """Given a list of instances (conversation) indices get the number of conversations (instances) while removing the ones that are the same (wrt the conversation text).

    Args:
        df (pd.DataFrame): Input dataframe containing conversation id (instance_idx column)
        cols (tuple[str, ...]): Data to add (full conversations or just conversations just containing human turns)
        dset_ptrs_path (str): Filepath for mapping from conversation ids (instance_idx) to actual conversation data (see: ``conversations/dataset_ranges``).
    
    Returns:
        int: deduplicated size
    """

    df = pd.DataFrame({
        'instance_idx': instance_indices
    })

    return get_instance_dedup_size(
        df,
        dset_ptrs_path=dset_ptrs_path
    )

def get_columns_from_runs(
        cols: list[str],
        fn: str,
        exp_name: str,
        base_dir: str = '.runs'
    ) -> pd.DataFrame:
    """Aggregates all the values for a single column across multiple rounds.

    Args:
        cols (list[str]): columns to get
        fn (str): name of file to get columns from across rounds
        exp_name (str): experiment name
        base_dir (str): base run directory path. Defaults to ``.runs``.
    """

    dfs: list[pd.DataFrame] = []

    for fp in glob.glob(os.path.join(base_dir, exp_name, '*', fn)):
        trial_df = pd.read_pickle(fp)

        dfs.append(trial_df[cols])
    
    return pd.concat(dfs)
