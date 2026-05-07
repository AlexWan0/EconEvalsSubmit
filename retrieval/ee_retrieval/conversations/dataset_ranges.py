from dataclasses import dataclass
import pickle
import numpy as np
from tqdm import tqdm
import gc
import logging
logger = logging.getLogger(__name__)

from ..utils.data import Turn, convo_to_str
from .datasets import ConvoDataset, dset_class_map


@dataclass
class DatasetRange():
    """Mappping from a range of conversation ids (instance_idx values) to dataset names for single chunk of a dataset.

    Args:
        dset_name (str): Dataset name, defined in ``conversations/datasets/__init__.py``.
        start (int): Start of conversation id (instance_idx) range.
        stop (int): Non-inclusive stop of conversation id (instance_idx) range.
    """
    dset_name: str
    start: int
    stop: int

@dataclass
class RetrievedInstance:
    """Wrapper for actual conversation data from underlying dataset.
    """
    instance_data: list[Turn]
    instance_text: str
    dset_name: str

@dataclass
class InstanceDatum:
    """Wrapper for actual conversation data from underlying dataset.
    """
    instance_idx: int
    source: str
    convo: list[Turn]

# in-memory cache of loaded datasets
initialized_dsets: dict[str, ConvoDataset] = {}

def clear_initialized():
    initialized_dsets.clear()
    gc.collect()

def get_instance(
        ranges: list[DatasetRange],
        idx: int,
        ignored_dsets: set = set()
    ) -> RetrievedInstance:
    """Get instance data using the conversation id mappings, a target index, while possibly ignoring certain datasets.
    """

    dset_name, offset_idx = get_dset_from_range(ranges, idx)
    
    if dset_name in ignored_dsets:
        # print(f'WARNING: ignoring dataset, returning placeholder instance')
        return RetrievedInstance(
            [],
            'placeholder',
            dset_name
        )

    if dset_name not in initialized_dsets:
        logger.info(f'initializing dataset: {dset_name}')
        initialized_dsets[dset_name] = dset_class_map[dset_name](support_get_item=True)

    convo_data = initialized_dsets[dset_name].get_convo(idx - offset_idx)

    return RetrievedInstance(
        instance_data=convo_data,
        instance_text=convo_to_str(convo_data),
        dset_name=dset_name
    )


def get_dset_from_range(ranges: list[DatasetRange], idx: int) -> tuple[str, int]:
    """Get dataset name using the conversation id mappings and a target index.
    """

    found_range = None
    for r in ranges:
        if idx >= r.start and idx < r.stop:
            found_range = r
            break
    
    if found_range is None:
        raise ValueError(f'no valid range found for idx: {idx}')
    
    dset_name = found_range.dset_name
    
    # get offset; start val from earliest added range w/ matching dset
    for r in ranges:
        if r.dset_name == dset_name:
            return dset_name, r.start
    
    raise ValueError('no valid range found; searching for start')

@dataclass
class DatasetPointers:
    """Class wrapper for a list of DatasetRanges, each of which describe a mapping between a list of conversation ids (instance_idx values) and dataset names for a chunk of samples.

    The conversation id (instance_idx) is used to uniquely identify each conversation. To map that to actual conversation data, we use the mappings to recover the dataset name and the original index in the dataset. We can then use that to recover the original conversation data.
    """
    _idx_dset_ranges: list[DatasetRange]

    def get_instances_ordered_batch(self, indices: list[int]) -> list[InstanceDatum]:
        """Map a list of conversation ids (instance_idx values) to actual conversation data (InstanceDatum).

        Performs this more efficiently sorting indices by datasets so we aren't switching between disk accesses for large datasets.

        Args:
            indices (list[int]): global conversation indices (instance_idx values)
        
        Returns:
            list[InstanceDatum]: actual conversation data for those indices
        """
        vals_arr = np.array(indices)
        vals_argsort = np.argsort(vals_arr)
        vals_sorted = vals_arr[vals_argsort]

        result: list[None | InstanceDatum] = [None] * len(indices)

        for rev_rank, val in enumerate(tqdm(vals_sorted, desc='get_instances_batch')):
            orig_pos = vals_argsort[rev_rank]

            assert vals_arr[orig_pos] == val
            ret_inst = get_instance(
                self._idx_dset_ranges,
                val,
                ignored_dsets=set()
            )

            result[int(orig_pos)] = InstanceDatum(
                instance_idx=val,
                source=ret_inst.dset_name,
                convo=ret_inst.instance_data
            )

        for r, expected_val in zip(result, indices):
            assert r is not None
            assert expected_val == r.instance_idx

        return result # type: ignore

    def get_instance(self, idx: int) -> InstanceDatum:
        """Map conversation id (instance_idx) to actual conversation data (InstanceDatum).

        Args:
            indices (int): conversation id (instance_idx)
        
        Returns:
            InstanceDatum: actual conversation data for this id
        """

        ret_inst = get_instance(
            self._idx_dset_ranges,
            idx,
            ignored_dsets=set()
        )

        return InstanceDatum(
            instance_idx=idx,
            source=ret_inst.dset_name,
            convo=ret_inst.instance_data
        )

    def save(self, out_fp: str = 'data/chat_corpora/dataset_pointers.pkl'):
        """Serialize 
        """
        with open(out_fp, 'wb') as f_out:
            pickle.dump(self, f_out)
    
    @classmethod
    def load(cls, out_fp: str = 'data/chat_corpora/dataset_pointers.pkl') -> 'DatasetPointers':
        with open(out_fp, 'rb') as f_in:
            return pickle.load(f_in)
