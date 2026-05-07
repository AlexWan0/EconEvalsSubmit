from dataclasses import dataclass, replace
import faiss
import numpy as np
import os
from uuid import uuid4
from typing import Iterable
import pickle
import glob
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from ...conversations import DatasetRange, DatasetPointers
from ...utils import (
    compress_file_and_remove,
    decompress_file_and_remove,
    compressed_file_path,
    du,
    cached_func
)


EMBED_DIM: int = 1536

@dataclass(frozen=True)
class IndexRetrieval:
    """Wrapper for index retrievals for a single index.
    
    Args:
        index_fp (str): file-path of index that produced this result
        sim_mat (np.ndarray): similarity scores (dot product); (n_queries, top_k)
        _ret_mat (np.ndarray): retrieved indices for *this* index; (n_queries, top_k)
        _agg_idx_start (int): global position for the first item in this index; used as a offset to convert from local positions to global positions
    """
    index_fp: str
    sim_mat: np.ndarray
    _ret_mat: np.ndarray
    _agg_idx_start: int

    @property
    def agg_idx_mat(self) -> np.ndarray:
        """Gets the global conversation id (instance_idx) retrieve results.
        
        Returns:
            np.ndarray: 2D array of ints; (n_queries, top_k) corresponding to the top-k conversation ids (instance_idx values) retrieved from this index
        """
        return self._ret_mat + self._agg_idx_start

@dataclass(frozen=True)
class MultiIndexRetrieval:
    """Wrapper for index retrievals across multiple indices.
    
    Args:
        index_fps (frozenset[str]): file-paths of faiss indices whose results were aggregated to produce this result
        sim_mat (np.ndarray): similarity scores (dot product); (n_queries, top_k)
        agg_idx_mat (np.ndarray): 2D array of ints; (n_queries, top_k) corresponding to the top-k conversation ids (instance_idx values) retrieved from this index
    """
    index_fps: frozenset[str]
    sim_mat: np.ndarray # 2D array of floats; (n_queries, top_k)
    agg_idx_mat: np.ndarray # 2D array of ints; (n_queries, top_k), values are *agg indices*

BLOBS_FOLDER_NAME = 'blobs'
POINTERS_FOLDER_NAME = 'pointers'

@dataclass(frozen=True)
class Index:
    """Class for an individual FAISS index corresponding to a subset of a single dataset.
    
    Args:
        dset_name (str): name of dataset; see: ``conversations/datasets/__init__.py`` for a list of dataset names.
        agg_idx_start (int): first conversation id (instance_idx) that's covered under this index
        agg_idx_stop (int): last conversation id (instance_idx) + 1 that's covered under this index
        index_fp (str): file-path of faiss index contents
    """
    dset_name: str
    agg_idx_start: int
    agg_idx_stop: int # the range is (agg_idx_start, agg_idx_stop]; i.e., not inclusive of agg_idx_stop
    index_fp: str

    @property
    def index_name(self) -> str:
        return f'{self.dset_name}:{self.agg_idx_start}-{self.agg_idx_stop}'

    def _compress(self, threads: int = 6, level: int = 8, print_sizes: bool = False):
        if print_sizes:
            logger.info(f'starting size: {du(self.index_fp):.3f}gb')

        comp_path = compress_file_and_remove(self.index_fp, threads=threads, level=level)
        logger.info(f'compressed to {comp_path}')

        if print_sizes:
            logger.info(f'compressed size: {du(comp_path):.3f}gb')

    def _decompress(self, threads: int = 6):
        decomp_path = decompress_file_and_remove(self.index_fp, threads=threads)
        logger.info(f'decompressed to {decomp_path}')

    def _is_compressed(self):
        comp_path = compressed_file_path(self.index_fp)

        if os.path.isfile(comp_path):
            assert not os.path.isdir(self.index_fp)
            return True
        
        return False
    
    @classmethod
    def from_embeds(
            cls,
            embeds: np.ndarray,
            agg_idx_start: int,
            base_dir: str,
            dset_name: str,
            compress: bool = False,
            compress_threads: int = 6,
            compress_level: int = 8,
        ) -> 'Index':
        """Creates an Index from an array of embeds.

        Args:
            embeds (np.ndarray): 2D array of floats (n_conversations, embed_size)
            agg_idx_start (int): starting conversation id (instance_idx) to use for this faiss index
            base_dir (str): storage location of faiss index data
            dset_name (str): dataset name; should be set based on defined datasets in ``conversations/datasets/__init__.py``.
            compress (bool): whether to compress the index blobs on disk; recommend to set to False. Defaults to ``False``.
            compress_threads (int): threads to use for compression. Defaults to ``6``.
            compress_level (int): level of compression to use (zstandard). Defaults to ``8``.
        
        Returns:
            Index: new faiss Index instance.
        """

        index_fp = os.path.join(base_dir, BLOBS_FOLDER_NAME, str(uuid4()) + '.index')
        assert not os.path.isdir(index_fp)

        # index_fp = os.path.abspath(index_fp)
        os.makedirs(os.path.dirname(index_fp), exist_ok=True)

        n_samples, found_embed_dim = embeds.shape
        agg_idx_stop = agg_idx_start + n_samples

        assert found_embed_dim == EMBED_DIM

        logger.info(f'creating new index where n_samples={n_samples}, embed_dim={EMBED_DIM}, agg_idx range=({agg_idx_start}, {agg_idx_stop}), index_fp={index_fp}')

        # init faiss index
        faiss_index = faiss.IndexFlatIP(EMBED_DIM)

        faiss_index.add(
            embeds
        ) # type: ignore

        # https://github.com/facebookresearch/faiss/issues/3165#issuecomment-1846243552
        # flat IVF lets us use mmap
        # faiss_index = faiss.index_factory(EMBED_DIM, "IVF1,Flat", faiss.METRIC_INNER_PRODUCT)
        # zero = np.zeros((1, EMBED_DIM), dtype=np.float32)
        # faiss_index.train(zero)
        # faiss_index.add(
        #     embeds
        # )

        # write faiss index
        faiss.write_index(
            faiss_index,
            index_fp
        )

        res = Index(
            dset_name=dset_name,
            agg_idx_start=agg_idx_start,
            agg_idx_stop=agg_idx_stop,
            index_fp=index_fp
        )

        # compress if specified
        if compress:
            res._compress(
                threads=compress_threads,
                level=compress_level,
                print_sizes=True
            )

        return res

    def retrieve(
            self,
            query_embeds: np.ndarray,
            top_k: int,
            decompress: bool = True,
            compress: bool = False,
            compress_threads: int = 6,
            compress_level: int = 8,  
        ) -> IndexRetrieval:
        """Retrieves from the Index

        Args:
            query_embeds (np.ndarray): 2D array (n_queries, embed_dim)
            top_k (int): number of samples to retrieve per query
            decompress (bool): try to decompress the index blobs if necessary. Defaults to ``True``.
            compress (bool): whether to compress the index blobs on disk; recommend to set to False. Defaults to ``False``.
            compress_threads (int): threads to use for compression. Defaults to ``6``.
            compress_level (int): level of compression to use (zstandard). Defaults to ``8``.

        Returns:
            IndexRetrieval: wrapper for retrieved conversation ids (instance_idx values)
        """


        n_queries, found_embed_dim = query_embeds.shape
        assert found_embed_dim == EMBED_DIM

        logger.info(f'{n_queries} queries, top_k={top_k}')

        # decompress if necessary
        if decompress and self._is_compressed():
            logger.info('decompressing index')
            self._decompress()

        # retrieve from index
        faiss_index = faiss.read_index(self.index_fp, faiss.IO_FLAG_MMAP)
        sim_val, retrieved = faiss_index.search(query_embeds, top_k)
        assert sim_val.shape == retrieved.shape == (n_queries, top_k)

        # recompress index if specified
        if compress:
            self._compress(
                threads=compress_threads,
                level=compress_level,
                print_sizes=True
            )

        return IndexRetrieval(
            index_fp=self.index_fp,
            sim_mat=sim_val,
            _ret_mat=retrieved,
            _agg_idx_start=self.agg_idx_start
        )

    def save_pointer(self, base_dir: str) -> str:
        """Saves wrapper around faiss Index (this dataclass).
        
        The wrapper gets saved under ``{base_dir}/pointers/...``, whereas the actual faiss index gets saved under ``{base_dir}/blobs/...``.

        Args:
            base_dir (str): directory for index data
        
        Returns:
            str: index_name; also used as filename (w/o the file extension)
        """
        out_path = os.path.join(base_dir, POINTERS_FOLDER_NAME, self.index_name + '.pkl.ptr')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'wb') as f_out:
            pickle.dump(self, f_out)

        return self.index_name

    @classmethod
    def from_pointer(cls, base_dir: str, index_name: str) -> 'Index':
        """Loads wrapper around faiss Index (this dataclass) from disk.

        Args:
            base_dir (str): directory for index data
            index_name (str): which index to load (aka the filename of the pointer w/o the file extension)
        
        Returns:
            Index: index instance loaded from disk
        """
        out_path = os.path.join(base_dir, POINTERS_FOLDER_NAME, index_name + '.pkl.ptr')
        with open(out_path, 'rb') as f_in:
            return pickle.load(f_in)


MULTIINDEX_POINTER_FN = 'multiindex.pkl.ptr'

@dataclass(frozen=True)
class MultiIndex:
    base_dir: str

    def get_indices(self, reset_base_dir: bool = True) -> list[Index]:
        """Gets the individual Index instances that makes up this MultiIndex.

        Retrieval will need to be done for each of the individual Index instances, then aggregated. Returned Index instances are ordered by agg_idx_start. The agg_idx_start, agg_idx_stop must be contiguous. ``agg_idx_start``, ``agg_idx_stop`` corresponding to the global conversation ids (instance_idx values).

        Args:
            reset_base_dir (bool): the index_fp property in Index is defined on creation to be a relative path. However, if we change our paths, the index paths still need to work; enabling this replaces the original index_fp with ones starting from the specified MultiIndex base_dir.
        
        Returns:
            list[Index]: list of Index instances, ordered by agg_idx_start
        """
        indices_unordered = []
        for fn in os.listdir(os.path.join(self.base_dir, POINTERS_FOLDER_NAME)):
            index_name = fn.removesuffix('.pkl.ptr')

            indices_unordered.append(
                Index.from_pointer(
                    self.base_dir,
                    index_name
                )
            )
        
        indices_sorted = list(sorted(indices_unordered, key=lambda x: x.agg_idx_start))
        assert indices_sorted[0].agg_idx_start == 0, f"Shards must start at global index 0; got {indices_sorted[0].agg_idx_start}"

        # check that the starts and stops are contiguous
        prev_stop: int | None = None
        for index in indices_sorted:
            if prev_stop is not None:
                assert index.agg_idx_start == prev_stop

            prev_stop = index.agg_idx_stop

        # maybe reset_base_dir
        if reset_base_dir:
            for i in range(len(indices_sorted)):
                new_index_fp = str((
                    Path(self.base_dir) /
                    BLOBS_FOLDER_NAME /
                    (Path(indices_sorted[i].index_fp).name)
                ))

                indices_sorted[i] = replace(
                    indices_sorted[i],
                    index_fp=new_index_fp
                )

        return indices_sorted

    def retrieve(
            self,
            query_embeds: np.ndarray,
            top_k: int,
            restrict_datasets: list[str] | None = None
        ) -> MultiIndexRetrieval:
        """Retrieve from all the faiss Index instances and aggregate the results.

        Retrieves the ``top_k`` most semantically similar from each Index for all ``query_embeds``, then aggregates.

        Args:
            query_embeds (np.ndarray): 2D array (n_queries, embed_dim)
            top_k (int): number of instances to retrieve per category
            restrict_datasets (list[str] | None): if specified, only considers these datasets during retrieval. See ``conversations/datasets/__init__.py`` for dataset names.
        
        Returns:
            MultiIndexRetrieval: aggregated retrieval results
        """
        
        if restrict_datasets is not None:
            logger.info(f'limiting retrievals to: {restrict_datasets}')

        # 1. Validate dimensions
        n_queries, found_embed_dim = query_embeds.shape
        assert found_embed_dim == EMBED_DIM, (
            f"Expected embed dim {EMBED_DIM}, got {found_embed_dim}"
        )

        # 2. Retrieve from each shard
        retrievals: list[IndexRetrieval] = []
        for idx in self.get_indices():
            if (
                (restrict_datasets is None) or
                (idx.dset_name in restrict_datasets)
            ):
                logger.info(f'retrieving {idx.index_fp}\t{idx.dset_name}')
                retrievals.append(
                    idx.retrieve(query_embeds, top_k)
                )

        # 3. Stack similarity scores and global indices
        sim_all = np.concatenate([r.sim_mat for r in retrievals], axis=1)
        agg_all = np.concatenate([r.agg_idx_mat for r in retrievals], axis=1)
        assert agg_all.shape == sim_all.shape == (n_queries, top_k * len(retrievals))

        # 4. Determine top_k across all shards with a stable sort
        topk_idx = np.argsort(-sim_all, axis=1, kind='stable')[:, :top_k]
        topk_sims = np.take_along_axis(sim_all, topk_idx, axis=1)
        topk_aggs = np.take_along_axis(agg_all, topk_idx, axis=1)

        # 5. Return a unified MultiIndexRetrieval
        return MultiIndexRetrieval(
            index_fps = frozenset(r.index_fp for r in retrievals),
            sim_mat = topk_sims,
            agg_idx_mat = topk_aggs
        )

    def add_dataset(
        self,
        embeds_iter: Iterable[list[float]],
        dset_name: str,
        shard_size: int
    ) -> list[Index]:
        """Adds a dataset to the MultiIndex by streaming from an iterator.

        It's likely that the full dataset cannot be processed in memory, so it lazily shards the set of input embeddings based on the specified ``shard_size``. Each shard holds up to ``shard_size`` embeddings. Each shard results in an Index instance.
        
        Args:
            embeds_iter (Iterable[list[float]]): what to iterate through to get the embeddings, each list of floats should be a single embedding
            dset_name (str): name of dataset to add
            shard_size (int): size of the shard (increase or decrease this based on memory availability). Larger shards decreases inference time, so set this as large as your system can handle (``250_000`` tends to work comfortably on my system of 32gb, but this can probably be set higher).

        Returns:
            list[Index]: Index instances produced for each shard
        """
        # ensure pointers directory exists
        pointers_dir = os.path.join(self.base_dir, POINTERS_FOLDER_NAME)
        os.makedirs(pointers_dir, exist_ok=True)

        # determine next_start from any existing pointer files
        ptr_files = glob.glob(os.path.join(pointers_dir, "*.pkl.ptr"))
        if ptr_files:
            existing = self.get_indices()
            next_start = max(idx.agg_idx_stop for idx in existing)
        else:
            next_start = 0

        new_indices: list[Index] = []
        buffer: list[list[float]] = []

        def _flush():
            nonlocal next_start
            if not buffer:
                return
            logger.info(f'flushing buffer of size {len(buffer)}')
            batch = np.array(buffer)
            idx = Index.from_embeds(batch, next_start, self.base_dir, dset_name)
            idx.save_pointer(self.base_dir)
            new_indices.append(idx)
            next_start += len(buffer)
            buffer.clear()

        # accumulate embeddings, flushing every shard_size
        for emb in embeds_iter:
            assert len(emb) == EMBED_DIM, (
                f"Expected embed dim {EMBED_DIM}, got {len(emb)}"
            )
            buffer.append(emb)
            if len(buffer) >= shard_size:
                _flush()

        # flush any remainder
        _flush()
        return new_indices

    def get_dataset_pointers(self) -> DatasetPointers:
        """Gets the mapping from conversation ids (instance_idx values) to dataset instances.

        Returns:
            DatasetPointers: mapping between conversation ids across chunks to actual dataset instances
        """
        ranges = [
            DatasetRange(
                dset_name=idx.dset_name,
                start=idx.agg_idx_start,
                stop=idx.agg_idx_stop
            )
            for idx in self.get_indices()
        ]

        return DatasetPointers(_idx_dset_ranges=ranges)


@cached_func()
def cached_retrieve(
        multi_index: MultiIndex,
        embeds: np.ndarray,
        top_k: int,
        restrict_datasets: list[str] | None
    ) -> MultiIndexRetrieval:
    """Wrapper for retrieving from a MultiIndex, with outputs cached to disk. See ``MultiIndex.retrieve``.

    Args:
        multi_index (MultiIndex): MultiIndex to retrieve from
        query_embeds (np.ndarray): 2D array (n_queries, embed_dim)
        top_k (int): number of instances to retrieve per category
        restrict_datasets (list[str] | None): if specified, only considers these datasets during retrieval. See ``conversations/datasets/__init__.py`` for dataset names.
    
    Returns:
        MultiIndexRetrieval: aggregated retrieval results
    """
    
    return multi_index.retrieve(
        query_embeds=embeds,
        top_k=top_k,
        restrict_datasets=restrict_datasets
    )
