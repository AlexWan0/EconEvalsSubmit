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

from .batched import embed_concat, embed_model_conditional_ensemble, embed_direct
from .embeds import TASK_LIST_PROMPT, CONDITIONAL_TASK_PROMPT


@dataclass
class EmbedArgs:
    """Configuration values for embedding task statements & occupations.
    
    Args:
        n_trials (int): for conversation gen: number of examples to generate per category
        task_model_name (str): for conversation gen: model to use to propose topics
        convo_model_name (str): for conversation gen: model to use to generate convos given topics

        embed_methods (list[str]): methods to use for embedding categories that we then ensemble over
    """
    n_trials: int = 4
    task_model_name: str = 'openai/gpt-4.1'
    convo_model_name: str = 'openai/gpt-4.1-mini'

    embed_methods: list[str] = field(default_factory=lambda: ['concat', 'convos'])

def get_ensemble_embeds(
        input_data: list[dict[str, str]],
        embed_args: EmbedArgs,
        task_col: str = 'category_task',
        occ_col: str = 'category_occ',
    ) -> np.ndarray:
    """Generates embeddings for a list of O*NET task statements/occupations ensembled across multiple embedding methods.

    Args:
        input_data (list[dict[str, str]]): list of dicts each with key task_col and occ_col corresponding to the O*NET task statement and occupation.
        embed_args (EmbedArgs): config parameters of embedding methods (including additional arguments for conversation generation)
        task_col: see ``input_data``
        occ_col: see ``inputt_data``
    
    Returns:
        np.ndarray: normalized embeddings; one for each instance of the ``input_data``
    """

    # TODO: don't remap cols; keep consistent
    dwa_data_mapped = [
        {
            'task': row[task_col],
            'occupation': row[occ_col]
        }
        for row in input_data
    ] # embeds expect task & occupation, exclude dwa

    logger.info(f'get_embeds sample input: {dwa_data_mapped[0]}')

    embeds: list[np.ndarray] = []

    for emb_method in embed_args.embed_methods:
        logger.info(f'embedding with method {emb_method}')

        if emb_method == 'concat':
            res = embed_concat(
                dwa_data_mapped
            )
        
        elif emb_method == 'dwa':
            res = embed_direct(
                [
                    row['dwa'] for row in input_data
                ]
            )

        elif emb_method == 'iwa':
            res = embed_direct(
                [
                    row['iwa'] for row in input_data
                ]
            )

        elif emb_method == 'convos':
            embeds_ens_samp_lst = embed_model_conditional_ensemble(
                dwa_data_mapped,
                task_prompt=TASK_LIST_PROMPT,
                convo_prompt=CONDITIONAL_TASK_PROMPT,
                n_trials=embed_args.n_trials,
                task_model_name=embed_args.task_model_name,
                convo_model_name=embed_args.convo_model_name,
                task_temperature=1.0,
                convo_temperature=1.0
            )
            res = np.vstack(embeds_ens_samp_lst)

        else:
            raise ValueError(f'invalid embedding method: {emb_method}')
    
        embeds.append(res)

    logger.info(f'ensembling over {len(embeds)} embeddings')
    embeds_ens = sum(embeds) / len(embeds)

    # renormalize
    embeds_ens = embeds_ens / np.linalg.norm(embeds_ens, axis=1, keepdims=True)

    return embeds_ens