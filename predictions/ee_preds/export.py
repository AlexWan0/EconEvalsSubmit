from dataclasses import dataclass
import glob
from hashlib import sha256
import os
import logging

import orjson
import pandas as pd

from .types import Turn

logger = logging.getLogger(__name__)

QUESTIONS_FN = 'question.jsonl'
MODEL_ANSWERS_DIR = 'model_answer'
CATEGORY_DATA_FN = 'category_data.json'


def fill_missing_response(
        x: str | None,
        missing_response_value: str = 'Server error'
    ) -> str:
    if x is None:
        return missing_response_value
    if not isinstance(x, str):
        raise TypeError(f'expected response to be str | None, got {type(x)}')
    return x


def hash_convo(convo: list[Turn]) -> str:
    convo_minimal = [
        {
            'role': t['role'],
            'content': t['content'],
        }
        for t in convo
    ]
    return sha256(orjson.dumps(
        convo_minimal,
        option=orjson.OPT_SORT_KEYS
    )).hexdigest()


def make_question(row: pd.Series) -> dict:
    return {
        'uid': str(row['idx']),
        'category': 'ei',
        'subcategory': str(row['category_idx']),
        'convo': [
            {'content': x['content'], 'role': x['role']}
            for x in row['convo_subset']
        ],
    }


def make_response(row: pd.Series, model_name: str) -> dict:
    return {
        'uid': str(row['idx']),
        'ans_id': f'{model_name}:{row["idx"]}',
        'model': model_name,
        'messages': row['convo_subset'] + [{
            'content': row['response'],
            'role': 'assistant',
        }],
        'tstamp': 0.0,
        'metadata': {},
    }


@dataclass
class ExportArgs:
    """
    Exports dataset & model predictions into a format for arena-hard-auto.
    """
    base_dir: str
    model_names: list[str] | None = None

    df_fn: str = 'with_turns.pkl.zst'
    predictions_folder_name: str = 'predictions'
    out_folder_name: str = 'predictions-arena-hard'

    include_dwa_cols: bool = True
    strict_hash_check: bool = True
    missing_response_value: str = 'Server error'

    def _get_model_names(self) -> list[str]:
        if self.model_names is not None:
            return self.model_names

        pred_folder = os.path.join(self.base_dir, self.predictions_folder_name)
        fps = sorted(glob.glob(os.path.join(pred_folder, '*.jsonl')))

        model_names: list[str] = []
        for fp in fps:
            fn = os.path.basename(fp)
            if fn.endswith('_tkns.jsonl'):
                continue
            model_names.append(fn.removesuffix('.jsonl'))

        return model_names

    def _read_jsonl(self, fp: str) -> list[dict]:
        rows: list[dict] = []
        with open(fp, 'rb') as f_in:
            for line in f_in:
                if len(line.strip()) == 0:
                    continue
                rows.append(orjson.loads(line))
        return rows

    def run(self):
        turns_df: pd.DataFrame = pd.read_pickle(
            os.path.join(self.base_dir, self.df_fn)
        )
        for col in ('convo_subset', 'category_idx'):
            assert col in turns_df.columns, (col, turns_df.columns)
        turns_df = turns_df.copy()
        turns_df['idx'] = list(range(len(turns_df)))
        turns_df['convo_subset_hash'] = turns_df['convo_subset'].apply(hash_convo)

        out_dir = os.path.join(self.base_dir, self.out_folder_name)
        model_answer_dir = os.path.join(out_dir, MODEL_ANSWERS_DIR)
        os.makedirs(model_answer_dir, exist_ok=True)

        model_names = self._get_model_names()
        logger.info(f'loading {len(model_names)} model(s)')

        for model_name in model_names:
            pred_fp = os.path.join(
                self.base_dir,
                self.predictions_folder_name,
                f'{model_name}.jsonl',
            )
            if not os.path.isfile(pred_fp):
                logger.warning(f'{model_name} data not found, skipping')
                continue

            pred_rows = self._read_jsonl(pred_fp)
            pred_df = pd.DataFrame(pred_rows)
            if len(pred_df) == 0:
                logger.warning(f'{model_name} had empty predictions, skipping')
                continue

            for col in ('idx', 'convo_hash', 'response', 'error'):
                assert col in pred_df.columns, (col, pred_df.columns)

            pred_df['response'] = pred_df['response'].apply(
                lambda x: fill_missing_response(
                    x,
                    missing_response_value=self.missing_response_value
                )
            )
            n_errors = pred_df['error'].notna().sum()
            if n_errors > 0:
                logger.error(f'{model_name} errors: {n_errors} / {len(pred_df)}')
            else:
                logger.info(f'{model_name} errors: {n_errors} / {len(pred_df)}')

            merged = turns_df.merge(
                pred_df,
                on='idx',
                how='right',
            )

            mismatch = merged['convo_subset_hash'] != merged['convo_hash']
            n_mismatch = mismatch.sum()
            if n_mismatch > 0:
                msg = f'{model_name} convo hash mismatches: {n_mismatch}'
                if self.strict_hash_check:
                    raise ValueError(msg)
                logger.warning(msg)
                merged = merged[~mismatch].copy()

            questions = merged.apply(make_question, axis=1).tolist()
            responses = merged.apply(
                lambda row: make_response(row, model_name),
                axis=1
            ).tolist()

            questions_fp = os.path.join(out_dir, QUESTIONS_FN)
            with open(questions_fp, 'wb') as f_out:
                for q in questions:
                    f_out.write(orjson.dumps(q) + b'\n')

            responses_fp = os.path.join(model_answer_dir, f'{model_name}.jsonl')
            with open(responses_fp, 'wb') as f_out:
                for r in responses:
                    f_out.write(orjson.dumps(r) + b'\n')

            export_cols = ['category_idx']
            if self.include_dwa_cols:
                for maybe_col in ('dwa', 'dwa_detailed'):
                    if maybe_col in merged.columns:
                        export_cols.append(maybe_col)

            category_data_df: pd.DataFrame = (
                merged[export_cols]
                .drop_duplicates(subset='category_idx')
                .reset_index(drop=True)
            )
            category_data_df = category_data_df.rename(columns={
                'category_idx': 'subcategory_idx',
                'dwa': 'subcategory_name',
            })
            category_data_df.to_json(
                os.path.join(out_dir, CATEGORY_DATA_FN),
                indent=4
            )
