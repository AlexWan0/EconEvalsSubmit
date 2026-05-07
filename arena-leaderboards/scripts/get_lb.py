import argparse
import pandas as pd
import os
import json
from functools import reduce
import re
import logging

from arena_hard.analysis import load_judgments, print_leaderboard, print_leaderboard_with_style_features

logger = logging.getLogger(__name__)


def extract_assistant_B(text: str) -> str:
    pattern = r"<\|The Start of Assistant B's Answer\|>\n(.*)\n<\|The End of Assistant B's Answer\|>"

    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)
    
    raise ValueError(f'no assistant response found for {text}')

def convo_to_str(convo: list[dict[str, str]]) -> str:
    convo_turns = []
    for turn in convo:
        assert isinstance(turn, dict), convo

        assert turn['role'] == 'user' or turn['role'] == 'assistant'

        convo_turns.append(
            f'Human: {turn["content"]}'
            if turn["role"] == "user" else f'Assistant: {turn["content"]}'
        )

    return '\n\n'.join(convo_turns)

def fill_missing_uid_model(df: pd.DataFrame, uid_col: str = "uid", model_col: str = "model") -> pd.DataFrame:
    uids = df[uid_col].unique()
    models = df[model_col].unique()

    full_index = pd.MultiIndex.from_product([uids, models], names=[uid_col, model_col]) # type: ignore
    df_full = df.set_index([uid_col, model_col]).reindex(full_index).reset_index()
    
    return df_full

def make_lb_data(
        lb_df: pd.DataFrame,
        battles_df: pd.DataFrame,
        model_answer_df: pd.DataFrame,
        base_dir: str,
        questions_fn: str = 'question.jsonl',
    ) -> dict:
    
    lb_df = lb_df.rename(columns={
        'Model': 'model',
        'Scores (%)': 'score'
    })

    with open(os.path.join(base_dir, questions_fn)) as f_in:
        # cols: uid, category, subcategory, convo
        questions_df = pd.DataFrame([json.loads(line) for line in f_in])
    
    battles_df['uid'] = battles_df['uid'].astype(str)
    questions_df = questions_df[questions_df['uid'].isin(battles_df['uid'].tolist())]
    logger.info('len(questions_df): %s', len(questions_df))
    
    uid2subcategory_name = battles_df.drop_duplicates(subset=['uid', 'subcategory_name']).set_index('uid')['subcategory_name']
    questions_df['subcategory_name'] = questions_df['uid'].map(uid2subcategory_name)
    questions_df['convo_str'] = questions_df['convo'].apply(convo_to_str)

    responses: dict[str, list[str]] = {}
    battles_df_dedup = battles_df.drop_duplicates(subset=['model', 'uid', 'scores_tuple'])

    # load judge/model response rows (new)
    battles_df_dedup = battles_df_dedup.reset_index(drop=True)
    battles_df_dedup['judgements_str'] = battles_df_dedup['games'].apply(
        lambda games: ('\n' + '-' * 80 + '\n').join([
            f'(Judge output {i + 1}):\n' + g['judgment']['answer'] for i, g in enumerate(games)
        ]),
    )

    def scores_str(scores_tuple: tuple[tuple[int], tuple[int]]) -> str:
        return f'Avg. score: {(sum(scores_tuple[0]) + sum(scores_tuple[1])) / 2} / 3.0\tForward score: {sum(scores_tuple[0])}\tReverse score: {sum(scores_tuple[1])}'

    battles_df_dedup['judge_answer'] = battles_df_dedup['games'].apply(
        lambda games: extract_assistant_B(games[1]['prompt'][1]['content'])
    )

    battles_df_dedup['judge_info'] = battles_df_dedup.apply(
        lambda row: {
            'judge_scores_raw': {
                'avg': (sum(row["scores_tuple"][0]) + sum(row["scores_tuple"][1])) / 2,
                'forward_score': sum(row["scores_tuple"][0]),
                'backward_score': sum(row["scores_tuple"][1]),
            },
            'judge_scores': scores_str(row["scores_tuple"]),
            'judge': str(row["judgements_str"]),
            'judge_answer': str(row["judge_answer"])
        },
        axis=1
    )

    assert len(model_answer_df[['uid', 'model']].drop_duplicates()) == len(model_answer_df)
    model_answer_df['convo'] = model_answer_df['messages'].apply(lambda x: convo_to_str(x[:-1]))

    logger.debug('len(battles_df_dedup) before uid/model dedup: %s', len(battles_df_dedup))
    battles_df_dedup = battles_df_dedup.drop_duplicates(['uid', 'model']) # TODO: ??
    battles_df_dedup = battles_df_dedup.merge(
        model_answer_df[['uid', 'model', 'convo', 'answer']],
        on=['uid', 'model'],
        how='left'
    )

    logger.debug('len(battles_df_dedup) after merge: %s', len(battles_df_dedup))
    battles_df_dedup = fill_missing_uid_model(battles_df_dedup)
    logger.debug('len(battles_df_dedup) after fill_missing_uid_model: %s', len(battles_df_dedup))

    model_info_keys = battles_df_dedup['judge_info'].dropna().iloc[0].keys()
    battles_df_dedup['answer'] = battles_df_dedup['answer'].fillna('')
    battles_df_dedup['judge_info'] = battles_df_dedup['judge_info'].apply(
        lambda d: {**({k: '' for k in model_info_keys} if not isinstance(d, dict) else d)}
    )
    logger.debug('rows with null judge_info:\n%s', battles_df_dedup[battles_df_dedup['judge_info'].isna()])

    battles_df_dedup['model_info'] = battles_df_dedup.apply(
        lambda row: {
            'model_answer': row['answer'],
            **row['judge_info']
        },
        axis=1
    )

    model_info_lst_map = battles_df_dedup.sort_values(['model', 'uid']).groupby('model').agg({
        'model_info': list
    })['model_info'].to_dict()

    convo_lst = battles_df_dedup.drop_duplicates('uid').sort_values('uid')['convo'].to_list()

    lengths = [*(len(x) for x in model_info_lst_map.values()), len(convo_lst)]
    assert len(set(lengths)) == 1, lengths

    return {
        'rows': lb_df.to_dict(orient='records'),
        'prompts': convo_lst,
        'response_data': model_info_lst_map,
        'responses': []
    }

def _norm_fn(orig: str) -> str:
    orig, ext = os.path.splitext(orig)
    orig = orig.replace(' ', '_')
    return ''.join(c for c in orig if (c == '/' or c == '_' or c.isalnum())) + ext

def extend_to_json(fp: str, data: list):
    if not os.path.isfile(fp):
        with open(fp, 'w') as f_out:
            json.dump(data, f_out, indent=4)
        
        return

    with open(fp, 'r') as f_in:
        orig_data = json.load(f_in)
    
    assert isinstance(orig_data, list)

    orig_data.extend(data)

    with open(fp, 'w') as f_out:
        json.dump(orig_data, f_out, indent=4)

def filter_battles_server_error(
        battles_all: pd.DataFrame,
        server_error_text: str = 'Server error'
    ) -> pd.DataFrame:
    _SERVER_ERROR_STRS = (
        f"<|The Start of Assistant A's Answer|>\n{server_error_text}\n<|The End of Assistant A's Answer|>",
        f"<|The Start of Assistant B's Answer|>\n{server_error_text}\n<|The End of Assistant B's Answer|>"
    )

    battles_all['prompt_1'] = battles_all['games'].apply(
        lambda g: g[0]['prompt'][1]['content'] if g[0] is not None else None
    )
    battles_all['prompt_2'] = battles_all['games'].apply(
        lambda g: g[1]['prompt'][1]['content'] if g[1] is not None else None
    )

    battles_all['has_server_error'] = reduce(
        lambda x, y: x | y,
        (
            *(battles_all['prompt_1'].str.contains(se_str, regex=False) for se_str in _SERVER_ERROR_STRS),
            *(battles_all['prompt_2'].str.contains(se_str, regex=False) for se_str in _SERVER_ERROR_STRS),
        )
    )
    
    logger.info(
        'has_server_error: %s / %s',
        sum(battles_all['has_server_error']),
        len(battles_all['has_server_error'])
    )

    return battles_all[~battles_all['has_server_error']].copy()

if __name__ == "__main__":
    # python arena-leaderboards/scripts/get_lb.py --base-dir .runs/tutorial/predictions-arena-hard  --stratify-col subcategory_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument(
        "--baseline-model",
        type=str,
        default='openai:o3-mini-2025-01-31',
        help="Model name used as baseline when computing leaderboard scores.",
    )
    parser.add_argument("--judge-names", "-j", nargs="+", default=["openai:gpt-4.1-mini"])
    parser.add_argument("--control-features", "-f", nargs="+", default=[])
    parser.add_argument("--category", "-c", nargs="+", default=['ei'])
    parser.add_argument("--stratify-col", type=str, default=None)
    parser.add_argument('--questions-fn', type=str, default='question.jsonl')
    parser.add_argument('--category-fn', type=str, default='category_data.json')
    parser.add_argument('--judgments-folder', type=str, default='model_judgment')
    parser.add_argument('--answers-folder', type=str, default='model_answer')
    parser.add_argument('--server-error-text', type=str, default='Server error')
    parser.add_argument('--filter-error', action='store_true', default=False)
    parser.add_argument('--export-without-prompts', action='store_true', default=False) # don't export prompts & response data
    
    parser.add_argument('--output_fn', type=str, default='{strat_val}.json')
    args = parser.parse_args()

    # load battles
    battles_all = load_judgments(
        args.judge_names,
        base_dir=args.base_dir,
        judgments_dir=args.judgments_folder,
        questions_fn=args.questions_fn,
        category_df_fn=args.category_fn,
    )

    # load model answers
    model_df_parts = []
    for model_name in battles_all['model'].unique():
        model_fp = os.path.join(args.base_dir, args.answers_folder, model_name + '.jsonl')
        with open(model_fp) as f_in:
            # cols: uid, category, subcategory, convo
            model_df_parts.append(pd.DataFrame([
                json.loads(line) for line in f_in
            ]))

    model_answer_df = pd.concat(model_df_parts)
    model_answer_df['answer'] = model_answer_df['messages'].apply(lambda x: x[-1]['content'])

    model_answer_df = fill_missing_uid_model(model_answer_df)
    model_answer_df = model_answer_df.fillna({
        'answer': ''
    })
    model_answer_df['messages'] = model_answer_df['messages'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # maybe filter
    if args.filter_error:
        logger.info('filtering battles for server error')
        battles_all = filter_battles_server_error(
            battles_all,
            server_error_text=args.server_error_text
        )

    # setup dirs
    lb_out_folder_path = os.path.join(
        args.base_dir,
        'lb_out',
    )

    os.makedirs(lb_out_folder_path, exist_ok=True)

    added_fns: list[str] = []

    for category in args.category:
        # print(f'-------------------- {category} --------------------')

        assert category in battles_all.category.unique(), f"Invalid category: {category}"
        
        battles_cat: pd.DataFrame = battles_all[battles_all.category == category].reset_index(drop=True) # type: ignore

        if args.stratify_col is not None and args.stratify_col not in battles_cat.columns:
            raise ValueError(f"stratify-col: \"{args.stratify_col}\" not in cols: {battles_cat.columns.tolist()}")
        
        battles_subcats: list[tuple[str, pd.DataFrame]] = []
        if args.stratify_col is not None:
            for strat_val, group_rows in battles_cat.groupby(args.stratify_col):
                battles_subcats.append((strat_val, group_rows))
        else:
            battles_subcats.append(('ALL', battles_cat))
        
        for strat_val, battles_subcat in battles_subcats:
            logger.info('stratify value: %s', strat_val)

            if args.control_features:
                logger.info('control features: %s', args.control_features)
                
                print_leaderboard_with_style_features(
                    battles_subcat, 
                    category,
                    args.baseline_model,
                    args.control_features,
                    base_dir=args.base_dir,
                    answers_dir=args.answers_folder
                )

            else:
                lb_df = print_leaderboard(
                    battles_subcat,
                    category,
                    baseline_model=args.baseline_model,
                )

                # save response data
                lb_data = make_lb_data(
                    lb_df,
                    battles_subcat,
                    model_answer_df,
                    base_dir=args.base_dir,
                    questions_fn=args.questions_fn,
                )

                if args.export_without_prompts:
                    lb_data['prompts'] = []
                    lb_data['response_data'] = {}

                lb_fn = _norm_fn(args.output_fn.format(strat_val=strat_val))

                out_fp = os.path.join(
                    lb_out_folder_path,
                    lb_fn
                )
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)

                with open(out_fp, 'w') as f_out:
                    json.dump(lb_data, f_out, indent=4)
                
                added_fns.append(lb_fn)

            # print('-' * 40)
            logger.info('')
    
    # for lb export
    added_ids = [
        fn.removesuffix('.json')
        for fn in added_fns
    ]

    manifest_data = [
        {'id': a_id, 'label': a_id, 'file': a_fn}
        for a_id, a_fn in zip(added_ids, added_fns)
    ]

    # append to manifest data
    manifest_fp = os.path.join(lb_out_folder_path, 'manifest.json')
    extend_to_json(
        manifest_fp,
        manifest_data
    )

    # append to groups data
    groups_fp = os.path.join(lb_out_folder_path, 'groups.json')
    extend_to_json(
        groups_fp,
        [{
            'id': f'{args.stratify_col if args.stratify_col is not None else "ALL"}',
            'label': f'{args.stratify_col if args.stratify_col is not None else "ALL"}',
            'members': added_ids
        }]
    )
