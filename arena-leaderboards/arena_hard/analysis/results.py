import pandas as pd
import argparse
import os
import torch
from glob import glob
from tqdm import tqdm
import json
import logging

from .math_utils import one_hot_encode, to_winrate_probabilities, bootstrap_pairwise_model

logger = logging.getLogger(__name__)


def load_judgments(
        judge_names: list[str],
        base_dir: str,
        weight: int=3,
        judgments_dir: str = "model_judgment",
        questions_fn: str = "question.jsonl",
        category_df_fn: str = "category_data.json" # expects cols: subcategory_idx and subcategory_name
    ):

    dfs = []
    for judge_name in judge_names:
        logger.info("loading %s judgments", judge_name)
        target_path = os.path.join(
            base_dir,
            judgments_dir, 
            judge_name, 
            "*.jsonl"
        )

        logger.info("judgment path: %s", target_path)

        dfs.extend([
            pd.read_json(f, lines=True) for f in tqdm(glob(target_path))
        ])
    data = pd.concat(dfs).reset_index(drop=True)
    
    logger.debug("loaded judgment columns: %s", data.columns.tolist())

    # load subcategory data; TODO: move elsewhere/clean this up
    questions_path = os.path.join(base_dir, questions_fn)
    subcat2dwa = pd.read_json(os.path.join(base_dir, category_df_fn)).set_index('subcategory_idx')['subcategory_name'].to_dict()

    uid2subcat_idx: dict[int, int] = {}
    uid2subcat_name: dict[int, str] = {}
    with open(questions_path, 'r') as f_in:
        for line in f_in:
            if len(line.strip()) == 0:
                continue
            
            line_data = json.loads(line)

            uid2subcat_idx[int(line_data['uid'])] = int(line_data['subcategory'])
            uid2subcat_name[int(line_data['uid'])] = subcat2dwa[int(line_data['subcategory'])]

    data['subcategory_idx'] = data['uid'].map(uid2subcat_idx)
    data['subcategory_name'] = data['uid'].map(uid2subcat_name)

    logger.info("subcategory counts:\n%s", data.value_counts('subcategory_name'))

    # if data.model.isin(judge_names).any():
    #     print(f"WARNING: {judge_names} is already in the data. Removing it.")
    #     data = data[~data.model.isin(judge_names)].reset_index(drop=True)

    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    
    logger.info("number of null judgments found: %s", len(data) - len(_data))
    
    # map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )

    _data['scores_tuple'] = _data.games.map(
        lambda x: (tuple(label_to_score[x[1]['score']]), tuple([1 - s for s in label_to_score[x[0]['score']]]))
    )
    
    battles = _data[['uid', 'model', 'category', 'games', 'scores_tuple', 'scores', 'subcategory_idx', 'subcategory_name']].explode('scores').reset_index(drop=True)
       
    return battles


def get_model_style_metadata(base_dir: str, answers_dir: str = "model_answer"):
    model_metadata = {}
    for file in glob(os.path.join(base_dir, answers_dir, "*.jsonl")):
        df = pd.read_json(file, lines=True)
        model_metadata[df.iloc[0]['model']] = df.set_index('uid')['metadata'].to_dict()
        
    return model_metadata


def format_confidence_interval(mean_scores: pd.DataFrame, lower_scores: pd.DataFrame, upper_scores: pd.DataFrame, baseline: str | None = None):
    leaderboard = pd.merge(
        mean_scores, 
        lower_scores, 
        on="model"
    ).merge(
        upper_scores, 
        on="model"
    )
    
    leaderboard["Scores (%)"] = leaderboard["scores"].map(lambda x: round(x * 100, 1))
    
    leaderboard["CI (%)"] = leaderboard.apply(
        lambda row: f"(-{round((row['scores'] - row['lower']) * 100, 1)} / +{round((row['upper'] - row['scores']) * 100, 1)})", 
        axis=1
    )
    
    _leaderboard = leaderboard.rename(
        columns={"model": "Model"}
    ).drop(
        columns=["lower", "upper", "scores"]
    )
    
    if baseline:
        _leaderboard = pd.concat(
            [_leaderboard, pd.DataFrame({"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}, index=[0])]
        )
    
    return _leaderboard.sort_values(by="Scores (%)", ascending=False).reset_index(drop=True)


def print_leaderboard(
        battles: pd.DataFrame,
        category: str,
        baseline_model: str,
        n_bootstrap: int = 100,
    ) -> pd.DataFrame:
    subcategories: list[str] = []
    if "subcategory_name" in battles.columns:
        subcategories = sorted(str(x) for x in battles["subcategory_name"].dropna().unique())

    _battles = battles.drop(columns=['category'])[['model', 'scores']]
    
    # remove model path
    _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
    
    # bootstrap sample 100 means
    bootstraps = pd.concat([
        _battles.groupby("model").sample(frac=1.0, replace=True).groupby("model").mean()
        for _ in range(n_bootstrap)
    ])
    
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    
    _leaderboard = format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline_model)
    
    print(f"##### Category: {category} #####")
    if subcategories:
        print(f"Subcategories: {', '.join(subcategories)}")
    print(_leaderboard.to_string())

    return _leaderboard
        

def print_leaderboard_with_style_features(
        battles: pd.DataFrame,
        category: str,
        baseline_model: str,
        control_features,
        base_dir: str,
        answers_dir: str = "model_answer"
    ):
    subcategories: list[str] = []
    if "subcategory_name" in battles.columns:
        subcategories = sorted(str(x) for x in battles["subcategory_name"].dropna().unique())

    style_metadata = get_model_style_metadata(
        base_dir=base_dir,
        answers_dir=answers_dir
    )
    
    model_features = battles.apply(lambda row: 
        style_metadata[row['model']][row['uid']], 
        axis=1
    ).tolist()
    baseline_features = battles.apply(
        lambda row: style_metadata[baseline_model][row['uid']],
        axis=1
    ).tolist()
    
    # remove model path
    battles['model'] = battles['model'].map(lambda x: x.split('/')[-1])
    
    model_feature_tensor = torch.tensor([
        [v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()]
        for metadata in model_features
    ], dtype=torch.float32)

    baseline_feature_tensor = torch.tensor([
        [v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()]
        for metadata in baseline_features
    ], dtype=torch.float32)
    
    final_feature_tensor = torch.zeros_like(model_feature_tensor)
    final_feature_tensor[:, 0] = (
        model_feature_tensor[:, 0] - baseline_feature_tensor[:, 0]
    ) / (
        model_feature_tensor[:, 0] + baseline_feature_tensor[:, 0]
    )
    
    model_md_density = model_feature_tensor[:, 1:] / (model_feature_tensor[:, :1] + 1)
    baseline_md_density = baseline_feature_tensor[:, 1:] / (baseline_feature_tensor[:, :1] + 1)
    
    assert not model_md_density.isnan().any()
    assert not baseline_md_density.isnan().any()
    
    final_feature_tensor[:, 1:] = (
        model_md_density - baseline_md_density
    ) / (
        model_md_density + baseline_md_density + 1
    )
    
    assert not final_feature_tensor.isnan().any()
    
    normalized_feature_tensor = (
        final_feature_tensor - torch.mean(final_feature_tensor, axis=0) # type: ignore
    ) / torch.std(
        final_feature_tensor, axis=0 # type: ignore
    )
    
    assert not normalized_feature_tensor.isnan().any()
    
    outcomes = torch.tensor(battles.scores.tolist())
    
    assert not outcomes.isnan().any()
    
    model_features, unique_models = one_hot_encode(
        battles.model.tolist(), 
        baseline=baseline_model
    )
    all_features = torch.cat([model_features, normalized_feature_tensor], dim=1)
    
    assert not all_features.isnan().any()
    
    if "length" in control_features and "markdown" in control_features:
        num_features = 4
    elif "length" in control_features:
        all_features = all_features[:, :1]
        num_features = 1
    elif "markdown" in control_features:
        all_features = all_features[:, 1:]
        num_features = 3
    else:
        assert False, "Invalid control features"
        
    coefs, _ = bootstrap_pairwise_model(all_features, outcomes, loss_type="bt")
    
    _coefs = coefs[:, :-num_features]
    
    table = pd.DataFrame(
        columns=unique_models, 
        data=to_winrate_probabilities(
            _coefs, 
            unique_models,
            baseline_model=baseline_model
        ).tolist()
    )
    
    _leaderboard = format_confidence_interval(
        table.quantile(0.5).to_frame("scores").reset_index().rename(columns={"index": "model"}), 
        table.quantile(0.05).to_frame("lower").reset_index().rename(columns={"index": "model"}), 
        table.quantile(0.95).to_frame("upper").reset_index().rename(columns={"index": "model"}), 
    )

    print(f"##### Category: {category} #####")
    if subcategories:
        print(f"Subcategories: {', '.join(subcategories)}")
    print(_leaderboard.to_string())
    logger.info("Feature Coefs: %s", torch.quantile(coefs[:, -num_features:], 0.5, axis=0))  # type: ignore
