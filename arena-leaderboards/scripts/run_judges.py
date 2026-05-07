# import os
# os.environ['FAST_OPENAI_CACHE_SIZE_LIMIT_GB'] = '10000'

import argparse
import glob

from arena_hard.configs import ArenaConfig, MultiturnArenaConfig
from arena_hard.dataloading import QuestionData, ModelAnswerMap
from arena_hard.judgement import run_all
import os


def setup_judgement_dir(base_dir: str, arena_cfg: ArenaConfig) -> dict[str, str]:
    judge_model_norm = arena_cfg.judge_endpoint.model_str.replace('/', ':')

    output_files: dict[str, str] = {}
    output_dir = os.path.join(base_dir, "model_judgment", judge_model_norm)
    for model in arena_cfg.model_list:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    return output_files

def get_model_list_from_data(base_dir: str) -> list[str]:
    model_answer_fps = sorted(
        glob.glob(os.path.join(base_dir, "model_answer", "*.jsonl"))
    )
    model_list = [
        os.path.basename(fp).removesuffix('.jsonl')
        for fp in model_answer_fps
    ]

    if len(model_list) == 0:
        raise ValueError(f'no model answers found under {os.path.join(base_dir, "model_answer")}')

    return model_list

if __name__ == '__main__':
    # python arena-leaderboards/scripts/run_judges.py --base-dir .runs/tutorial/predictions-arena-hard

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--setting-file", type=str, default="arena-leaderboards/configs/multiturn.yaml")
    parser.add_argument('--config-type', choices=['single-turn', 'multi-turn'], default='multi-turn')
    parser.add_argument(
        "--baseline-model",
        type=str,
        default='openai:o3-mini-2025-01-31',
        help="Model name used as assistant A baseline for all pairwise judgments.",
    )
    parser.add_argument(
        "--model-list",
        nargs="+",
        default=None,
        help="Optional override for models to judge. If omitted, models are inferred from <base-dir>/model_answer/*.jsonl",
    )

    args = parser.parse_args()

    model_list = args.model_list if args.model_list is not None else get_model_list_from_data(args.base_dir)

    # load data
    if args.config_type == 'single-turn':
        arena_cfg = ArenaConfig.load_yaml(args.setting_file, model_list=model_list)
    elif args.config_type == 'multi-turn':
        arena_cfg = MultiturnArenaConfig.load_yaml(args.setting_file, model_list=model_list)
    else:
        raise ValueError(f'invalid config_type: {args.config_type}')

    model_answer_map = ModelAnswerMap.load_dir(
        os.path.join(args.base_dir, "model_answer")
    )
    
    questions_fp = os.path.join(args.base_dir, "question.jsonl")
    questions = QuestionData.load_auto(questions_fp)

    # get output paths
    output_fp_map = setup_judgement_dir(args.base_dir, arena_cfg)

    # run
    run_all(
        arena_cfg,
        questions,
        model_answer_map,
        output_fp_map,
        baseline_model=args.baseline_model,
    )
