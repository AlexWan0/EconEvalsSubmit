import argparse
from pathlib import Path

from exposure_lib.prompts.summary import PROMPT_SUMMARY
import pandas as pd

from exposure_lib.scoring_orig_ret import OrigRetArgs, run_orig_ret_scoring, RetrievalArgs
from exposure_lib.utils import (
    add_response_column_from_convo,
    read_llm_args_json,
    read_pickle_gzip,
    save_score_count_plot,
)
from exposure_lib.prompts.residual import (
    PROMPT_RETRIEVAL_RESIDUAL,
    PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012,
    BETA_SCORE_MAP_RESID,
    BETA_SCORE_MAP_RESID_RELABEL_012,
    parse_new_label,
    PROMPT_RETRIEVAL_RESIDUAL_THINKING,
    PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012_2,
    PROMPT_RETRIEVAL_RESIDUAL_THINKING_2,
)
from exposure_lib.prompts.residual_fs import (
    FS_SAMPLES_BY_TOML_FN,
)
from exposure_lib.summary import summarize_conversations, SummaryArgs
from exposure_lib.utils import LLMArgs


def _search_result_dict_to_newsyn_input_df(search_result_obj: object) -> pd.DataFrame:
    if not isinstance(search_result_obj, dict):
        raise ValueError(
            "Input must be a dict with DataFrame values "
            "(e.g., output from synth_gen/scripts/run_search.py)."
        )
    if "verified_df" not in search_result_obj:
        raise ValueError(
            "Input dict is missing key 'verified_df'. "
            f"Found keys: {list(search_result_obj.keys())}"
        )

    verified_df = search_result_obj["verified_df"]
    if not isinstance(verified_df, pd.DataFrame):
        raise ValueError("search result dict['verified_df'] must be a DataFrame")
    if verified_df.empty:
        raise ValueError("search_result['verified_df'] is empty; no passing rows to score")

    required_cols = {"category_occ", "category_task", "query"}
    missing_cols = required_cols - set(verified_df.columns)
    if missing_cols:
        raise ValueError(
            "search_result.verified_df is missing required columns for newsyn scoring: "
            f"{sorted(missing_cols)}"
        )

    out_df = verified_df.dropna(subset=["query"]).copy()
    out_df["convo_subset"] = out_df["query"].apply(
        lambda q: [{"role": "user", "content": str(q)}]
    )

    return out_df


_NONE_FEWSHOT_NAME = "none"


def _resolve_fewshot_name(fewshot_name: str) -> str:
    available = sorted(FS_SAMPLES_BY_TOML_FN)
    name = _NONE_FEWSHOT_NAME if fewshot_name.lower() == _NONE_FEWSHOT_NAME else fewshot_name
    if name != _NONE_FEWSHOT_NAME and name not in FS_SAMPLES_BY_TOML_FN:
        raise ValueError(
            f"Unknown few-shot name '{fewshot_name}'. Expected one of: "
            f"{available + [_NONE_FEWSHOT_NAME]}"
        )
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/search_results.pkl.gz",
        help="Path to dict pickle output by synth_gen/scripts/run_search.py.",
    )
    parser.add_argument(
        "--task-space-path",
        default="data/task_space.pkl.zst",
        help="Path to task-space DataFrame with category_occ/category_task.",
    )
    parser.add_argument(
        "--reference-label-path",
        default="data/eloundou_etal.tsv",
        help="Path to reference label TSV file.",
    )
    parser.add_argument(
        "--response-llm-config-dir",
        default="scripts/model_configs",
        help="Directory containing response LLM config JSON files.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
    )
    parser.add_argument(
        "--prompt-name",
        choices=[
            'basic',
            'thinking',
            'thinking2',
            'relabel',
        ],
        default='basic'
    )
    parser.add_argument(
        "--force-reference-label",
        default=None,
        help="Force a specific reference label (e.g., 'E0', 'E1', 'E2').",
    )
    parser.add_argument(
        "--fewshot-name",
        default=_NONE_FEWSHOT_NAME,
        help=(
            "Few-shot TOML filename (e.g., fs_samples.toml) from "
            "prompts/residual_fs_toml. Use 'none' for no few-shot samples."
        ),
    )
    parser.add_argument(
        "--judge-reasoning",
        choices=[
            'low',
            'medium',
            'high',
        ],
        default='low'
    )

    # models
    parser.add_argument(
        "--llms",
        nargs="+",
        help="One or more response LLM config filenames (or paths).",
        default=[]
    )
    
    # outputs
    parser.add_argument(
        "--output-path",
        default="data/scored_orig_ret2.pkl",
        help="Path to output pickle file.",
    )
    
    # plots
    parser.add_argument(
        "--plot-score-col",
        default="beta:retrieval",
        help="Score column to plot as a count plot.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Path to output score count plot image. By default the path is calculated from the output path by replacing the .pkl extension with _countplot.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loaded_obj = read_pickle_gzip(args.input_path)
    input_df = _search_result_dict_to_newsyn_input_df(loaded_obj)

    task_space_df = pd.read_pickle(args.task_space_path)
    assert isinstance(task_space_df, pd.DataFrame), f"Expected DataFrame at {args.task_space_path}, got {type(task_space_df)}"

    task_space_df = task_space_df.merge(
        input_df[["category_occ", "category_task"]].drop_duplicates(),
        on=["category_occ", "category_task"],
        how="inner",
    )

    reference_label_df = pd.read_csv(args.reference_label_path, sep="\t")
    task_space_df = task_space_df.merge(
        reference_label_df[["Title", "Task", "human_labels"]].drop_duplicates(subset=['Title', 'Task']),
        left_on=["category_occ", "category_task"],
        right_on=["Title", "Task"],
        how="left",
    ).rename(columns={"human_labels": "reference_label"}).drop(columns=["Title", "Task"])

    print(task_space_df.columns)

    # optionally override reference labels with a forced label from args
    if args.force_reference_label:
        print(f"Forcing reference label to '{args.force_reference_label}' for all rows (pre-relabel).")
        task_space_df["reference_label"] = args.force_reference_label

    # relabel reference labels
    if args.prompt_name == 'relabel':
        task_space_df = task_space_df[task_space_df['reference_label'].isin(['E0', 'E1', 'E2'])].copy() # no E3
        task_space_df['reference_label'] = task_space_df['reference_label'].map({
            'E0': 'E0',
            'E2': 'E1',
            'E1': 'E2',
        })
        prompt = PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012
        beta_map = BETA_SCORE_MAP_RESID_RELABEL_012
    
    elif args.prompt_name == 'relabel2':
        task_space_df = task_space_df[task_space_df['reference_label'].isin(['E0', 'E1', 'E2'])].copy() # no E3
        task_space_df['reference_label'] = task_space_df['reference_label'].map({
            'E0': 'E0',
            'E2': 'E1',
            'E1': 'E2',
        })
        prompt = PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012_2
        beta_map = BETA_SCORE_MAP_RESID_RELABEL_012
    
    elif args.prompt_name == 'thinking':
        prompt = PROMPT_RETRIEVAL_RESIDUAL_THINKING
        beta_map = BETA_SCORE_MAP_RESID
    
    elif args.prompt_name == 'thinking2':
        prompt = PROMPT_RETRIEVAL_RESIDUAL_THINKING_2
        beta_map = BETA_SCORE_MAP_RESID
    
    elif args.prompt_name == 'basic':
        prompt = PROMPT_RETRIEVAL_RESIDUAL
        beta_map = BETA_SCORE_MAP_RESID
    
    else:
        raise ValueError(f"Invalid prompt name: {args.prompt_name}")

    fewshot_name = _resolve_fewshot_name(args.fewshot_name)

    # configs
    model_cfgs: list[str] = []
    if len([f for f in args.llms if f.endswith('.json')]) == 0:
        print('using dummy configs since none of the input config files are JSON')
        model_cfgs = [
            "dummy:The LLM performs at the level of a worker on this user request.",
            "dummy:The LLM performs slightly below the level of a worker on this user request.",
            "dummy:The LLM performs significantly below the level of a worker on this user request.",
        ]
        # model_cfgs = [
        #     "dummy:An LLM response that responds adequately to the user request.",
        # ]

    else:
        model_cfgs = [
            f'path:{x}'
            for x in args.llms
            if x.endswith('.json')
        ]

    # run
    input_df_base = input_df.copy()
    scored_dfs: list[pd.DataFrame] = []
    for model_cfg_str in model_cfgs:
        if model_cfg_str.startswith("path:"):
            config_fn = model_cfg_str.removeprefix("path:")
            config_path = Path(config_fn)
            if config_path.parent == Path("."):
                config_path = Path(args.response_llm_config_dir) / config_path

            response_llm_args = read_llm_args_json(str(config_path))
            retrieved_df_model, response_col = add_response_column_from_convo(
                input_df_base.copy(),
                llm_args=response_llm_args,
            )
            retrieved_df_model["response"] = retrieved_df_model[response_col]

        elif model_cfg_str.startswith("dummy:"):
            fixed_str = model_cfg_str.removeprefix("dummy:")
            print(f"Using dummy config for model '{fixed_str}' since no JSON config files were provided.")
            retrieved_df_model = input_df_base.copy()
            retrieved_df_model["response"] = f"The LLM's response has been summarized for brevity:\n<llmResponseSummary>\n{fixed_str}\n</llmResponseSummary>"


        if args.summarize:
            retrieved_df_model = summarize_conversations(
                retrieved_df_model,
                SummaryArgs(
                    prompt=PROMPT_SUMMARY,
                    llm_args=LLMArgs(
                        model_name="openai/gpt-5-mini@reasoning_effort=low",
                        temperature=1.0,
                        max_tokens=4096,
                        num_workers=128,
                    ),
                    output_col="request_summary",
                )
            )
            retrieved_df_model["convo_subset"] = retrieved_df_model["request_summary"].apply(
                lambda x: [{
                    "role": "user",
                    "content": f'The user\'s request has been summarized for brevity:\n<userRequestSummary>\n{x}\n</userRequestSummary>'
                }]
            )

        fs_turns = (
            []
            if fewshot_name == _NONE_FEWSHOT_NAME
            else FS_SAMPLES_BY_TOML_FN[fewshot_name]
        )

        scored_df = run_orig_ret_scoring(
            retrieved_df=retrieved_df_model,
            task_space_df=task_space_df,
            args=OrigRetArgs(
                other_cols=frozenset(['reference_label']),
                prompt_ret=prompt,
                prompt_parser=parse_new_label,
                beta_map=beta_map,
                retrieval_args=RetrievalArgs(
                    use_task_retrieval=True,
                    use_dwa_retrieval=False,
                    coverage_args=None,
                ),
                llm_args=LLMArgs(
                    # model_name=f"openai/gpt-5-mini@reasoning_effort={args.judge_reasoning}",
                    model_name=f"openai/gpt-5.2@reasoning_effort={args.judge_reasoning}",
                    # model_name=f"google/gemini-3.1-pro-preview",
                    temperature=1.0,
                    max_tokens=8192,
                    num_workers=128,
                    cache_flag="conditioned_scoring",
                ),
                few_shot_samples=fs_turns,
            ),
            task_dwa_df=None,
        )
        scored_df["response_model_name"] = "".join(model_cfg_str.split(":")[1:])[:30]
        scored_dfs.append(scored_df)

    scored_df = pd.concat(scored_dfs, ignore_index=True)

    scored_df.to_pickle(args.output_path)

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = args.output_path.rsplit('.', 1)[0] + '_countplot.png'

    save_score_count_plot(
        scored_df,
        score_col=args.plot_score_col,
        output_path=plot_path,
        model_col="response_model_name",
    )
    print(f"Saved {len(scored_df)} rows to {args.output_path}")
    print(f"Saved count plot to {plot_path}")


if __name__ == "__main__":
    main()
