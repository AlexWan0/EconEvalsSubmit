from dataclasses import dataclass, field

import pandas as pd

from .prompts.interview import (
    INTERVIEW_BETA_MAP,
    INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE,
    build_activity_task_match_prompt,
    build_categorize_prompt,
    bucket_activity_task_match_score,
    parse_activity_task_match_blocks,
    parse_interview_label,
    PROMPT_INTERVIEW_BINARIZE,
)
from .utils import (
    LLMArgs,
    assert_cols,
    lint_convo_column,
    map_score_column,
    run_multiturn_prompts,
    run_prompt_column,
    cached_xlsx_df
)
from typing import Literal


@dataclass
class InterviewArgs:
    # input args
    required_cols: frozenset[str] = frozenset(
        {
            "category_occ",
            "category_task",
            "task_detailed",
            "job_title",
            "years_experience",
            "company_description",
            "query",
            "convo_subset",
            "response",
        }
    )

    # method args
    messages: list[dict] = field(default_factory=lambda: INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE)
    score_map: dict[str, float] = field(default_factory=lambda: INTERVIEW_BETA_MAP)
    interview_llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=1.0,
            max_tokens=32_000,
            num_workers=128,
            cache_flag="",
        )
    )
    # if set, will only run interview scoring prompts starting from this turn index
    # set to 'skip' to skip the interview scoring
    interview_start_from: Literal['skip'] | int | None = None

    categorization_prompt_template: str = PROMPT_INTERVIEW_BINARIZE
    result_processor_prompt_template: str = ""
    result_processor_prompt_col: str = "activity_task_match_prompt"
    result_processor_output_col: str = "interview:activity_task_matches"
    result_processor_response_col: str = "interview:activity_task_match_response"
    result_for_categorization_col: str = "result_for_categorization"
    use_manual_activity_task_scoring: bool = False
    manual_score_col: str = "interview:manual_activity_task_score"
    result_processor_llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=8192,
            num_workers=128,
            cache_flag="",
        )
    )
    activity_task_match_parse_retries: int = 4
    categorize_llm_args: LLMArgs = field(
        default_factory=lambda: LLMArgs(
            model_name="openai/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=128,
            num_workers=128,
            cache_flag="",
        )
    )

    # output args
    output_col: str = "interview:binary"
    beta_col: str = "beta:interview"

    # misc args
    prompt_col: str = "score_prompt"


def build_interview_prompt_table(
    df: pd.DataFrame,
    args: InterviewArgs,
) -> pd.DataFrame:
    assert_cols(df, sorted(args.required_cols))

    out_df = lint_convo_column(df, convo_col="convo_subset")
    return run_multiturn_prompts(
        out_df,
        messages=args.messages,
        llm_args=args.interview_llm_args,
        pbar_name="Interview response turns ({model_name})",
        start_from_turn=args.interview_start_from,
    )


def add_occupation_tasks_column(df):
    """
    Add the full task option list available for each occupation.

    Assumes the input contains all task rows that should be considered for each
    occupation in the current scoring run. Returns occupation_tasks as a
    newline-delimited bullet list for prompt insertion.
    """
    assert_cols(df, ["category_occ", "category_task"])

    unique_tasks = (
        df[["category_occ", "category_task"]]
        .drop_duplicates()
        .sort_values(["category_occ", "category_task"])
    )
    tasks_by_occ = (
        unique_tasks
        .groupby("category_occ")["category_task"]
        .apply(list)
        .to_dict()
    )

    # use the original set of task statemnets instead
    # TODO: don't hard-code
    task_statements_df = cached_xlsx_df('https://www.onetcenter.org/dl_files/database/db_30_2_excel/Task%20Statements.xlsx')
    onet_unique_tasks = (
        task_statements_df[['Title', 'Task']]
        .drop_duplicates()
        .sort_values(['Title', 'Task'])
    )
    onet_tasks_by_occ = (
        onet_unique_tasks
        .groupby('Title')['Task']
        .apply(list)
        .to_dict()
    )

    # check that the tasks for each occupation in the input df are a subset of the onet tasks for that occupation
    for occ, tasks in tasks_by_occ.items():
        onet_tasks = onet_tasks_by_occ.get(occ)
        if onet_tasks is None:
            raise ValueError(f"Occupation {occ} for input df not found in O*NET tasks")
        missing_tasks = set(tasks) - set(onet_tasks)
        if missing_tasks:
            raise ValueError(f"Occupation {occ} for input df has tasks not found in O*NET: {missing_tasks}")

    # use the original set of task statements
    out_df = df.copy()
    out_df["occupation_tasks"] = out_df["category_occ"].apply(
        lambda occ: "\n".join(f"- {task}" for task in onet_tasks_by_occ[occ])
    )
    return out_df


def print_activity_task_matching_inputs(df, args):
    """
    Print prompts sent to the activity-to-task matching model.

    Assumes args.result_processor_prompt_col has already been populated. This is
    intended for auditing v4 preprocessing calls.
    """
    assert_cols(
        df,
        [
            "category_occ",
            "category_task",
            args.result_processor_prompt_col,
        ],
    )

    print("\n" + "=" * 100, flush=True)
    print("ACTIVITY-TO-TASK MATCHING INPUTS", flush=True)
    print("=" * 100, flush=True)
    for idx, row in df.iterrows():
        print("\n" + "-" * 100, flush=True)
        print(f"Row index: {idx}", flush=True)
        print(f"Occupation: {row['category_occ']}", flush=True)
        print(f"Task: {row['category_task']}", flush=True)
        print("-" * 100, flush=True)
        print(row[args.result_processor_prompt_col], flush=True)


def print_activity_task_matching_source_turns(df, final_res_col):
    """
    Print the final interview turn used to build activity-task matching prompts.

    Assumes final_res_col is the last generated interview turn. Returns nothing;
    this is only for auditing the exact conversation content sent onward.
    """
    assert_cols(
        df,
        [
            "category_occ",
            "category_task",
            final_res_col,
        ],
    )

    print("\n" + "=" * 100, flush=True)
    print("LAST INTERVIEW TURNS USED FOR ACTIVITY-TO-TASK MATCHING", flush=True)
    print("=" * 100, flush=True)
    for idx, row in df.iterrows():
        print("\n" + "-" * 100, flush=True)
        print(f"Row index: {idx}", flush=True)
        print(f"Occupation: {row['category_occ']}", flush=True)
        print(f"Task: {row['category_task']}", flush=True)
        print(f"Source column: {final_res_col}", flush=True)
        print("-" * 100, flush=True)
        print(row[final_res_col], flush=True)


def print_demo_prompt_outputs(df):
    """
    Print outputs generated from the chatbot demonstration walkthrough prompt.

    Assumes the v3/v4 interview message template is in use, where the user
    demonstration prompt at message index 12 produces assistant result _result_13.
    """
    result_col = "_result_13"
    assert_cols(
        df,
        [
            "category_occ",
            "category_task",
            result_col,
        ],
    )

    print("\n" + "=" * 100, flush=True)
    print("OUTPUT OF DEMONSTRATION WALKTHROUGH PROMPT (_result_13)", flush=True)
    print("=" * 100, flush=True)
    for idx, row in df.iterrows():
        print("\n" + "-" * 100, flush=True)
        print(f"Row index: {idx}", flush=True)
        print(f"Occupation: {row['category_occ']}", flush=True)
        print(f"Task: {row['category_task']}", flush=True)
        print(f"Source column: {result_col}", flush=True)
        print("-" * 100, flush=True)
        print(row[result_col], flush=True)


def print_activity_task_matching_outputs(df, args):
    """
    Print raw outputs from the activity-to-task matching model.

    Assumes args.result_processor_output_col has already been populated by
    run_prompt_column. This is intended for auditing v4 preprocessing calls.
    """
    assert_cols(
        df,
        [
            "category_occ",
            "category_task",
            args.result_processor_output_col,
        ],
    )

    print("\n" + "=" * 100, flush=True)
    print("XML GENERATED BY PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS", flush=True)
    print("=" * 100, flush=True)
    for idx, row in df.iterrows():
        print("\n" + "-" * 100, flush=True)
        print(f"Row index: {idx}", flush=True)
        print(f"Occupation: {row['category_occ']}", flush=True)
        print(f"Task: {row['category_task']}", flush=True)
        print("-" * 100, flush=True)
        print(row[args.result_processor_output_col], flush=True)


def set_activity_task_match_response_column(df, args):
    """
    Copy the raw activity-to-task matching model text into a stable saved column.

    Assumes run_prompt_column has populated the model output string column for
    args.result_processor_output_col. Returns a copy with the audit column set.
    """
    assert_cols(df, [f"_model_output_str_{args.result_processor_output_col}"])

    out_df = df.copy()
    out_df[args.result_processor_response_col] = out_df[
        f"_model_output_str_{args.result_processor_output_col}"
    ]
    return out_df


def run_result_processor(prompt_df, args, final_res_col):
    """
    Run the v4 activity-to-task matching step before categorization.

    Assumes final_res_col contains the final free-form interview answer and
    args.result_processor_prompt_template is non-empty. Returns prompt_df with
    a processed result column used as the categorization input.
    """
    out_df = add_occupation_tasks_column(prompt_df)
    # print_activity_task_matching_source_turns(out_df, final_res_col)
    out_df[args.result_processor_prompt_col] = out_df.apply(
        lambda row: build_activity_task_match_prompt(
            row,
            row[final_res_col] if pd.notna(row[final_res_col]) else "",
            final_turn_template=args.result_processor_prompt_template,
        ),
        axis=1,
    )

    scored_df = run_prompt_column(
        out_df,
        prompt_col=args.result_processor_prompt_col,
        llm_args=args.result_processor_llm_args,
        output_col=args.result_processor_output_col,
        parser=lambda x: "" if x is None else str(x),
        pbar_name="Interview activity-task matching prompts ({model_name})",
    )
    scored_df = set_activity_task_match_response_column(scored_df, args)

    # print_activity_task_matching_outputs(scored_df, args)
    return scored_df


def build_activity_task_match_retry_prompt(base_prompt, invalid_output, error_message, retry_number):
    """
    Build a corrected XML regeneration prompt for failed activity matching output.

    Assumes base_prompt is the original activity-to-task matching prompt and
    invalid_output is the previous raw model output. Returns a new prompt whose
    changed text avoids returning the same cached malformed output.
    """
    return f"""
The previous response to the prompt below could not be parsed.

Parser error:
{error_message}

Previous invalid response:
```
{invalid_output}
```

Regenerate the response from scratch. Preserve the facts and numeric values from the respondent answer, but strictly follow the requested XML structure and exact field labels. This is retry {retry_number}.

Original prompt:
{base_prompt}
""".strip()


def score_activity_task_match_row(row, args):
    """
    Score one activity-to-task matching XML response.

    Assumes row contains args.result_processor_output_col and category_task.
    Returns the bucket label used by args.score_map.
    """
    return bucket_activity_task_match_score(
        row[args.result_processor_output_col],
        row["category_task"],
    )


def collect_activity_task_match_scores(df, args, row_indices):
    """
    Score selected rows and collect parse failures without hiding them.

    Assumes row_indices are present in df. Returns a score map and a failure map
    so callers can retry only malformed XML generations.
    """
    scores = {}
    failures = {}
    for idx, row in df.loc[row_indices].iterrows():
        try:
            scores[idx] = score_activity_task_match_row(row, args)
        except ValueError as error:
            failures[idx] = error

    return scores, failures


def print_completed_occupation_exposure_estimates(scored_df, args, printed_occupations):
    """
    Print exposure estimates once an occupation has all rows scored.

    Assumes scored_df includes category_occ, category_task, args.output_col, and
    args.result_processor_output_col.
    Only occupations whose rows are fully scored and not yet printed are emitted.
    """
    assert_cols(
        scored_df,
        [
            "category_occ",
            "category_task",
            args.output_col,
            args.result_processor_output_col,
        ],
    )

    for category_occ, occ_df in scored_df.groupby("category_occ"):
        if category_occ in printed_occupations:
            continue
        if occ_df[args.output_col].isna().any():
            continue

        print("\n" + "=" * 100, flush=True)
        print(f"EXPOSURE ESTIMATES COMPLETE FOR OCCUPATION: {category_occ}", flush=True)
        print("=" * 100, flush=True)
        for idx, row in occ_df.iterrows():
            print(
                f"row={idx} | task={row['category_task']} | estimate={row[args.output_col]}",
                flush=True,
            )

        unmatched_counts = [
            count_activities_not_matching_scored_task(
                row[args.result_processor_output_col],
            )
            for _, row in occ_df.iterrows()
        ]
        mean_unmatched = sum(unmatched_counts) / len(unmatched_counts)
        exposure_values = [args.score_map[row[args.output_col]] for _, row in occ_df.iterrows()]
        occupation_exposure = sum(exposure_values) / len(exposure_values)
        print(
            "Occupation exposure measure: "
            f"{occupation_exposure:.4f}",
            flush=True,
        )
        print(
            "Mean unmatched activities per task: "
            f"{mean_unmatched:.2f}",
            flush=True,
        )
        printed_occupations.add(category_occ)


def count_activities_not_matching_scored_task(xml_output):
    """
    Count activities marked as not belonging to the task currently being scored.

    Assumes xml_output follows PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS with
    belongs_to_task_being_scored yes/no per activity.
    """
    activity_blocks = parse_activity_task_match_blocks(xml_output)
    return sum(
        1
        for block in activity_blocks
        if not block["belongs_to_task_being_scored"]
    )


def update_activity_task_match_outputs(scored_df, retry_df):
    """
    Copy regenerated prompt/output columns back onto the main scoring frame.

    Assumes retry_df has the same row index values as scored_df for retried
    rows. Mutates and returns scored_df.
    """
    for col in retry_df.columns:
        if col in scored_df.columns:
            scored_df.loc[retry_df.index, col] = retry_df[col]

    return scored_df


def run_activity_task_match_retry(scored_df, args, base_prompt_col, failures, retry_number):
    """
    Re-prompt rows whose activity-task XML failed parsing.

    Assumes failures maps row indices to parser exceptions. Returns scored_df
    with regenerated raw XML for those rows.
    """
    retry_indices = list(failures.keys())
    retry_df = scored_df.loc[retry_indices].copy()
    retry_df[args.result_processor_prompt_col] = retry_df.apply(
        lambda row: build_activity_task_match_retry_prompt(
            row[base_prompt_col],
            row[args.result_processor_output_col],
            failures[row.name],
            retry_number,
        ),
        axis=1,
    )

    print(
        f"Retrying {len(retry_df)} activity-task matching XML outputs "
        f"(retry {retry_number} of {args.activity_task_match_parse_retries})",
        flush=True,
    )
    retry_df = run_prompt_column(
        retry_df,
        prompt_col=args.result_processor_prompt_col,
        llm_args=args.result_processor_llm_args,
        output_col=args.result_processor_output_col,
        parser=lambda x: "" if x is None else str(x),
        pbar_name="Retrying interview activity-task matching prompts ({model_name})",
    )
    retry_df = set_activity_task_match_response_column(retry_df, args)
    # print_activity_task_matching_outputs(retry_df, args)

    return update_activity_task_match_outputs(scored_df, retry_df)


def run_manual_activity_task_scoring(prompt_df, args):
    """
    Score activity-task XML, retrying malformed generations up to the configured limit.

    Assumes prompt_df already contains activity-task prompts and outputs. Returns
    a scored DataFrame with args.beta_col computed from manual activity matching,
    leaving rows with exhausted parse retries as missing scores.
    """
    scored_df = prompt_df.copy()
    base_prompt_col = f"_{args.result_processor_prompt_col}_base"
    scored_df[base_prompt_col] = scored_df[args.result_processor_prompt_col]
    remaining_indices = list(scored_df.index)
    scored_df[args.output_col] = pd.NA
    printed_occupations = set()

    for attempt in range(args.activity_task_match_parse_retries + 1):
        scores, failures = collect_activity_task_match_scores(
            scored_df,
            args,
            remaining_indices,
        )
        if scores:
            score_series = pd.Series(scores)
            scored_df.loc[score_series.index, args.output_col] = score_series
            # print_completed_occupation_exposure_estimates(
            #     scored_df,
            #     args,
            #     printed_occupations,
            # )

        if not failures:
            scored_df = scored_df.drop(columns=[base_prompt_col])
            scored_df[args.manual_score_col] = scored_df[args.output_col]
            return map_score_column(
                scored_df,
                exposure_col=args.output_col,
                output_col=args.beta_col,
                score_map=args.score_map,
            )

        if attempt >= args.activity_task_match_parse_retries:
            failed_indices = list(failures.keys())
            print(
                "Activity-task XML parsing still failed after "
                f"{args.activity_task_match_parse_retries} retries for "
                f"{len(failed_indices)} rows. Leaving those manual scores missing.",
                flush=True,
            )
            for failed_idx in failed_indices:
                print(
                    f"row={failed_idx} | last parser error={failures[failed_idx]}",
                    flush=True,
                )

            scored_df.loc[failed_indices, args.output_col] = None
            scored_df = scored_df.drop(columns=[base_prompt_col])
            scored_df[args.manual_score_col] = scored_df[args.output_col]
            return map_score_column(
                scored_df,
                exposure_col=args.output_col,
                output_col=args.beta_col,
                score_map=args.score_map,
            )

        scored_df = run_activity_task_match_retry(
            scored_df,
            args,
            base_prompt_col,
            failures,
            retry_number=attempt + 1,
        )
        remaining_indices = list(failures.keys())

    raise AssertionError("Manual activity-task scoring retry loop exited unexpectedly")


def run_interview_scoring(
    df: pd.DataFrame,
    args: InterviewArgs,
) -> pd.DataFrame:
    if not args.interview_start_from == 'skip':
        prompt_df = build_interview_prompt_table(df, args)
    else:
        prompt_df = df.copy()

    # print_demo_prompt_outputs(prompt_df)

    final_res_idx = len(args.messages) - 1

    final_res_col = f"_result_{final_res_idx}"
    if final_res_col not in prompt_df.columns:
        raise ValueError(
            f"Judge turn index {final_res_idx} has no generated result column {final_res_col}"
        )

    result_col = final_res_col
    if args.result_processor_prompt_template:
        prompt_df = run_result_processor(prompt_df, args, final_res_col)
        result_col = args.result_processor_output_col

    if args.use_manual_activity_task_scoring:
        return run_manual_activity_task_scoring(prompt_df, args)

    prompt_df[args.result_for_categorization_col] = prompt_df[result_col]
    prompt_df[args.prompt_col] = prompt_df.apply(
        lambda row: build_categorize_prompt(
            row,
            row[args.result_for_categorization_col]
            if pd.notna(row[args.result_for_categorization_col])
            else "",
            final_turn_template=args.categorization_prompt_template
        ),
        axis=1,
    )

    scored_df = run_prompt_column(
        prompt_df,
        prompt_col=args.prompt_col,
        llm_args=args.categorize_llm_args,
        output_col=args.output_col,
        parser=lambda x: parse_interview_label(x, allowable_entries=tuple(args.score_map.keys())),
        pbar_name="Interview categorize prompts ({model_name})",
    )

    scored_df = map_score_column(
        scored_df,
        exposure_col=args.output_col,
        output_col=args.beta_col,
        score_map=args.score_map,
    )

    return scored_df
