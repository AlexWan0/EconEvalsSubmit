from typing import Callable, Literal
from fast_openai import run_auto_str, RequestArgs
import re
import pandas as pd
import argparse


def _get_tasks_dwas_df(version: str = '29.2') -> pd.DataFrame:
    version_url_str = version.replace('.', '_')

    tasks_dwas_df = pd.read_excel(f'https://www.onetcenter.org/dl_files/database/db_{version_url_str}_excel/Tasks%20to%20DWAs.xlsx')
    
    return tasks_dwas_df

def _full_model_name(model_name: str, reasoning_effort: str | None) -> str:
    if model_name.startswith("openai/"):
        return (
            f"{model_name}@reasoning_effort={reasoning_effort}"
            if reasoning_effort
            else model_name
        )

    if reasoning_effort and "gpt-5" in model_name:
        return f"openai/{model_name}@reasoning_effort={reasoning_effort}"

    return f"openai/{model_name}"

def _parse_exposure_label(res) -> str | None:
    """
    Parses the <answer> tag from an XML-formatted model output.
    Expected values: A, B, C, D, or E.
    """
    if res is None or getattr(res, "output", None) is None:
        return None

    output_str = str(res.output).strip()

    m = re.search(r"<answer>\s*([A-E])\s*</answer>", output_str, re.IGNORECASE)
    return m.group(1).upper() if m else None

def _run_prompt_column(
    df: pd.DataFrame,
    prompt_col: str,
    model_name: str = "openai/gpt-4.1-mini",
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    num_workers: int = 128,
    parser: Callable = _parse_exposure_label,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Run a set of prompts through a model and attach parsed outputs."""
    full_model_name = _full_model_name(model_name, reasoning_effort)
    out_col = output_col or prompt_col.replace("model_input", "model_output")
    temp_col = f"_temp_{out_col}"

    df[temp_col] = run_auto_str( # type: ignore
        df[prompt_col],
        full_model_name=full_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        num_workers=num_workers,
        request_args=RequestArgs(
            use_cache=True,
            hash_keys=True,
        ),
    )

    df[out_col] = df[temp_col].apply(parser)    
    return df

def _parse_task_detail(res) -> str | None:
    if res is None or getattr(res, 'output', None) is None:
        return None
    return str(res.output).strip()

prompt_template = """
The following is a task associated with an occupation:
Occupation: {title}
Task: {task}

Unfortunately, the task description is vague and may overlap with other tasks for this occupation, which are listed below:
{other_list}

Add a single sentence to this task description that clarifies its scope by specifying what it does not include. Your output should be exactly three sentences. The first sentence is the original task. The second sentence starts with: "However, this does not include". Do not answer with anything other than these three sentences.
Finally, ensure that your output is self-contained. It should not explicitly specify that it is adding a sentence or refer to other tasks (e.g., "unlike other tasks", "as opposed to other tasks", etc.).
""".strip()

def _build_task_prompt(row, tasks_by_title) -> str:
    other_tasks = [t for t in tasks_by_title[row['Title']] if t != row['Task']]
    other_list = '\n'.join(f'- {t}' for t in other_tasks) if other_tasks else '- (no other tasks provided)'
    return prompt_template.format(title=row['Title'], task=row['Task'], other_list=other_list)

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Generate detailed task descriptions for O*NET tasks.")
    parser.add_argument("--onet-version", type=str, default="29.2", help="Version of the O*NET database to use (e.g., '29.2').")
    parser.add_argument("--output-path", type=str, default='data/detailed_tasks.csv', help="Path to save the output CSV file.")
    args = parser.parse_args()

    # get onet data
    tasks_dwas_df = _get_tasks_dwas_df(version=args.onet_version)

    # make prompt
    tasks_by_title = tasks_dwas_df.groupby('Title')['Task'].apply(list)
    tasks_dwas_df['model_input_task_detail'] = tasks_dwas_df.apply(
        lambda r: _build_task_prompt(r, tasks_by_title), axis=1
    )

    # run prompts
    tasks_dwas_df = _run_prompt_column(
        tasks_dwas_df,
        prompt_col='model_input_task_detail',
        model_name='openai/gpt-4.1-mini',
        temperature=0.3,
        max_tokens=512,
        num_workers=256,
        parser=_parse_task_detail,
        output_col='task_detailed',
    )

    tasks_dwas_df = tasks_dwas_df.rename(
        columns={
            'Title': 'category_occ',
            'Task': 'category_task',
        }
    )

    tasks_dwas_df = tasks_dwas_df[['O*NET-SOC Code', 'category_occ', 'Task ID', 'category_task', 'DWA ID', 'DWA Title', 'Date', 'Domain Source', 'task_detailed']]
    tasks_dwas_df.to_csv(args.output_path, index=False)
