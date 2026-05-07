import pandas as pd
import argparse


def _get_tasks_dwas_df(version: str = '29.2') -> pd.DataFrame:
    version_url_str = version.replace('.', '_')

    tasks_dwas_df = pd.read_excel(f'https://www.onetcenter.org/dl_files/database/db_{version_url_str}_excel/Tasks%20to%20DWAs.xlsx')
    
    return tasks_dwas_df


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Generate detailed task descriptions for O*NET tasks.")
    parser.add_argument("--onet-version", type=str, default="29.2", help="Version of the O*NET database to use (e.g., '29.2').")
    parser.add_argument("--output-path-task-space", type=str, default='data/task_space.pkl.zst', help="Path to save the output CSV file.")
    parser.add_argument("--output-path-task-dwa", type=str, default='data/task_dwa.pkl.zst', help="Path to save the output CSV file.")
    args = parser.parse_args()

    # get onet data
    tasks_dwas_df = _get_tasks_dwas_df(version=args.onet_version)

    tasks_dwas_df = tasks_dwas_df.rename(
        columns={'Task': 'category_task', 'Title': 'category_occ', 'DWA Title': 'dwa'}
    )[['category_task', 'category_occ', 'dwa']]

    print(len(tasks_dwas_df), tasks_dwas_df.columns)
    tasks_dwas_df.to_pickle(args.output_path_task_dwa)

    task_space_df = (
        tasks_dwas_df[['category_task', 'category_occ']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(len(task_space_df), task_space_df.columns)
    task_space_df.to_pickle(args.output_path_task_space)
