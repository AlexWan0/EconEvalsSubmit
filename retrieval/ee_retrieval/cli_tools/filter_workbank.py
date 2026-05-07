import tyro
from dataclasses import dataclass
from tyro.conf import OmitArgPrefixes
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from .base import ONETWorkbankDesires, DataFrameFilter, DataArgs


@dataclass
class WBFilter:
    '''
    Produces dataframe filter for the conversation-category pairs based on tasks that exist in WorkBank.
    '''
    data_args: OmitArgPrefixes[DataArgs]
    
    out_fn: str = "wb_filter.filter"

    target_title_col: str = "category_occ"
    target_task_col: str = "category_task"

    wb_title_col: str = "Occupation (O*NET-SOC Title)"
    wb_task_col: str = "Task"

def main(args: WBFilter):
    logger.info(f'called with args: {args.__dict__}')

    wb_args = args.data_args.onet_args

    if not isinstance(wb_args, ONETWorkbankDesires):
        raise ValueError('must use dwaworkbank mode')

    wb_desires_df = pd.read_csv(wb_args.wb_desires_fp)

    res_filter = DataFrameFilter(
        col_names=(args.target_title_col, args.target_task_col),
        col_values=frozenset(
            (row[args.wb_title_col], row[args.wb_task_col])
            for i, row in wb_desires_df.iterrows()
        )
    )
    logger.info(f'filter has size {len(res_filter.col_values)}')

    output_fp = args.data_args.build_path(args.out_fn, is_folder=False)

    res_filter.save(output_fp)

    logger.info(f'saved to {output_fp}')

if __name__ == '__main__':
    main(tyro.cli(WBFilter))
