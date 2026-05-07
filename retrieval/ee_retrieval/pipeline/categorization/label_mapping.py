import pandas as pd
from typing import Literal


def _keep(p: tuple[str | None, ...] | str, allow: list[str], default: str = 'E') -> bool:
    if isinstance(p, str):
        return p.strip() in allow
    
    thresh = len(p)

    p_clean = (
        x if x is not None else default
        for x in p
    )

    thresh_mask = [c.strip() in allow for c in p_clean]

    if sum(thresh_mask) >= thresh:
        return True

    return False

def _keep_ABCD(p: tuple[str, ...] | str) -> bool:
    return _keep(p, ['A', 'B', 'C', 'D'])

def _keep_ABC(p: tuple[str, ...] | str) -> bool:
    return _keep(p, ['A', 'B', 'C'])

def _keep_AB(p: tuple[str, ...] | str) -> bool:
    return _keep(p, ['A', 'B'])

def _keep_A(p: tuple[str, ...] | str) -> bool:
    return _keep(p, ['A'])

def apply_do_keep(
        df: pd.DataFrame,
        keep_method: Literal['ABC', 'ABCD', 'A'],
        preds_col: str,
        ens_res_col: str
    ):
    """Convert parsed predictions to a keep/drop decision (whether to keep or drop the sample).
    Adds a column (`ens_res_col`) to the dataframe (`df`) using values in `preds_col`.
    
    Args:
        df (pd.DataFrame): dataframe which contains rows to parse
        keep_method (Literal['ABC', 'ABCD', 'A']): method to map output categories to keep/drop; currently just processes MCQ answers
        preds_col (str): input column
        ens_res_col (str): output column name
    """

    if keep_method == 'ABC':
        df[ens_res_col] = df[preds_col].apply(_keep_ABC)
    
    elif keep_method == 'ABCD':
        df[ens_res_col] = df[preds_col].apply(_keep_ABCD)
    
    elif keep_method == 'A':
        df[ens_res_col] = df[preds_col].apply(_keep_A)
    
    else:
        raise ValueError(f'invalid keep_method: {keep_method}')
