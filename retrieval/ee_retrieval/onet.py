from dataclasses import dataclass
import pandas as pd
from typing import Literal
from fast_openai import RequestArgs, run_openai_str
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)
from typing import Annotated

from .task_classification import TASK_CLASSIF_PROMPTS
from .summarization import (
    get_task_summ_map,
    get_dwa_summ_map,
    get_iwa_summ_map
)
from .utils import cached_func, check_vars, Checksum


@cached_func()
def _make_classified_dwa_df(tasks_dwa_df: pd.DataFrame, method: str = 'v1') -> pd.DataFrame:
    prompt_cfg =  TASK_CLASSIF_PROMPTS[method]

    tasks_dwa_df['_task_model_input'] = tasks_dwa_df['DWA Title'].apply(
        lambda x: prompt_cfg.prompt.format(task_input=x)
    )

    tasks_dwa_df['_task_model_output'] = run_openai_str(
        tasks_dwa_df['_task_model_input'],
        request_args=RequestArgs(
            hash_keys=True
        ),
        pbar_name='_make_classified_dwa_df: {model_name}',
        **prompt_cfg.model_kwargs
    )

    tasks_dwa_df['parsed_answer'] = tasks_dwa_df['_task_model_output'].apply(
        lambda x: prompt_cfg.parse_func(x.output)
    )

    tasks_dwa_df = tasks_dwa_df[tasks_dwa_df['parsed_answer'] == prompt_cfg.keep_label]

    return tasks_dwa_df


@dataclass(frozen=True)
class ONETData(ABC):
    """An abstract class for O*NET data.

    Given some underlying raw O*NET data, it should provide interface for producing data in formats (e.g., dataframes) we need & transformed in ways we need (e.g., adding more detail through summarization).
    """

    def __post_init__(self):
        check_vars(self)

    @abstractmethod
    def get_dwa_df(self) -> pd.DataFrame:
        """Gets the many-to-many mapping of DWAs & task statements/occuaptions & associated metadata. Should have columns as specified here: https://www.onetcenter.org/dictionary/29.2/excel/tasks_to_dwas.html

        Returns:
            pd.DataFrame: dataframe containing DWAs & corresponding task statements/occupations
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_iwa_df(self) -> pd.DataFrame:
        """Gets the one-to-many mapping of IWAs & DWAs & associated metadata. Should have columns as specified here: https://www.onetcenter.org/dictionary/29.2/excel/dwa_reference.html

        Returns:
            pd.DataFrame: dataframe containing IWAs & corresponding DWAs
        """
        raise NotImplementedError()

    @abstractmethod
    def get_classified_dwa_df(self) -> pd.DataFrame:
        """(deprecated) Filters rows of the DWA by some classification (e.g., whether the task could hypothetically be performed by an LM).

        Returns:
            pd.DataFrame: dataframe containing the kept rows based on the classification
        """
        raise NotImplementedError()
    
    def get_dwa_summ_map(self, method: str = 'v2') -> dict[str, str]:
        """Produces a map from original DWA Titles to more detailed DWA Titles baesd on a summary of adjacent task statements in the O*NET hierarchy.

        Args:
            method (str): DWA summarization method to use (see: ``summarization/dwas.py``). Defaults to ``v2``.

        Returns:
            dict[str, str]: mapping from original task statements to detailed task statements
        """
        dwa_summ_df = get_dwa_summ_map(
            self.get_dwa_df(),
            method=method # type: ignore
        )

        # TODO: add uniqueness check instead of dropping
        dwa_summ_map = dwa_summ_df.drop_duplicates(subset=['DWA Title']).set_index('DWA Title')['dwa_detailed']

        return dwa_summ_map.to_dict()
    
    def get_iwa_summ_map(self, method: str = 'v1', dwa_summ_method: str = 'v2_posonly') -> dict[str, str]:
        """Produces a map from original IWA Titles to more detailed IWA Titles baesd on a summary of DWAs based on the O*NET hierarchy.

        Args:
            method (str): IWA summarization method to use (see: ``summarization/iwas.py``). Defaults to ``v1`` (currently, the only method).
            dwa_summ_method (str): DWA summarization method to use (see: ``summarization/dwas.py``). Defaults to ``v2_posonly``.

        Returns:
            dict[str, str]: mapping from original IWA titles to detailed IWA titles.
        """

        dwa_df = self.get_dwa_df()
        iwa_df = self.get_iwa_df()
        dwa_summ_map = self.get_dwa_summ_map(dwa_summ_method)

        iwa_summ_df = get_iwa_summ_map(
            dwa_df,
            iwa_df,
            dwa_summ_map,
            method=method # type: ignore
        )

        iwa_summ_map = iwa_summ_df.set_index('DWA Title')['iwa_detailed']

        return iwa_summ_map.to_dict()

    def get_task_summ_map(self, method: str = 'v1') -> dict[tuple[str, str], str]:
        """Produces a map from original task statements & occupations to more detailed task statements for each occupation by contrasting the task statement against other tasks performed by the occupation.

        Args:
            method (str): summarization method to use (see: ``summarization/tasks.py``). Defaults to ``v1`` (currently, the only method).

        Returns:
            dict[tuple[str, str], str]: mapping from original (task occupations, task statements) to more detailed task statements (specific to that occupation).
        """

        task_summ_df = get_task_summ_map(
            self.get_dwa_df(),
            method=method # type: ignore
        )

        task_summ_map = task_summ_df.set_index(['category_occ', 'category_task'])['category_task_detailed']

        return task_summ_map.to_dict()

    

@dataclass(frozen=True)
class ONETDataV1_292(ONETData):
    """An abstract class for O*NET data. O*NET interface for 29.2
    
    Args:
        dwa_fp (str): location of O*NET 29.2 file ``Tasks to DWAs 29.2.xlsx``
        dwa_classif_method (``Literal['disable', 'v1', 'v2']``): method to use to classify as being able to be hypothetically completed by an LM & filter out non-qualifying samples. Default to ``disable``.
        
        iwa_fp (str): location of O*NET 29.2 file ``DWA Reference 29.2.xlsx``
    """
    dwa_fp: Annotated[str, Checksum('159c477138b6927')] = 'data/onet_data/Tasks to DWAs 29.2.xlsx'
    dwa_classif_method: Literal['disable', 'v1', 'v2'] = 'disable'
    
    iwa_fp: Annotated[str, Checksum('2bc21199fec3dc2')] = 'data/onet_data/DWA Reference 29.2.xlsx'

    def get_dwa_df(self) -> pd.DataFrame:
        """Gets the many-to-many mapping of DWAs & task statements/occuaptions & associated metadata. Is exactly the data here as a pandas dataframe: https://www.onetcenter.org/dictionary/29.2/excel/tasks_to_dwas.html.

        Returns:
            pd.DataFrame: dataframe containing DWAs & corresponding task statements/occupations for O*NET 29.2
        """

        return pd.read_excel(self.dwa_fp)

    def get_iwa_df(self) -> pd.DataFrame:
        """Gets the one-to-many mapping of IWAs & DWAs & associated metadata. Is exactly the data here as a pandas dataframe: https://www.onetcenter.org/dictionary/29.2/excel/dwa_reference.html

        Returns:
            pd.DataFrame: dataframe containing IWAs & corresponding DWAs
        """

        return pd.read_excel(self.iwa_fp)

    def get_classified_dwa_df(self) -> pd.DataFrame:
        """(deprecated; disabled by default) Filters rows of the DWA by some classification (e.g., whether the task could hypothetically be performed by an LM).

        Returns:
            pd.DataFrame: dataframe containing the kept rows based on the classification
        """

        if self.dwa_classif_method == 'disable':
            return self.get_dwa_df()

        return _make_classified_dwa_df(
            self.get_dwa_df(),
            method=self.dwa_classif_method
        )

@dataclass(frozen=True)
class ONETWorkbankDesires(ONETData):
    """O*NET interface which only use rows with DWAs included in WorkBank domain_worker_desires_v0.csv
    
    Args:
        wb_dwa_fp (str): Location of O*NET 29.2 file: ``Tasks to DWAs 29.2.xlsx``
        wb_desires_fp (str): Location of Workbank file: ``domain_worker_desires.csv``
    """

    wb_dwa_fp: Annotated[str, Checksum('159c477138b6927')] = 'data/onet_data/Tasks to DWAs 29.2.xlsx'
    wb_desires_fp: Annotated[str, Checksum('58e969946a4620')] = 'data/workbank_data/domain_worker_desires.csv'

    def get_dwa_df(self) -> pd.DataFrame:
        """Gets the many-to-many mapping of DWAs & task statements/occuaptions & associated metadata. Is exactly the data here as a pandas dataframe: https://www.onetcenter.org/dictionary/29.2/excel/tasks_to_dwas.html.

        Returns:
            pd.DataFrame: dataframe containing DWAs & corresponding task statements/occupations for O*NET 29.2
        """

        return pd.read_excel(self.wb_dwa_fp)
    
    def get_classified_dwa_df(self) -> pd.DataFrame:
        """Selects the subset of the DWAs & task statements/occuaptions that's used in the WorkBank worker_desires data.

        Returns:
            pd.DataFrame: dataframe containing the kept rows based on what's used in the WorkBank worker_desires data.
        """

        dwa_df = self.get_dwa_df()

        wb_desires_df = pd.read_csv(self.wb_desires_fp)

        # get title/tasks that are mentioned by onet
        included_tasks = set(wb_desires_df.apply(
            lambda row: (row['Occupation (O*NET-SOC Title)'], row['Task']),
            axis=1
        ))

        # get the dwas for those corresponding tasks
        included_dwas = set(dwa_df[dwa_df.apply(
            lambda row: (row['Title'], row['Task']) in included_tasks,
            axis=1
        )]['DWA Title'])

        # filter the orig data by these dwas
        return dwa_df[dwa_df['DWA Title'].isin(included_dwas)]

    def get_iwa_df(self) -> pd.DataFrame:
        raise NotImplementedError()
