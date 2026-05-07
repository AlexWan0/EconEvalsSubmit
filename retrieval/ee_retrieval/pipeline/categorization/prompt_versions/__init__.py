import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import importlib
import sys
from types import ModuleType
from dataclasses import dataclass
from typing import Callable

from ....utils.prompt_utils import extract_tag


ROOT = Path(__file__).parent
PROMPTS_MODULES: dict[str, ModuleType] = {} # values are python modules which contain the prompts

def _reload_prompts():
    """Loads prompts in the current directory into ``PROMPTS_MODULES``.
    """
    PROMPTS_MODULES.clear()

    for file in ROOT.glob("prompts_*.py"):
        module_name = file.stem
        version = module_name.split("_", 1)[1]

        fq_name = f'{__name__}.{module_name}'

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        if fq_name in sys.modules:
            mod = importlib.reload(sys.modules[fq_name])

        else:
            mod = importlib.import_module(fq_name)

        PROMPTS_MODULES[version] = mod

    logger.info(f'categorization prompts found: {PROMPTS_MODULES.keys()}')

_reload_prompts()

@dataclass
class CategorizationPrompts:
    """Prompts used for classification: the ``prompts_version`` & ``classif_type`` specifies a classification pipeline that uses these prompts.

    Args:
        classif_prompt (str): prompt used to perform the actual categorization (whether a user request belongs to a given category); prompt is invoked after all other preprocessing steps applied to the user-requests
        summarize_prompt (str): prompt used to summarize the raw user requests made to the AI assistant
        summarize_parser (Callable[[str], str | None]): function used to parse out summaries from raw model responses; returns None if no valid summary is found
    """
    classif_prompt: str
    summarize_prompt: str
    summarize_parser: Callable[[str], str | None]

def _get_classif_prompt(classif_type: str, prompts_version: str) -> str:
    if classif_type == 'replace':
        classif_prompt = PROMPTS_MODULES[prompts_version].CLASSIF_PROMPT
    
    elif classif_type == 'replace_occ':
        classif_prompt = PROMPTS_MODULES[prompts_version].CLASSIF_PROMPT_OCC
    
    elif classif_type == 'skills':
        classif_prompt = PROMPTS_MODULES[prompts_version].CLASSIF_PROMPT_SKILLS
    
    elif classif_type == 'direct':
        classif_prompt = PROMPTS_MODULES[prompts_version].CLASSIF_PROMPT_DIRECT
    
    elif classif_type == 'direct_iwa':
        classif_prompt = PROMPTS_MODULES[prompts_version].IWA_CLASSIF_PROMPT_DIRECT
    
    elif classif_type == 'direct_dwa_2':
        classif_prompt = PROMPTS_MODULES[prompts_version].DWA_CLASSIF_PROMPT_DIRECT
    
    elif classif_type == 'dwa':
        classif_prompt = PROMPTS_MODULES[prompts_version].CLASSIF_PROMPT_TASK_ONLY
        
        assert 'category_task_detailed' in classif_prompt
        classif_prompt = classif_prompt.replace('category_task_detailed', 'dwa')
    
    elif classif_type == 'dwa_2':
        classif_prompt = PROMPTS_MODULES[prompts_version].DWA_CLASSIF_PROMPT_TASK_ONLY
        
        assert 'category_task_detailed' in classif_prompt
        classif_prompt = classif_prompt.replace('category_task_detailed', 'dwa_detailed')
    
    elif classif_type == 'dwa_occ':
        classif_prompt = PROMPTS_MODULES[prompts_version].DWA_CLASSF_PROMPT_OCC
    
    else:
        raise ValueError(f'invalid classif_prompt={classif_type}')

    return classif_prompt

def _get_summarize_prompt(classif_type: str, prompts_version: str) -> tuple[str, Callable[[str], str | None]]:
    if classif_type == 'dwa_2':
        summarize_prompt = PROMPTS_MODULES[prompts_version].DWA_SUMMARIZE_PROMPT
    else:
        summarize_prompt = PROMPTS_MODULES[prompts_version].SUMMARIZE_PROMPT
    
    summarize_parser = (lambda x: x) if '<answer>' not in summarize_prompt else (lambda x: extract_tag(x, 'answer'))

    return summarize_prompt, summarize_parser

def get_prompts(classif_type: str, prompts_version: str) -> CategorizationPrompts:
    """Given ``classif_type`` and ``prompts_version`` get the actual prompts used by the LM for categorization.

    Args:
        classif_type (str): method of categorization
        prompts_version (str): version of prompts
    
    Returns:
        CategorizationPrompts: dataclass wrapper around the prompts & output parsers to be used
    """
    summarize_prompt, summarize_parser = _get_summarize_prompt(classif_type, prompts_version)

    return CategorizationPrompts(
        classif_prompt=_get_classif_prompt(classif_type, prompts_version),
        summarize_prompt=summarize_prompt,
        summarize_parser=summarize_parser
    )
