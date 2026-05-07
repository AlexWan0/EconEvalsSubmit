import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import importlib
import sys
from types import ModuleType


ROOT = Path(__file__).parent
PROMPTS_MODULES: dict[str, ModuleType] = {} # values are python modules which contain the prompts

def _reload_prompts():
    """Loads prompts from the current directory into ``PROMPTS_MODULES``.
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

    logger.info(f'quality (homework) prompts found: {PROMPTS_MODULES.keys()}')

_reload_prompts()
