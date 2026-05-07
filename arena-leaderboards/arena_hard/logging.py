import os
import logging
from datetime import datetime
import sys

import coloredlogs


_LOG_CONFIGURED = False
_LOG_PATH: str | None = None

FIELD_STYLES = {
    'asctime': {'color': 'green'},
    'filename': {'color': 'cyan'},
    'funcName': {'color': 'cyan'},
}

LEVEL_STYLES = {
    'error': {'background': 'red', 'color': 'yellow'},
    'warning': {'background': 'red', 'color': 'white'},
    'info': {'color': 'white'},
    'debug': {'color': 'white'},
}


def config_logging(log_dir: str = ".logs") -> str:
    global _LOG_CONFIGURED, _LOG_PATH

    if _LOG_CONFIGURED:
        assert _LOG_PATH is not None
        return _LOG_PATH

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(log_dir, f"{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    coloredlogs.install(
        level='INFO',
        logger=logger,
        fmt="%(asctime)s [%(filename)s - %(funcName)s] %(levelname)8s: %(message)s",
        datefmt="%m-%d %H:%M",
        field_styles=FIELD_STYLES,
        level_styles=LEVEL_STYLES
    )

    file_handler = logging.FileHandler(_LOG_PATH, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(filename)s - %(funcName)s] %(levelname)8s: %(message)s",
        datefmt="%m-%d %H:%M"
    ))
    logger.addHandler(file_handler)

    logger.info('configured log for call %s', sys.argv[0])

    _LOG_CONFIGURED = True
    return _LOG_PATH
