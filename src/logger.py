import logging
import os
import sys
import time
import warnings
from pathlib import Path

import transformers

LOGGER_NAME = "otter"


def silence_transformers_warnings():
    # transformers renames beta/gamma parameters when loading older encoder
    # checkpoints and warns once per tensor, which buries the training log.
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    transformers.logging.set_verbosity_error()


def setup_logger(
    output_dir: str, name: str = LOGGER_NAME, is_main_process: bool = True
) -> logging.Logger:
    if is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Wait for the main process to create the directory before writing into it.
        waited = 0.0
        while not Path(output_dir).exists() and waited < 10:
            time.sleep(0.1)
            waited += 0.1
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    rank = int(os.environ.get("LOCAL_RANK", 0))
    log_file = Path(output_dir) / ("run.log" if rank == 0 else f"run_rank{rank}.log")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    return logger
