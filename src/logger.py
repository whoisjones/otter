import logging
import sys
import os
from pathlib import Path


def setup_logger(output_dir: str, name: str = "dual_encoder", is_main_process: bool = True) -> logging.Logger:
    """
    Set up logger that writes to both console and file.
    
    Args:
        output_dir: Directory to save log file
        name: Logger name
        is_main_process: Whether this is the main process (only main process logs to console)
    
    Returns:
        Configured logger
    """
    # Create output directory if it doesn't exist (only on main process to avoid race conditions)
    if is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Wait for main process to create directory
        import time
        max_wait = 10
        waited = 0
        while not Path(output_dir).exists() and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler (detailed format with timestamps) - all processes write to file
    # Use process-specific log file to avoid conflicts
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        log_file = Path(output_dir) / "run.log"
    else:
        log_file = Path(output_dir) / f"run_rank{rank}.log"
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple format, just the message) - only on main process
    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

