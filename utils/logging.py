"""
Structured logging for neural cryptanalysis experiments.

Usage:
    from utils.logging import get_logger, setup_logging

    # At program start:
    setup_logging(level='INFO', log_file='experiment.log')

    # In any module:
    logger = get_logger(__name__)
    logger.info("Training started", extra={'rounds': 5, 'cipher': 'speck32'})
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Custom formatter with color support for terminal
class ColorFormatter(logging.Formatter):
    """Colored log output for terminal readability."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


class ExperimentFormatter(logging.Formatter):
    """Structured formatter that includes experiment context."""

    def format(self, record: logging.LogRecord) -> str:
        # Add experiment context fields if present
        extras = []
        for key in ['cipher', 'rounds', 'model', 'epoch', 'accuracy', 'loss']:
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        if extras:
            record.msg = f"{record.msg} [{', '.join(extras)}]"
        return super().format(record)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    experiment_name: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Configure logging for the entire project.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional, logs to file if provided)
        experiment_name: Name prefix for log file
        debug: If True, sets level to DEBUG
    """
    if debug:
        level = 'DEBUG'

    root = logging.getLogger('neural_cryptanalysis')
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    # Console handler with colors
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(
        '%(asctime)s %(levelname)s %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)

    # File handler (structured, no colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(ExperimentFormatter(
            '%(asctime)s %(levelname)-8s %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger scoped under the project namespace.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    # Strip project prefix if present for shorter names
    if name.startswith('neural_cryptanalysis.'):
        name = name[len('neural_cryptanalysis.'):]
    return logging.getLogger(f'neural_cryptanalysis.{name}')


class TrainingLogger:
    """
    Helper for logging training progress with tqdm-friendly output.

    Usage:
        tlog = TrainingLogger('speck32', 5, 'gohr_mlp')
        tlog.epoch(1, train_loss=0.65, val_loss=0.62, val_acc=0.71)
        tlog.finished(best_acc=0.78, total_time=120.5)
    """

    def __init__(self, cipher: str, n_rounds: int, model_name: str):
        self.logger = get_logger('training')
        self.cipher = cipher
        self.n_rounds = n_rounds
        self.model_name = model_name
        self.logger.info(
            f"Training {model_name} on {cipher} ({n_rounds}r)")

    def epoch(self, epoch: int, **metrics):
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        self.logger.info(f"Epoch {epoch:3d} | {' | '.join(parts)}")

    def finished(self, best_acc: float, total_time: float):
        self.logger.info(
            f"Training complete — best_acc={best_acc:.4f}, "
            f"time={total_time:.1f}s "
            f"[{self.cipher}/{self.n_rounds}r/{self.model_name}]")

    def warning(self, msg: str):
        self.logger.warning(msg)
