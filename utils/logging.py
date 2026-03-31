
import logging
import sys
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[41m',
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


class ExperimentFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
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
    if debug:
        level = 'DEBUG'

    root = logging.getLogger('neural_cryptanalysis')
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(
        '%(asctime)s %(levelname)s %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)

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
    if name.startswith('neural_cryptanalysis.'):
        name = name[len('neural_cryptanalysis.'):]
    return logging.getLogger(f'neural_cryptanalysis.{name}')


class TrainingLogger:

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
