"""Per-run console and UTF-8 file logging."""
import logging
from pathlib import Path


def make_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
