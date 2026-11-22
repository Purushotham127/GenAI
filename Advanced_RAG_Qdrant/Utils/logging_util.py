import logging
from pathlib import Path

log_file = Path(__file__).resolve().parents[1] / "logs" / "app.log"
log_level: str = "INFO"
log_file.parent.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger("App")
if logger.handlers:
    print("Logger already created!")

logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

log_format = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)

formatter = logging.Formatter(
    fmt=log_format,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)