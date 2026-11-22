from .logging_util import logger
from functools import wraps
import time
from datetime import datetime

def calc_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        start_perf = time.perf_counter()
        logger.info(f" {func.__name__} started at {start_time}")

        try:
            return func(*args, **kwargs)
        finally:
            end = datetime.now()
            latency_ms = (time.perf_counter() - start_perf) * 1000
            logger.info(
                "%s completed at %s | latency = %.2f ms",
                func.__name__,
                end,
                latency_ms
            )
    return wrapper
