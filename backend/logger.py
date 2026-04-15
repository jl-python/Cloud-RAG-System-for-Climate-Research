import logging
import sys
from contextvars import ContextVar

# Context vars for per-request tracing
query_id_var: ContextVar[str] = ContextVar("query_id", default="-")
latency_var:  ContextVar[str] = ContextVar("latency",  default="-")


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.query_id = query_id_var.get("-")
        record.latency  = latency_var.get("-")
        return True


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("climate_rag")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(ContextFilter())

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [qid=%(query_id)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


logger = _build_logger()