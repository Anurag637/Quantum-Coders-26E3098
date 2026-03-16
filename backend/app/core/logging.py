import logging
import sys
import uuid
from typing import Optional

from pythonjsonlogger import jsonlogger

from backend.app.core.config import get_settings


settings = get_settings()


# ===============================
# Request ID Context
# ===============================

class RequestIdFilter(logging.Filter):
    """
    Injects request_id into logs if available.
    """

    def __init__(self):
        super().__init__()
        self._request_id: Optional[str] = None

    def set_request_id(self, request_id: str):
        self._request_id = request_id

    def clear(self):
        self._request_id = None

    def filter(self, record):
        record.request_id = self._request_id or "-"
        return True


request_id_filter = RequestIdFilter()


# ===============================
# Logging Configuration
# ===============================

def setup_logging():
    """
    Configure global logging.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_JSON:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] "
            "[request_id=%(request_id)s] %(message)s"
        )

    handler.setFormatter(formatter)
    handler.addFilter(request_id_filter)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging configured.")


# ===============================
# Request ID Utilities
# ===============================

def generate_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(request_id: str):
    request_id_filter.set_request_id(request_id)


def clear_request_id():
    request_id_filter.clear()