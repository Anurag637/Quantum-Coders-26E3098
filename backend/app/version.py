import os
from datetime import datetime


# ===============================
# Core Version Info
# ===============================

MAJOR = 0
MINOR = 1
PATCH = 0

VERSION = f"{MAJOR}.{MINOR}.{PATCH}"


# ===============================
# Optional Build Metadata
# ===============================

BUILD_ID = os.getenv("BUILD_ID", "dev")
GIT_COMMIT = os.getenv("GIT_COMMIT", "local")
BUILD_TIME = os.getenv("BUILD_TIME", datetime.utcnow().isoformat())


def get_version_info() -> dict:
    """
    Returns structured version metadata.
    Useful for /version endpoint and observability.
    """
    return {
        "version": VERSION,
        "build_id": BUILD_ID,
        "git_commit": GIT_COMMIT,
        "build_time": BUILD_TIME,
    }