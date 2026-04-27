"""
Vercel Serverless Entry Point
=============================
Vercel's Python runtime discovers this file and exposes the `app` object
as the ASGI handler for all routes matched in vercel.json.
"""
import sys
import os

# Ensure the project root is on sys.path so that 'backend.app.main' resolves
# regardless of how Vercel sets its working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export the FastAPI application for Vercel
from backend.app.main import app  # noqa: F401, E402
