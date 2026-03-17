import sys
import os
import traceback

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

try:
    from app.main import app
    print("FastAPI app instance created successfully.")
except Exception as e:
    print("FAILED to create FastAPI app instance")
    traceback.print_exc()
