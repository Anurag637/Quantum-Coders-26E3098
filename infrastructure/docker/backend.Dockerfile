FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml README.md ./

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (only main)
RUN poetry install --only main --no-root

# Copy backend application
COPY backend/ ./backend/

ENV PYTHONPATH=/app/backend

# Run the backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
