FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY src/ ./src/
COPY models/ ./models/

# Create models dir (model downloaded at startup from R2)
RUN mkdir -p models

# Non-root user
RUN useradd -m -u 1000 maia
USER maia

EXPOSE 8000

# Railway sets $PORT; default to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
