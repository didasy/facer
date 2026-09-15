FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies for OpenCV and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pip network robustness: large wheels (opencv ~61 MB) time out on slow links
# with pip's default 15 s read timeout. Raise timeout + retries.
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

# Copy application code
COPY ./app ./app

EXPOSE 8000

# Run Uvicorn with standard production setup
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
