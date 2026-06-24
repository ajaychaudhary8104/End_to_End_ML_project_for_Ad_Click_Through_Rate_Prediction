FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY requirements.txt .
COPY setup.py .
COPY README.md* .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Copy project
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app
    
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]