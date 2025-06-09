# --- Build stage ---
FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    unixodbc-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install essential packages first (avoid conflicts)
RUN pip install --no-cache-dir python-dotenv==1.0.0 fastapi==0.100.1 uvicorn[standard]==0.23.2 pydantic==2.0.3

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

RUN python -c "from dotenv import load_dotenv; print('✓ dotenv OK')"
RUN python -c "import fastapi; print('✓ fastapi OK')"
RUN python -c "import uvicorn; print('✓ uvicorn OK')"

COPY . .

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies (including ODBC runtime libs)
RUN apt-get update && apt-get install -y \
    unixodbc \
    libodbc1 \
    unixodbc-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/

COPY --from=build /app /app

RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server-chatbot:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
