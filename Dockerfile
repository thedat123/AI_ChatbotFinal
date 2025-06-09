# Build stage: Use Python 3.11 slim image for building dependencies
FROM python:3.11-slim AS build
WORKDIR /app

# Copy requirements file to install dependencies
COPY requirements.txt .

# Install build tools and dependencies, then clean up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY . .

# Runtime stage: Use Python 3.11 slim image for a lean runtime environment
FROM python:3.11-slim
WORKDIR /app

# Copy installed dependencies and source code from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /app /app

# Expose a configurable port (default 8000)
ARG APP_PORT=8000
EXPOSE ${APP_PORT}

# Set environment variables for Python and application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_PORT=${APP_PORT}

# Run Uvicorn with multiple workers for production, using the configured port
CMD ["uvicorn", "server-chatbot:app", "--host", "0.0.0.0", "--port", "${APP_PORT}", "--workers", "4"]
