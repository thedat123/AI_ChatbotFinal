# Build stage
FROM python:3.11-slim AS build
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Copy dependencies and source from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /app /app

# Expose a configurable port (default 8000)
ARG APP_PORT=8000
EXPOSE ${APP_PORT}

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_PORT=${APP_PORT}

# Run Uvicorn with multiple workers for production
CMD uvicorn server-chatbot:app --host 0.0.0.0 --port ${APP_PORT} --workers 4