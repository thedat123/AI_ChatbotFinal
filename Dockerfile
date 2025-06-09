# --- Build stage ---
FROM python:3.11-slim AS build
WORKDIR /app

# Cài pip & wheel (đảm bảo compatibility khi cài gói)
RUN apt-get update && apt-get install -y build-essential

# Copy và cài dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# --- Runtime stage ---
FROM python:3.11-slim
WORKDIR /app

# Copy thư viện từ build stage
COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=build /app /app

# Expose port
ARG APP_PORT=8000
EXPOSE ${APP_PORT}

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_PORT=${APP_PORT}

# Run app with uvicorn
CMD ["uvicorn", "server-chatbot:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
