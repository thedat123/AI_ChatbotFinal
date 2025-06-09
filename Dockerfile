# --- Build stage ---
FROM python:3.11-slim AS build

WORKDIR /app

# Update package list and install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    unixodbc-dev \
    curl \
    wget \
    git \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 17 for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential packages
RUN pip install --upgrade pip setuptools wheel

# Install essential packages first to avoid conflicts
RUN pip install --no-cache-dir \
    python-dotenv==1.0.0 \
    fastapi==0.100.1 \
    uvicorn[standard]==0.23.2 \
    pydantic==2.0.3

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Verify installations
RUN python -c "from dotenv import load_dotenv; print('✓ dotenv OK')"
RUN python -c "import fastapi; print('✓ fastapi OK')"
RUN python -c "import uvicorn; print('✓ uvicorn OK')"

# Test ODBC connection
RUN python -c "import pyodbc; print('✓ pyodbc OK')"

# Copy application code
COPY . .

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies including ODBC driver
RUN apt-get update && apt-get install -y \
    unixodbc \
    libodbc1 \
    unixodbc-dev \
    curl \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 17 for SQL Server (runtime)
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=build /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --from=build /app /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ODBCINI=/etc/odbc.ini \
    ODBCSYSINI=/etc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "server-chatbot:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
