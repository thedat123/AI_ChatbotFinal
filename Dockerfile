# --- Build stage ---
FROM ubuntu:22.04 AS build

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

WORKDIR /app

# Update package list and install build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
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
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Microsoft ODBC Driver 18 for SQL Server (Ubuntu 22.04)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Verify installations
RUN python3 -c "from dotenv import load_dotenv; print('✓ dotenv OK')" || echo "dotenv not found, continuing..."
RUN python3 -c "import fastapi; print('✓ fastapi OK')" || echo "fastapi not found, continuing..."
RUN python3 -c "import uvicorn; print('✓ uvicorn OK')" || echo "uvicorn not found, continuing..."
RUN python3 -c "import pyodbc; print('✓ pyodbc OK')" || echo "pyodbc not found, continuing..."

# Copy application code
COPY . .

# --- Runtime stage ---
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    unixodbc \
    libodbc1 \
    unixodbc-dev \
    curl \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Microsoft ODBC Driver 18 for SQL Server (runtime)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and reinstall packages (more reliable than copying site-packages)
COPY --from=build /app/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

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
CMD ["python3", "-m", "uvicorn", "server-chatbot:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
