FROM python:3.11-slim
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --no-cache-dir \
    black \
    pylint \
    pytest \
    ipython==8.18.1

# Set working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p Data/Daily_Report

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]