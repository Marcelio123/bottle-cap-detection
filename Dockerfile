FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY bsort/ ./bsort/

# Install bsort and dependencies
RUN pip install --no-cache-dir -e .

# Create directories for mounting data
RUN mkdir -p /data /outputs /models

# Set entrypoint to bsort CLI
ENTRYPOINT ["bsort"]

# Default command shows help
CMD ["--help"]
