# ===============================
#   BUILDER STAGE
# ===============================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies only for building Python wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata
COPY pyproject.toml ./
COPY bsort/ ./bsort/

# Install dependencies into a temporary folder (no cache → smaller)
RUN pip install --no-cache-dir --prefix=/install .

# ===============================
#   FINAL RUNTIME STAGE
# ===============================
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies required by OpenCV and ML tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create your data directories
RUN mkdir -p /data /outputs /models

# bsort CLI entry point
ENTRYPOINT ["bsort"]
CMD ["--help"]
