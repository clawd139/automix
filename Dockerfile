# AutoMix - Neural DJ Transition Generation
# Docker image for training on CUDA machines (Lambda Cloud, etc.)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .python-version ./
COPY automix/ ./automix/

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Create directories for data/output
RUN mkdir -p /data /output

# Set entrypoint
ENTRYPOINT ["uv", "run", "automix"]

# Default command shows help
CMD ["--help"]

# Example usage:
# docker build -t automix .
# docker run --gpus all -v /path/to/data:/data -v /path/to/output:/output automix train --data /data --output /output --steps 100000
