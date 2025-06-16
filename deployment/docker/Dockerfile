# Optimized multi-stage build for nBedR
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
ARG BUILDKIT_INLINE_CACHE=1

# Set labels
LABEL org.opencontainers.image.title="nBedR" \
      org.opencontainers.image.description="RAG Embedding Toolkit for parallel document processing" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="nBedR Team" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies with aggressive cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-test.txt ./

# Install Python dependencies with pip cache mounting
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=all -r requirements.txt || \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies only
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r nbedr && useradd -r -g nbedr nbedr

# Set working directory
WORKDIR /app

# Copy Python packages from builder (more efficient)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary application files (exclude tests, docs, etc.)
COPY nbedr/ ./nbedr/
COPY nbedr.py ./
COPY pyproject.toml ./
COPY README.md ./
COPY CHANGELOG.md ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/output /app/logs /tmp/nbedr_coordination && \
    chown -R nbedr:nbedr /app /tmp/nbedr_coordination

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    NBEDR_COORDINATION_DIR=/tmp/nbedr_coordination \
    NBEDR_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 nbedr.py status || exit 1

# Switch to non-root user
USER nbedr

# Expose port for potential web interface
EXPOSE 8080

# Default command
ENTRYPOINT ["python3", "nbedr.py"]
CMD ["--help"]