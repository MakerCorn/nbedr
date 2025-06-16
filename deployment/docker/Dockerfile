# Optimized multi-stage build for nBedR
FROM python:3.11-slim AS builder

# Set build arguments for optional features
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
ARG BUILDKIT_INLINE_CACHE=1
ARG INSTALL_CLOUD=false
ARG INSTALL_VECTOR_STORES=false
ARG INSTALL_DOCUMENTS=false
ARG INSTALL_LOCAL_LLM=false

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

# Copy requirements files for better caching
COPY requirements-minimal.txt ./
COPY requirements-cloud.txt ./
COPY requirements-vector-stores.txt ./
COPY requirements-documents.txt ./
COPY requirements-local-llm.txt ./

# Install minimal dependencies first (always required)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=:all: -r requirements-minimal.txt || \
    pip install -r requirements-minimal.txt

# Conditionally install optional features
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_CLOUD" = "true" ]; then \
        pip install --only-binary=:all: -r requirements-cloud.txt || \
        pip install -r requirements-cloud.txt; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_VECTOR_STORES" = "true" ]; then \
        pip install --only-binary=:all: -r requirements-vector-stores.txt || \
        pip install -r requirements-vector-stores.txt; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DOCUMENTS" = "true" ]; then \
        pip install --only-binary=:all: -r requirements-documents.txt || \
        pip install -r requirements-documents.txt; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_LOCAL_LLM" = "true" ]; then \
        pip install --only-binary=:all: -r requirements-local-llm.txt || \
        pip install -r requirements-local-llm.txt; \
    fi

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