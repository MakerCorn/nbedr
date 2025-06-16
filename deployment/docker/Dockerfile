# Optimized multi-stage build for nBedR
FROM python:3.11-slim-bookworm AS builder

# Set build arguments for optional features
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
ARG BUILDKIT_INLINE_CACHE=1
ARG INSTALL_CLOUD=false
ARG INSTALL_VECTOR_STORES=false
ARG INSTALL_DOCUMENTS=false
ARG INSTALL_LOCAL_LLM=false
ARG INSTALL_HEAVY=false

# Set labels
LABEL org.opencontainers.image.title="nBedR" \
      org.opencontainers.image.description="RAG Embedding Toolkit for parallel document processing" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="nBedR Team" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies with security updates and aggressive cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
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
COPY requirements-heavy.txt ./

# Validate requirements files are present
RUN echo "Validating requirements files..." && \
    ls -la requirements-*.txt && \
    echo "All requirements files present."

# Upgrade pip and install minimal dependencies with security fixes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools>=75.6.0 wheel

# Install minimal dependencies (always required)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing minimal dependencies..." && \
    echo "Contents of requirements-minimal.txt:" && \
    cat requirements-minimal.txt && \
    echo "Starting pip install..." && \
    pip install --verbose --only-binary=:all: -r requirements-minimal.txt || \
    (echo "Binary install failed, trying with compilation..." && \
     pip install --verbose -r requirements-minimal.txt)

# Conditionally install cloud dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_CLOUD" = "true" ]; then \
        echo "Installing cloud dependencies..." && \
        pip install --only-binary=:all: -r requirements-cloud.txt || \
        pip install -r requirements-cloud.txt; \
    else \
        echo "Skipping cloud dependencies (INSTALL_CLOUD=$INSTALL_CLOUD)"; \
    fi

# Conditionally install vector store dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_VECTOR_STORES" = "true" ]; then \
        echo "Installing vector store dependencies..." && \
        pip install --only-binary=:all: -r requirements-vector-stores.txt || \
        pip install -r requirements-vector-stores.txt; \
    else \
        echo "Skipping vector store dependencies (INSTALL_VECTOR_STORES=$INSTALL_VECTOR_STORES)"; \
    fi

# Conditionally install document processing dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DOCUMENTS" = "true" ]; then \
        echo "Installing document processing dependencies..." && \
        pip install --only-binary=:all: -r requirements-documents.txt || \
        pip install -r requirements-documents.txt; \
    else \
        echo "Skipping document processing dependencies (INSTALL_DOCUMENTS=$INSTALL_DOCUMENTS)"; \
    fi

# Conditionally install local LLM dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_LOCAL_LLM" = "true" ]; then \
        echo "Installing local LLM dependencies..." && \
        pip install --only-binary=:all: -r requirements-local-llm.txt || \
        pip install -r requirements-local-llm.txt; \
    else \
        echo "Skipping local LLM dependencies (INSTALL_LOCAL_LLM=$INSTALL_LOCAL_LLM)"; \
    fi

# Conditionally install heavy dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_HEAVY" = "true" ]; then \
        echo "Installing heavy dependencies..." && \
        pip install --only-binary=:all: -r requirements-heavy.txt || \
        pip install -r requirements-heavy.txt; \
    else \
        echo "Skipping heavy dependencies (INSTALL_HEAVY=$INSTALL_HEAVY)"; \
    fi

# Verify installation by testing core imports
RUN echo "Verifying installation..." && \
    python -c "import openai; print('✓ OpenAI imported successfully')" && \
    python -c "import faiss; print('✓ FAISS imported successfully')" && \
    python -c "import pydantic; print('✓ Pydantic imported successfully')" && \
    python -c "import numpy; print('✓ NumPy imported successfully')" && \
    echo "✓ All core dependencies verified"

# Clean up build artifacts and caches to save space
RUN echo "Cleaning up build artifacts..." && \
    find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/* && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Production stage
FROM python:3.11-slim-bookworm AS production

# Install runtime dependencies with security updates
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
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