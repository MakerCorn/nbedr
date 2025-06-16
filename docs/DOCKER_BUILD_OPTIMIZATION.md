# Docker Build Optimization Guide

## Overview

The nBedR Docker build has been optimized for speed and flexibility by implementing modular requirements files and conditional dependency installation.

## Requirements Structure

### Core Files

- **`requirements-minimal.txt`** - Essential dependencies only (~15 packages)
  - Basic functionality with OpenAI provider and FAISS vector store
  - Fastest build time, smallest image size
  - Suitable for basic document processing workloads

- **`requirements-cloud.txt`** - Cloud provider support
  - AWS Bedrock, Azure AI, Google Vertex AI
  - SharePoint integration

- **`requirements-vector-stores.txt`** - Additional vector databases
  - ChromaDB, Pinecone, Elasticsearch, PostgreSQL/pgvector

- **`requirements-documents.txt`** - Extended document processing
  - PowerPoint, Word, enhanced PDF processing
  - Pandas/PyArrow for data export

- **`requirements-local-llm.txt`** - Local LLM providers
  - LMStudio, Ollama, Llama.cpp support
  - Lightweight HuggingFace transformers (CPU-only)

- **`requirements-heavy.txt`** - Large optional dependencies
  - Sentence transformers (500MB+ models)
  - ChromaDB (100MB+ dependencies)
  - Only install if specifically needed

## Build Options

### Minimal Build (Default)
```bash
docker build -t nbedr:minimal .
```
- ~15 dependencies only
- OpenAI + FAISS support
- Fastest build (~3-5 minutes)
- Smallest image size (~500MB)

### Cloud-Enabled Build
```bash
docker build --build-arg INSTALL_CLOUD=true -t nbedr:cloud .
```
- Adds AWS, Azure, Google cloud providers
- SharePoint integration

### Full-Featured Build
```bash
docker build \
  --build-arg INSTALL_CLOUD=true \
  --build-arg INSTALL_VECTOR_STORES=true \
  --build-arg INSTALL_DOCUMENTS=true \
  --build-arg INSTALL_LOCAL_LLM=true \
  -t nbedr:full .
```

### Custom Build
```bash
docker build \
  --build-arg INSTALL_VECTOR_STORES=true \
  --build-arg INSTALL_DOCUMENTS=true \
  -t nbedr:custom .
```

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `INSTALL_CLOUD` | `false` | AWS, Azure, Google cloud providers |
| `INSTALL_VECTOR_STORES` | `false` | Pinecone, Elasticsearch, pgvector (ChromaDB moved to heavy) |
| `INSTALL_DOCUMENTS` | `false` | PowerPoint, Word, advanced document processing |
| `INSTALL_LOCAL_LLM` | `false` | Local LLM providers (lightweight transformers) |
| `INSTALL_HEAVY` | `false` | Large dependencies (sentence-transformers, ChromaDB) |

## CI/CD Integration

The CI workflow builds two variants:

### Branch Builds
- **Minimal image** only (amd64 platform)
- Fastest testing and validation
- Tagged as: `ghcr.io/makercorn/nbedr:pr-123`, `ghcr.io/makercorn/nbedr:branch-name`

### Main Branch Builds
- **Minimal image**: `ghcr.io/makercorn/nbedr:latest`, `ghcr.io/makercorn/nbedr:sha-abc123`
  - Single platform (amd64) for faster builds
- **Full-featured image**: `ghcr.io/makercorn/nbedr:latest-full`, `ghcr.io/makercorn/nbedr:full`
  - Multi-platform (amd64, arm64) - **Full Mac ARM support**
  - Includes cloud providers, vector stores, document processing, local LLM support

## Performance Improvements

### Build Speed
- **Minimal builds**: ~60% faster than previous full builds
- **Layered caching**: Separate cache for minimal vs full builds
- **Binary packages**: Prefer pre-compiled wheels with `--only-binary=:all:`
- **Conditional installation**: Only install requested features

### Image Size
- **Minimal**: ~500MB (vs ~1.2GB previously)
- **Full**: ~1.1GB (optimized from ~1.5GB)
- **Reduced attack surface**: Fewer unnecessary packages

### Cache Strategy
- **Separate cache keys**: `buildcache-minimal` vs `buildcache-full`
- **Layer optimization**: Requirements files copied before application code
- **BuildKit cache mounts**: Persistent pip and apt caches

## Usage Examples

### Development
```bash
# Fast development build
docker build -t nbedr:dev .

# Run with volume mount for code changes
docker run -v $(pwd):/app nbedr:dev
```

### Production
```bash
# Production with cloud providers
docker build --build-arg INSTALL_CLOUD=true -t nbedr:prod .

# Run with environment configuration
docker run -e OPENAI_API_KEY=xxx -e VECTOR_DB_TYPE=faiss nbedr:prod
```

### Local LLM Setup
```bash
# Build with local LLM support
docker build --build-arg INSTALL_LOCAL_LLM=true -t nbedr:local .

# Run with local model server
docker run --network host nbedr:local
```

### Mac ARM Users
```bash
# Use the pre-built multi-platform full image
docker pull ghcr.io/makercorn/nbedr:latest-full

# Or build locally for ARM64
docker build --platform linux/arm64 \
  --build-arg INSTALL_CLOUD=true \
  --build-arg INSTALL_VECTOR_STORES=true \
  --build-arg INSTALL_DOCUMENTS=true \
  --build-arg INSTALL_LOCAL_LLM=true \
  -t nbedr:arm64 .
```

## Migration Guide

### From Previous Version
If you were using the old Docker setup:

1. **Minimal functionality**: No changes needed, use default build
2. **Cloud features**: Add `--build-arg INSTALL_CLOUD=true`
3. **Multiple vector stores**: Add `--build-arg INSTALL_VECTOR_STORES=true`
4. **Document processing**: Add `--build-arg INSTALL_DOCUMENTS=true`

### Manual Installation
You can also install features in a running container:
```bash
# Enter container
docker exec -it nbedr-container bash

# Install additional features
pip install -r requirements-cloud.txt
pip install -r requirements-vector-stores.txt
```

## Troubleshooting

### Build Failures
- **Binary package unavailable**: Remove `--only-binary=:all:` fallback compiles from source
- **Cache issues**: Clear BuildKit cache with `docker builder prune`
- **Platform issues**: Use `--platform linux/amd64` for M1 Macs

### Runtime Issues
- **Missing dependencies**: Check build args and installed features
- **Import errors**: Verify the required feature was enabled during build
- **Version conflicts**: Use separate builds for different feature sets

## Best Practices

1. **Use minimal builds** for development and testing
2. **Enable only needed features** to minimize build time and image size
3. **Use multi-stage builds** for production deployments
4. **Leverage cache layers** by keeping requirements changes minimal
5. **Test builds locally** before pushing to CI

## Future Enhancements

- **Multi-architecture support** for additional platforms
- **Distroless base images** for enhanced security
- **Dynamic feature detection** based on configuration
- **Build matrix testing** for all feature combinations