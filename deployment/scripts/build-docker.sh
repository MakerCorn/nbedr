#!/bin/bash
# Build and push nBedR Docker image

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/deployment/docker"

# Default values
REGISTRY="${REGISTRY:-nbedr}"
IMAGE_NAME="${IMAGE_NAME:-nbedr}"
TAG="${TAG:-latest}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(git describe --tags --always 2>/dev/null || echo "dev")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build and optionally push nBedR Docker image

OPTIONS:
    -r, --registry REGISTRY     Docker registry (default: nbedr)
    -i, --image IMAGE          Image name (default: nbedr)
    -t, --tag TAG              Image tag (default: latest)
    -p, --push                 Push image to registry
    --no-cache                 Build without cache
    --platform PLATFORM       Target platform (e.g., linux/amd64,linux/arm64)
    -h, --help                 Show this help

EXAMPLES:
    # Build locally
    $0

    # Build and push to Docker Hub
    $0 -r dockerhub-user -p

    # Build for Azure Container Registry
    $0 -r myregistry.azurecr.io -i nbedr -t v1.0.0 -p

    # Build for Amazon ECR
    $0 -r 123456789.dkr.ecr.us-west-2.amazonaws.com -i nbedr -p

    # Build for Google Container Registry
    $0 -r gcr.io/my-project -i nbedr -p

    # Multi-platform build
    $0 --platform linux/amd64,linux/arm64 -p

EOF
}

# Parse arguments
PUSH=false
NO_CACHE=""
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Build full image name
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

log "Building nBedR Docker image"
log "Registry: ${REGISTRY}"
log "Image: ${IMAGE_NAME}"
log "Tag: ${TAG}"
log "Full image: ${FULL_IMAGE}"
log "Version: ${VERSION}"
log "VCS Ref: ${VCS_REF}"
log "Build Date: ${BUILD_DATE}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    error "Docker is not running or not accessible"
fi

# Navigate to project root
cd "${PROJECT_ROOT}"

# Build the image
log "Building Docker image..."
BUILD_CMD="docker build \
    ${NO_CACHE} \
    ${PLATFORM} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --build-arg VERSION=${VERSION} \
    --build-arg VCS_REF=${VCS_REF} \
    -f deployment/docker/Dockerfile \
    -t ${FULL_IMAGE} \
    ."

echo "Build command: ${BUILD_CMD}"
eval ${BUILD_CMD}

if [[ $? -eq 0 ]]; then
    log "Successfully built ${FULL_IMAGE}"
else
    error "Failed to build Docker image"
fi

# Test the image
log "Testing Docker image..."
if docker run --rm ${FULL_IMAGE} --help >/dev/null 2>&1; then
    log "Image test passed"
else
    warn "Image test failed, but continuing..."
fi

# Push if requested
if [[ "${PUSH}" == "true" ]]; then
    log "Pushing image to registry..."
    
    # Check if we need to login (for cloud registries)
    case "${REGISTRY}" in
        *.azurecr.io)
            warn "Make sure you're logged in to Azure Container Registry: az acr login --name <registry-name>"
            ;;
        *.dkr.ecr.*.amazonaws.com)
            warn "Make sure you're logged in to Amazon ECR: aws ecr get-login-password | docker login --username AWS --password-stdin ${REGISTRY}"
            ;;
        gcr.io/*|*.gcr.io)
            warn "Make sure you're logged in to Google Container Registry: gcloud auth configure-docker"
            ;;
    esac
    
    docker push ${FULL_IMAGE}
    
    if [[ $? -eq 0 ]]; then
        log "Successfully pushed ${FULL_IMAGE}"
    else
        error "Failed to push Docker image"
    fi
fi

# Also tag as latest if not already
if [[ "${TAG}" != "latest" ]]; then
    LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"
    log "Tagging as latest: ${LATEST_IMAGE}"
    docker tag ${FULL_IMAGE} ${LATEST_IMAGE}
    
    if [[ "${PUSH}" == "true" ]]; then
        log "Pushing latest tag..."
        docker push ${LATEST_IMAGE}
    fi
fi

log "Build complete!"
log "Image: ${FULL_IMAGE}"

# Show image size
IMAGE_SIZE=$(docker images --format "table {{.Size}}" ${FULL_IMAGE} | tail -1)
log "Image size: ${IMAGE_SIZE}"

# Security scan (if available)
if command -v docker &> /dev/null && docker --help | grep -q scan; then
    log "Running security scan..."
    docker scan ${FULL_IMAGE} || warn "Security scan failed or not available"
fi