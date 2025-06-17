#!/bin/bash
# Dynamic Kubernetes Deployment Script with Secure Image Tag Management
# This script solves the hard-coded image tag problem while maintaining security

set -euo pipefail

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-gke}"
IMAGE_TAG="${IMAGE_TAG:-}"
NAMESPACE="${NAMESPACE:-nbedr}"
DRY_RUN="${DRY_RUN:-false}"
SECURITY_SCAN="${SECURITY_SCAN:-true}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to determine image tag dynamically
determine_image_tag() {
    if [[ -n "$IMAGE_TAG" ]]; then
        echo "$IMAGE_TAG"
        return
    fi

    # Determine tag based on environment and Git context
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Production: Use Git tag if available, otherwise error
        if git describe --tags --exact-match 2>/dev/null; then
            git describe --tags --exact-match
        else
            log_error "Production deployment requires a Git tag"
            exit 1
        fi
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        # Staging: Use branch name + short SHA
        BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9]/-/g')
        SHORT_SHA=$(git rev-parse --short HEAD)
        echo "${BRANCH}-${SHORT_SHA}"
    else
        # Development: Use short SHA
        git rev-parse --short HEAD
    fi
}

# Function to validate image exists
validate_image() {
    local image_ref="$1"
    log_info "Validating image exists: $image_ref"
    
    if docker manifest inspect "$image_ref" >/dev/null 2>&1; then
        log_success "Image validation passed"
        return 0
    else
        log_error "Image not found: $image_ref"
        return 1
    fi
}

# Function to scan image for vulnerabilities
scan_image_security() {
    local image_ref="$1"
    
    if [[ "$SECURITY_SCAN" != "true" ]]; then
        log_warning "Security scanning disabled"
        return 0
    fi

    log_info "Scanning image for vulnerabilities: $image_ref"
    
    # Use trivy to scan the image
    if command -v trivy >/dev/null 2>&1; then
        if trivy image --exit-code 1 --severity HIGH,CRITICAL "$image_ref"; then
            log_success "Security scan passed"
            return 0
        else
            log_error "Security scan failed - high/critical vulnerabilities found"
            return 1
        fi
    else
        log_warning "Trivy not available, skipping security scan"
        return 0
    fi
}

# Function to apply Kubernetes manifests with dynamic image tag
deploy_to_kubernetes() {
    local cloud_provider="$1"
    local image_tag="$2"
    local namespace="$3"
    
    local deployment_dir="deployment/kubernetes/${cloud_provider}"
    local image_ref="ghcr.io/makercorn/nbedr:${image_tag}"
    
    log_info "Deploying to $cloud_provider with image: $image_ref"
    
    # Validate deployment directory exists
    if [[ ! -d "$deployment_dir" ]]; then
        log_error "Deployment directory not found: $deployment_dir"
        exit 1
    fi
    
    # Validate image exists and scan for security
    if ! validate_image "$image_ref"; then
        exit 1
    fi
    
    if ! scan_image_security "$image_ref"; then
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Use kustomize to dynamically set the image tag
    cd "$deployment_dir"
    
    # Create a temporary kustomization.yaml with the correct image tag
    cp kustomization.yaml kustomization.yaml.backup
    
    # Update the image tag dynamically
    kustomize edit set image "nbedr=ghcr.io/makercorn/nbedr:${image_tag}"
    
    # Apply the manifests
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - showing what would be applied:"
        kustomize build . | kubectl apply --dry-run=client -f -
    else
        log_info "Applying Kubernetes manifests..."
        kustomize build . | kubectl apply -f -
        
        # Wait for deployment to be ready
        kubectl rollout status deployment/nbedr-deployment -n "$namespace" --timeout=300s
        
        # Verify deployment
        kubectl get pods -n "$namespace" -l app.kubernetes.io/name=nbedr
    fi
    
    # Restore original kustomization.yaml
    mv kustomization.yaml.backup kustomization.yaml
    
    cd - >/dev/null
}

# Function to select appropriate image variant based on environment
select_image_variant() {
    local environment="$1"
    local base_tag="$2"
    
    case "$environment" in
        production)
            # Production: Use distroless for maximum security
            echo "${base_tag}-distroless"
            ;;
        staging)
            # Staging: Use full-featured image for testing
            echo "${base_tag}-full"
            ;;
        development)
            # Development: Use minimal image for speed
            echo "$base_tag"
            ;;
        *)
            echo "$base_tag"
            ;;
    esac
}

# Main deployment logic
main() {
    log_info "Starting deployment to $ENVIRONMENT environment on $CLOUD_PROVIDER"
    
    # Determine the base image tag
    BASE_TAG=$(determine_image_tag)
    log_info "Base image tag: $BASE_TAG"
    
    # Select appropriate image variant for environment
    FINAL_TAG=$(select_image_variant "$ENVIRONMENT" "$BASE_TAG")
    log_info "Final image tag: $FINAL_TAG"
    
    # Deploy to Kubernetes
    deploy_to_kubernetes "$CLOUD_PROVIDER" "$FINAL_TAG" "$NAMESPACE"
    
    log_success "Deployment completed successfully!"
    log_info "Image deployed: ghcr.io/makercorn/nbedr:$FINAL_TAG"
    log_info "Environment: $ENVIRONMENT"
    log_info "Cloud Provider: $CLOUD_PROVIDER"
    log_info "Namespace: $NAMESPACE"
}

# Help function
show_help() {
    cat << EOF
Dynamic Kubernetes Deployment Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -e, --environment    Environment to deploy to (development|staging|production)
    -c, --cloud         Cloud provider (gke|eks|aks)
    -t, --tag           Override image tag (optional)
    -n, --namespace     Kubernetes namespace (default: nbedr)
    -d, --dry-run       Perform dry run without applying changes
    -s, --skip-scan     Skip security scanning
    -h, --help          Show this help message

EXAMPLES:
    # Deploy to staging on GKE with auto-generated tag
    $0 --environment staging --cloud gke

    # Deploy to production with specific tag
    $0 --environment production --cloud eks --tag v1.2.3

    # Dry run deployment
    $0 --environment development --cloud aks --dry-run

ENVIRONMENT VARIABLES:
    ENVIRONMENT         Default environment (development|staging|production)
    CLOUD_PROVIDER      Default cloud provider (gke|eks|aks)
    IMAGE_TAG           Override image tag
    NAMESPACE           Kubernetes namespace
    DRY_RUN            Perform dry run (true|false)
    SECURITY_SCAN      Enable security scanning (true|false)

IMAGE TAG STRATEGY:
    - Development: Uses Git SHA (e.g., abc1234)
    - Staging: Uses branch-SHA (e.g., feature-auth-abc1234)
    - Production: Requires Git tag (e.g., v1.2.3)

IMAGE VARIANTS:
    - Development: Minimal image (fastest)
    - Staging: Full-featured image (all features)
    - Production: Distroless image (maximum security)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--cloud)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -s|--skip-scan)
            SECURITY_SCAN="false"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required tools
for tool in kubectl kustomize git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_error "Required tool not found: $tool"
        exit 1
    fi
done

# Run main deployment
main