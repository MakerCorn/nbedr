#!/bin/bash
# Deploy nBedR to Kubernetes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/deployment/kubernetes"

# Default values
PLATFORM=""
NAMESPACE="nbedr"
DRY_RUN=false
WAIT_TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS] PLATFORM

Deploy nBedR to Kubernetes cluster

PLATFORMS:
    aks                        Azure Kubernetes Service
    eks                        Amazon Elastic Kubernetes Service  
    gke                        Google Kubernetes Engine
    base                       Generic Kubernetes (no cloud-specific features)

OPTIONS:
    -n, --namespace NAMESPACE  Kubernetes namespace (default: nbedr)
    -d, --dry-run             Show what would be deployed without applying
    -w, --wait TIMEOUT        Wait for deployment to be ready (default: 300s)
    --skip-secrets            Skip secrets creation (use existing)
    --skip-build              Skip image build step
    -h, --help                Show this help

EXAMPLES:
    # Deploy to AKS
    $0 aks

    # Deploy to EKS with dry-run
    $0 -d eks

    # Deploy to GKE with custom namespace
    $0 -n my-nbedr gke

    # Deploy to generic Kubernetes
    $0 base

PREREQUISITES:
    - kubectl configured for target cluster
    - kustomize installed
    - Docker image built and pushed to accessible registry
    - Secrets configured (see deployment/kubernetes/*/kustomization.yaml)

EOF
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Please install kubectl."
    fi
    
    # Check kustomize
    if ! command -v kustomize &> /dev/null; then
        warn "kustomize not found. Trying kubectl kustomize..."
        if ! kubectl kustomize --help &> /dev/null; then
            error "Neither kustomize nor kubectl kustomize available. Please install kustomize."
        fi
        KUSTOMIZE_CMD="kubectl kustomize"
    else
        KUSTOMIZE_CMD="kustomize build"
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Please configure kubectl."
    fi
    
    local cluster_info=$(kubectl cluster-info | head -1)
    info "Connected to: ${cluster_info}"
}

validate_platform() {
    local platform=$1
    case "${platform}" in
        aks|eks|gke|base)
            return 0
            ;;
        *)
            error "Invalid platform: ${platform}. Must be one of: aks, eks, gke, base"
            ;;
    esac
}

deploy_platform() {
    local platform=$1
    local platform_dir="${K8S_DIR}/${platform}"
    
    if [[ ! -d "${platform_dir}" ]]; then
        error "Platform directory not found: ${platform_dir}"
    fi
    
    log "Deploying to ${platform} platform..."
    
    cd "${platform_dir}"
    
    # Generate manifests
    log "Generating Kubernetes manifests..."
    local manifests_file="/tmp/nbedr-${platform}-manifests.yaml"
    ${KUSTOMIZE_CMD} . > "${manifests_file}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        info "Dry run mode - showing generated manifests:"
        echo "=================================="
        cat "${manifests_file}"
        echo "=================================="
        info "Dry run mode - would apply to cluster: $(kubectl config current-context)"
        return 0
    fi
    
    # Apply manifests
    log "Applying manifests to cluster..."
    kubectl apply -f "${manifests_file}"
    
    # Wait for deployment if requested
    if [[ -n "${WAIT_TIMEOUT}" ]]; then
        log "Waiting for deployment to be ready (timeout: ${WAIT_TIMEOUT})..."
        kubectl wait --for=condition=available deployment/nbedr-deployment \
            -n "${NAMESPACE}" --timeout="${WAIT_TIMEOUT}" || warn "Timeout waiting for deployment"
    fi
    
    # Show status
    log "Deployment status:"
    kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=nbedr
    
    log "Service status:"
    kubectl get services -n "${NAMESPACE}" -l app.kubernetes.io/name=nbedr
    
    # Show useful commands
    info "Useful commands:"
    echo "  # View logs:"
    echo "  kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/name=nbedr -f"
    echo ""
    echo "  # Scale deployment:"
    echo "  kubectl scale deployment/nbedr-deployment -n ${NAMESPACE} --replicas=5"
    echo ""
    echo "  # Update configuration:"
    echo "  kubectl edit configmap/nbedr-config -n ${NAMESPACE}"
    echo ""
    echo "  # Delete deployment:"
    echo "  kubectl delete -f ${manifests_file}"
    
    # Cleanup temp file
    rm -f "${manifests_file}"
}

show_platform_specific_notes() {
    local platform=$1
    
    case "${platform}" in
        aks)
            info "AKS-specific notes:"
            echo "  - Update ACR registry in deployment/kubernetes/aks/kustomization.yaml"
            echo "  - Configure Azure Storage credentials in secrets"
            echo "  - Ensure AKS cluster has Azure File CSI driver enabled"
            ;;
        eks)
            info "EKS-specific notes:"
            echo "  - Update ECR registry in deployment/kubernetes/eks/kustomization.yaml"
            echo "  - Configure IAM roles for service accounts (IRSA)"
            echo "  - Ensure EKS cluster has EFS CSI driver installed"
            echo "  - Update EFS file system ID in pvc-patch.yaml"
            ;;
        gke)
            info "GKE-specific notes:"
            echo "  - Update GCR registry in deployment/kubernetes/gke/kustomization.yaml"
            echo "  - Configure Workload Identity"
            echo "  - Ensure GKE cluster has Filestore CSI driver enabled"
            echo "  - Update GCP service account keys"
            ;;
        base)
            info "Generic Kubernetes notes:"
            echo "  - Update image registry in base configuration"
            echo "  - Configure storage classes for your cluster"
            echo "  - Update persistent volume configurations as needed"
            ;;
    esac
}

# Parse arguments
SKIP_SECRETS=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -w|--wait)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --skip-secrets)
            SKIP_SECRETS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            if [[ -z "${PLATFORM}" ]]; then
                PLATFORM="$1"
            else
                error "Too many arguments"
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "${PLATFORM}" ]]; then
    error "Platform is required. Use -h for help."
fi

validate_platform "${PLATFORM}"

log "Starting nBedR Kubernetes deployment"
log "Platform: ${PLATFORM}"
log "Namespace: ${NAMESPACE}"
log "Dry run: ${DRY_RUN}"

check_prerequisites
show_platform_specific_notes "${PLATFORM}"

# Confirm deployment
if [[ "${DRY_RUN}" == "false" ]]; then
    echo ""
    read -p "Continue with deployment to $(kubectl config current-context)? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled"
        exit 0
    fi
fi

deploy_platform "${PLATFORM}"

log "Deployment completed successfully!"