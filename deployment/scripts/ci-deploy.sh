#!/bin/bash
# CI/CD Integration Script for Dynamic Image Tag Management
# Solves hard-coded tag problems in automated pipelines

set -euo pipefail

# Function to generate semantic tags for CI/CD
generate_ci_tags() {
    local base_registry="ghcr.io/makercorn/nbedr"
    local tags=()
    
    # Always include SHA-based tag for traceability
    local short_sha=$(echo "${GITHUB_SHA:-$(git rev-parse HEAD)}" | cut -c1-8)
    tags+=("${base_registry}:sha-${short_sha}")
    
    # Branch-specific tags
    if [[ "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
        # Git tag push - use semantic versioning
        local tag_name="${GITHUB_REF_NAME:-$(git describe --tags --exact-match 2>/dev/null || echo '')}"
        if [[ -n "$tag_name" ]]; then
            tags+=("${base_registry}:${tag_name}")
            # Also tag as latest for release tags
            if [[ "$tag_name" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                tags+=("${base_registry}:latest")
            fi
        fi
    elif [[ "${GITHUB_REF_NAME:-}" == "main" ]]; then
        # Main branch - use latest and main tags
        tags+=("${base_registry}:latest")
        tags+=("${base_registry}:main")
        tags+=("${base_registry}:main-${short_sha}")
    elif [[ "${GITHUB_REF_NAME:-}" == "develop" ]]; then
        # Develop branch - use develop tag
        tags+=("${base_registry}:develop")
        tags+=("${base_registry}:develop-${short_sha}")
    else
        # Feature branches - use branch name
        local branch_name="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
        local safe_branch=$(echo "$branch_name" | sed 's/[^a-zA-Z0-9]/-/g' | tr '[:upper:]' '[:lower:]')
        tags+=("${base_registry}:${safe_branch}")
        tags+=("${base_registry}:${safe_branch}-${short_sha}")
    fi
    
    # Print tags as comma-separated string for Docker build
    IFS=','
    echo "${tags[*]}"
}

# Function to generate tags for different image variants
generate_variant_tags() {
    local variant="$1"  # minimal, full, distroless
    local base_tags="$2"
    
    # Convert comma-separated tags to array
    IFS=',' read -ra tag_array <<< "$base_tags"
    
    local variant_tags=()
    for tag in "${tag_array[@]}"; do
        if [[ "$variant" == "minimal" ]]; then
            # Minimal variant uses base tags as-is
            variant_tags+=("$tag")
        else
            # Add variant suffix
            variant_tags+=("${tag}-${variant}")
        fi
    done
    
    # Return as comma-separated string
    IFS=','
    echo "${variant_tags[*]}"
}

# Function to update Kubernetes manifests with dynamic tags
update_k8s_manifests() {
    local environment="$1"
    local image_tag="$2"
    
    log_info "Updating Kubernetes manifests for $environment with tag: $image_tag"
    
    # Determine which image variant to use based on environment
    local variant_tag
    case "$environment" in
        production)
            variant_tag="${image_tag}-distroless"
            ;;
        staging)
            variant_tag="${image_tag}-full"
            ;;
        *)
            variant_tag="$image_tag"
            ;;
    esac
    
    # Update all cloud provider configurations
    for cloud in gke eks aks; do
        local kustomize_file="deployment/kubernetes/${cloud}/kustomization.yaml"
        if [[ -f "$kustomize_file" ]]; then
            log_info "Updating $kustomize_file with image tag: $variant_tag"
            cd "deployment/kubernetes/${cloud}"
            kustomize edit set image "nbedr=ghcr.io/makercorn/nbedr:${variant_tag}"
            cd - >/dev/null
        fi
    done
}

# Main CI function
main() {
    echo "=== CI/CD Dynamic Tag Generation ==="
    
    # Generate base tags
    local base_tags
    base_tags=$(generate_ci_tags)
    echo "Base tags: $base_tags"
    
    # Generate variant-specific tags
    local minimal_tags
    local full_tags  
    local distroless_tags
    
    minimal_tags=$(generate_variant_tags "minimal" "$base_tags")
    full_tags=$(generate_variant_tags "full" "$base_tags") 
    distroless_tags=$(generate_variant_tags "distroless" "$base_tags")
    
    echo "Minimal tags: $minimal_tags"
    echo "Full tags: $full_tags"
    echo "Distroless tags: $distroless_tags"
    
    # Export for use in GitHub Actions
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        echo "minimal-tags=${minimal_tags}" >> "$GITHUB_OUTPUT"
        echo "full-tags=${full_tags}" >> "$GITHUB_OUTPUT"
        echo "distroless-tags=${distroless_tags}" >> "$GITHUB_OUTPUT"
        
        # Also export primary tag for deployments
        local primary_tag
        primary_tag=$(echo "$base_tags" | cut -d',' -f1 | sed 's/.*://')
        echo "primary-tag=${primary_tag}" >> "$GITHUB_OUTPUT"
    fi
}

# Logging function
log_info() {
    echo "[INFO] $1"
}

# Run main function
main "$@"