# Dynamic Image Tagging Strategy

## Problem Statement

Hard-coding image tags like `v1.0.0` in Kubernetes manifests creates several critical issues:

- 🚫 **Manual Updates Required**: Every release requires manual manifest updates
- 🚫 **Pipeline Failures**: CI/CD breaks when tags don't exist yet
- 🚫 **Version Drift**: Different environments may use different versions unintentionally
- 🚫 **Deployment Automation**: Prevents fully automated deployments
- 🚫 **Security Risk**: Using `:latest` fails security scans but hard-coding prevents updates

## Solution: Dynamic Tagging Strategy

Our solution uses **dynamic tag generation** with **security-first principles** while maintaining full automation.

## Tag Strategy by Environment

### 🏗️ Development Environment
```bash
# Pattern: <branch>-<short-sha>
feature-auth-abc1234
bugfix-login-def5678
main-abc1234
```

### 🧪 Staging Environment  
```bash
# Pattern: <tag>-full or <branch>-<sha>-full
develop-abc1234-full
main-def5678-full
v1.2.0-full
```

### 🚀 Production Environment
```bash
# Pattern: <semantic-version>-distroless
v1.2.0-distroless
v1.2.1-distroless
```

## Image Variants by Environment

| Environment | Image Variant | Security Level | Features |
|-------------|---------------|----------------|----------|
| Development | Minimal | Standard | Basic features, fast builds |
| Staging | Full | Enhanced | All features, full testing |
| Production | Distroless | Maximum | Minimal attack surface |

## Implementation Components

### 1. Dynamic Tag Generation Script

**`deployment/scripts/ci-deploy.sh`**
- Generates semantic tags based on Git context
- Creates variant-specific tags for different image types
- Exports tags for GitHub Actions consumption

```bash
# Usage in CI/CD
./deployment/scripts/ci-deploy.sh
# Outputs: minimal-tags, full-tags, distroless-tags
```

### 2. Deployment Automation Script

**`deployment/scripts/deploy.sh`**
- Handles dynamic tag resolution at deployment time
- Validates image existence before deployment
- Performs security scanning
- Supports multiple cloud providers

```bash
# Usage examples
./deploy.sh --environment staging --cloud gke
./deploy.sh --environment production --cloud eks --tag v1.2.0
./deploy.sh --environment development --dry-run
```

### 3. CI/CD Integration

**`.github/workflows/ci.yml`**
- Builds multiple image variants with dynamic tags
- Uses fallback strategies for tag generation
- Integrates with security scanning

**`.github/workflows/deploy.yml`**
- Automated deployment with tag resolution
- Environment-specific image variant selection
- Comprehensive validation and testing

## Kubernetes Manifest Strategy

### Base Manifests
```yaml
# deployment/kubernetes/base/deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: nbedr
        image: ghcr.io/makercorn/nbedr:latest  # Default fallback
        imagePullPolicy: IfNotPresent
```

### Dynamic Override via Kustomize
```yaml
# kustomization.yaml
images:
- name: nbedr
  newName: ghcr.io/makercorn/nbedr
  # newTag dynamically set by deployment scripts
  newTag: latest
```

### Runtime Tag Override
```bash
# Deployment script sets the actual tag
cd deployment/kubernetes/gke
kustomize edit set image "nbedr=ghcr.io/makercorn/nbedr:${DYNAMIC_TAG}"
kustomize build . | kubectl apply -f -
```

## Security Considerations

### 1. Image Validation
- ✅ **Existence Check**: Verify image exists before deployment
- ✅ **Security Scanning**: Trivy scan for vulnerabilities
- ✅ **Registry Authentication**: OIDC-based authentication
- ✅ **Signature Verification**: Future enhancement with cosign

### 2. Tag Immutability
- ✅ **SHA-based Tags**: Always include commit SHA for traceability
- ✅ **Semantic Versioning**: Production uses only tagged releases
- ✅ **Audit Trail**: Full deployment history with image references

### 3. Environment Isolation
- ✅ **Variant Selection**: Different security levels per environment
- ✅ **Namespace Separation**: Environment-specific namespaces
- ✅ **Access Controls**: RBAC and authentication per environment

## Usage Examples

### Manual Deployment
```bash
# Deploy latest develop to staging
./deployment/scripts/deploy.sh \
  --environment staging \
  --cloud gke

# Deploy specific version to production
./deployment/scripts/deploy.sh \
  --environment production \
  --cloud eks \
  --tag v1.2.0

# Dry run for testing
./deployment/scripts/deploy.sh \
  --environment development \
  --cloud aks \
  --dry-run
```

### GitHub Actions Workflow
```yaml
# Automatic deployment on main branch
- name: Deploy to staging
  if: github.ref == 'refs/heads/main'
  run: |
    ./deployment/scripts/deploy.sh \
      --environment staging \
      --cloud gke
```

### GitOps Integration
```bash
# Update GitOps repository with new image tag
git clone gitops-repo
cd gitops-repo/environments/staging
kustomize edit set image nbedr=ghcr.io/makercorn/nbedr:${NEW_TAG}
git commit -m "Update staging to ${NEW_TAG}"
git push
```

## Migration from Hard-coded Tags

### Step 1: Update Deployment Scripts
```bash
# Install deployment scripts
cp deployment/scripts/* /usr/local/bin/
chmod +x /usr/local/bin/deploy.sh
```

### Step 2: Update CI/CD Pipelines
```yaml
# Replace hard-coded tags with dynamic generation
- name: Generate tags
  id: tags
  run: ./deployment/scripts/ci-deploy.sh

- name: Build image
  uses: docker/build-push-action@v5
  with:
    tags: ${{ steps.tags.outputs.minimal-tags }}
```

### Step 3: Update Kubernetes Manifests
```bash
# Reset to latest as default
sed -i 's/:v[0-9]\+\.[0-9]\+\.[0-9]\+/:latest/g' deployment/kubernetes/*/kustomization.yaml
```

## Troubleshooting

### Common Issues

**Q: Image not found during deployment**
```bash
# Check if image exists
docker buildx imagetools inspect ghcr.io/makercorn/nbedr:${TAG}

# List available tags
gh api repos/makercorn/nbedr/packages/container/nbedr/versions
```

**Q: Deployment script fails with authentication error**
```bash
# Verify Kubernetes authentication
kubectl auth can-i create deployments -n nbedr

# Check cloud provider credentials
gcloud auth list  # GKE
aws sts get-caller-identity  # EKS
az account show  # AKS
```

**Q: Security scan fails on dynamic tags**
```bash
# Run security scan manually
trivy image ghcr.io/makercorn/nbedr:${TAG}

# Skip security scan if needed
export SECURITY_SCAN=false
./deployment/scripts/deploy.sh
```

### Best Practices

1. **Always Test Locally First**
   ```bash
   # Test with dry run
   ./deployment/scripts/deploy.sh --dry-run
   ```

2. **Use Environment Variables**
   ```bash
   export ENVIRONMENT=staging
   export CLOUD_PROVIDER=gke
   ./deployment/scripts/deploy.sh
   ```

3. **Monitor Deployments**
   ```bash
   # Watch deployment progress
   kubectl rollout status deployment/nbedr-deployment -n nbedr-staging
   ```

4. **Maintain Rollback Capability**
   ```bash
   # Quick rollback
   kubectl rollout undo deployment/nbedr-deployment -n nbedr-staging
   ```

## Advanced Configuration

### Custom Tag Patterns
```bash
# Override tag generation logic
export CUSTOM_TAG_PATTERN="${BRANCH}-${BUILD_NUMBER}-${SHORT_SHA}"
```

### Multi-Registry Support
```yaml
# Support multiple registries
images:
- name: nbedr
  newName: ${REGISTRY:-ghcr.io/makercorn}/nbedr
  newTag: ${TAG}
```

### Environment-Specific Overrides
```bash
# Production-specific configurations
if [[ "$ENVIRONMENT" == "production" ]]; then
  export SECURITY_SCAN=true
  export IMAGE_VARIANT=distroless
  export REPLICAS=5
fi
```

## Future Enhancements

- 🔮 **Image Signing**: cosign integration for image verification
- 🔮 **Policy Enforcement**: OPA Gatekeeper for tag compliance
- 🔮 **Automated Rollback**: Failure detection and automatic rollback
- 🔮 **Multi-Cloud**: Cross-cloud deployment orchestration
- 🔮 **Canary Deployments**: Gradual rollout with traffic splitting

---

**Last Updated**: June 2025  
**Version**: 1.0  
**Maintainer**: nBedR Team