# Container Security Policy

## Trusted Container Registries

nBedR only uses images from the following trusted registries:

### Primary Registries (Approved)
- **GitHub Container Registry (GHCR)**: `ghcr.io/`
  - Used for: nBedR application images
  - Security: Integrated with GitHub's security scanning and OIDC
  - Example: `ghcr.io/makercorn/nbedr:v1.0.0`

- **Docker Hub Official Images**: `docker.io/library/`
  - Used for: Base images (python, postgres)
  - Security: Official images maintained by Docker Inc.
  - Example: `docker.io/library/python:3.11-slim-bookworm`

- **Google Container Registry (GCR)**: `gcr.io/distroless/`
  - Used for: Distroless base images
  - Security: Minimal attack surface, no shell or package manager
  - Example: `gcr.io/distroless/python3-debian12:latest`

### Secondary Registries (Limited Use)
- **PostgreSQL Extensions**: `docker.io/pgvector/`
  - Used for: pgvector PostgreSQL extension
  - Security: Official pgvector project repository
  - Restriction: Only specific tagged versions, no :latest
  - Example: `docker.io/pgvector/pgvector:0.8.0-pg16`

### Prohibited Registries
- **Private or unknown registries** without security scanning
- **Registries without HTTPS/TLS** encryption
- **Self-hosted registries** without proper security validation
- **Docker Hub unofficial images** (those not in library/ namespace)

## Image Security Requirements

### Base Image Standards
1. **Use minimal base images**: Prefer distroless > alpine > slim > full
2. **Pin to specific versions**: Never use `:latest` tags in production
3. **Regular updates**: Update base images monthly or when security patches available
4. **Vulnerability scanning**: All images must pass security scans

### Security Contexts
1. **Non-root execution**: All containers run as non-root users
2. **High UIDs**: Use UIDs > 10000 for better security isolation
3. **Read-only filesystems**: Enable where possible with writable volume mounts
4. **Capability dropping**: Drop all Linux capabilities unless specifically needed
5. **Seccomp profiles**: Use RuntimeDefault seccomp profiles

### Image Tags and Versioning
```yaml
# ✅ GOOD - Specific versions
image: ghcr.io/makercorn/nbedr:v1.0.0
image: docker.io/pgvector/pgvector:0.8.0-pg16
image: gcr.io/distroless/python3-debian12@sha256:abc123...

# ❌ BAD - Mutable tags
image: ghcr.io/makercorn/nbedr:latest
image: docker.io/pgvector/pgvector:latest
image: python:3.11
```

## Docker Build Security

### Multi-stage Builds
- Use separate builder and runtime stages
- Copy only necessary artifacts to final stage
- Remove build tools and dependencies from runtime image

### Package Management
- Pin setuptools version to >= 75.6.0 for CVE fixes
- Remove vulnerable system packages (perl, tar, ncurses)
- Use `--only-binary=:all:` when possible to avoid compilation
- Regular dependency updates via Dependabot

### Build Arguments
```dockerfile
# Security-focused build arguments
ARG BUILDKIT_INLINE_CACHE=1
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Feature flags for minimal builds
ARG INSTALL_CLOUD=false
ARG INSTALL_VECTOR_STORES=false
ARG INSTALL_DOCUMENTS=false
ARG INSTALL_LOCAL_LLM=false
ARG INSTALL_HEAVY=false
```

## Kubernetes Security

### Pod Security Standards
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001

containers:
- securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    readOnlyRootFilesystem: true
    allowPrivilegeEscalation: false
    capabilities:
      drop: ["ALL"]
    seccompProfile:
      type: RuntimeDefault
```

### Image Pull Policies
```yaml
# ✅ GOOD - Prevent latest tag issues
imagePullPolicy: IfNotPresent

# ❌ BAD - Can cause inconsistency
imagePullPolicy: Always
```

## CI/CD Security

### Container Scanning
- **Trivy**: Filesystem, configuration, and container scanning
- **CodeQL**: Source code analysis
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning

### Build Security
- Multi-platform builds for compatibility
- Separate minimal and full-featured images
- Distroless images for maximum security
- Automated dependency updates

### Registry Security
- OIDC authentication with GitHub
- Signed container images (future enhancement)
- Regular security scanning of pushed images

## Monitoring and Compliance

### Security Monitoring
- Weekly Dependabot updates
- Automated vulnerability scanning
- Container registry health checks
- Security audit logs

### Compliance Standards
- **OWASP Container Security Top 10**
- **CIS Kubernetes Benchmark**
- **NIST Cybersecurity Framework**
- **12-Factor App Security Principles**

## Emergency Response

### Security Incident Response
1. **Immediate**: Stop pulling/using affected images
2. **Assessment**: Determine impact scope and severity
3. **Mitigation**: Deploy patched images or workarounds
4. **Communication**: Notify stakeholders and users
5. **Prevention**: Update policies to prevent recurrence

### Image Recall Process
1. **Identification**: Vulnerable image detected
2. **Documentation**: CVE details and impact assessment
3. **Replacement**: Build and test patched image
4. **Deployment**: Coordinate rolling update
5. **Verification**: Confirm vulnerability resolved

## Contact and Reporting

For security issues related to container images:
- **Security Email**: security@makercorn.com
- **GitHub Security Advisory**: [Private Reporting](https://github.com/MakerCorn/nbedr/security/advisories/new)

---

**Last Updated**: June 2025  
**Policy Version**: 1.0  
**Next Review**: September 2025