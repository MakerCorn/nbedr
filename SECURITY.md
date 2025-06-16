# Security Policy

## Supported Versions

We actively support security updates for the following versions of nBedR:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in nBedR, please report it privately to our security team.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Send an email to: **security@makercorn.com** (or create a private GitHub security advisory)
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Progress Updates**: We will keep you informed of our progress throughout the investigation
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Security Measures

nBedR implements several security measures:

#### Code Security
- **Static Analysis**: Automated security scanning with Bandit
- **Dependency Scanning**: Regular vulnerability checks with Safety and Snyk
- **Type Safety**: Comprehensive MyPy type checking
- **Input Validation**: Strict validation of all external inputs

#### Infrastructure Security
- **Container Security**: Non-root user execution, read-only filesystems
- **Kubernetes Hardening**: Security contexts, seccomp profiles, capability dropping
- **Network Security**: Least-privilege access patterns
- **Secrets Management**: Proper secret handling and rotation

#### CI/CD Security
- **Automated Scanning**: Multi-layer security scanning in CI/CD pipeline
- **Dependency Updates**: Automated dependency updates via Dependabot
- **Image Scanning**: Container vulnerability scanning with Trivy
- **Code Analysis**: Continuous security analysis with CodeQL

### Security Best Practices for Users

When deploying nBedR, follow these security best practices:

#### Environment Configuration
- Use strong, unique API keys for all providers
- Implement proper secret management (Kubernetes secrets, AWS Secrets Manager, etc.)
- Enable audit logging for all operations
- Use HTTPS/TLS for all network communications

#### Access Control
- Implement least-privilege access principles
- Use IAM roles instead of access keys where possible
- Regularly rotate credentials and API keys
- Monitor access patterns and unusual activities

#### Data Protection
- Encrypt sensitive data at rest and in transit
- Implement proper data retention policies
- Use secure channels for document processing
- Regular backup and recovery testing

#### Monitoring and Alerting
- Enable comprehensive logging and monitoring
- Set up alerts for security events
- Regular security audits and assessments
- Keep all components updated to latest versions

## Security Updates

Security updates are distributed through:

- **GitHub Releases**: All security patches are tagged and released
- **Container Images**: Updated images pushed to GitHub Container Registry
- **Documentation**: Security advisories published in repository
- **Email Notifications**: Critical security updates sent to maintainers

## Security Tools Integration

nBedR integrates with various security tools:

- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning
- **Snyk**: Comprehensive vulnerability management
- **Trivy**: Container and infrastructure scanning
- **CodeQL**: Semantic code analysis
- **Dependabot**: Automated dependency updates

## Compliance and Standards

nBedR follows security standards and best practices:

- **OWASP Top 10**: Protection against common web vulnerabilities
- **CIS Benchmarks**: Container and Kubernetes security hardening
- **NIST Cybersecurity Framework**: Comprehensive security approach
- **12-Factor App**: Secure application development principles

## Bug Bounty Program

We currently do not have a formal bug bounty program, but we welcome responsible disclosure of security vulnerabilities. Security researchers who identify and report vulnerabilities will be acknowledged in our security advisories (with permission).

## Contact Information

For security-related questions or concerns:

- **Security Email**: security@makercorn.com
- **GitHub Security Advisories**: [Private reporting](https://github.com/MakerCorn/nbedr/security/advisories/new)
- **General Questions**: Create a public issue for non-security related questions

---

**Last Updated**: June 2025
**Version**: 1.0

Thank you for helping keep nBedR secure!