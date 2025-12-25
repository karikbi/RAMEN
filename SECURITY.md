# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of RAMEN seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Disclose Publicly
Please do not create a public GitHub issue for security vulnerabilities.

### 2. Report Privately
Send a detailed report to: **[your-email@example.com]**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline
- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (critical issues within 7 days)

### 4. Disclosure Policy
We follow coordinated disclosure:
- We will work with you to understand and fix the issue
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will publicly disclose the vulnerability after a fix is released

## Security Best Practices

### API Keys and Secrets
- Never commit API keys or secrets to the repository
- Use environment variables or GitHub Secrets
- Rotate keys regularly
- Use minimal required permissions

### GitHub Actions
- Review workflow files for security issues
- Use pinned versions for actions (e.g., `actions/checkout@v4.1.0`)
- Limit workflow permissions to minimum required
- Validate all inputs

### Dependencies
- Regularly update dependencies
- Review security advisories for dependencies
- Use `pip-audit` or similar tools to scan for vulnerabilities

### Data Handling
- Validate all external data sources
- Sanitize user inputs
- Use HTTPS for all API communications
- Implement rate limiting

## Known Security Considerations

### API Rate Limits
The pipeline respects API rate limits to avoid abuse. Do not modify rate limiting code.

### Image Processing
Images are processed locally and validated before storage. Malicious images are rejected.

### Storage Access
R2 storage uses signed URLs with expiration. Never expose R2 credentials publicly.

## Security Updates

Security updates will be announced via:
- GitHub Security Advisories
- Release notes in CHANGELOG.md
- GitHub Discussions (if critical)

## Contact

For security-related questions: **[your-email@example.com]**

For general questions: Use [GitHub Discussions](https://github.com/yourusername/RAMEN/discussions)
