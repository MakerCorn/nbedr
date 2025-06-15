# Python 3.11 Default Environment Setup

This document summarizes the changes made to set Python 3.11 as the default environment for nBedR.

## Changes Made

### 1. Project Configuration Files

#### `.python-version` (NEW)
- Created to specify Python 3.11 as the default version
- Used by pyenv and other Python version management tools

#### `pyproject.toml` (VERIFIED)
- ✅ `requires-python = ">=3.11"` already set
- ✅ `python_version = "3.11"` in mypy configuration already set

#### `deployment/docker/Dockerfile` (VERIFIED)
- ✅ Already uses `FROM python:3.11-slim`

#### `.github/workflows/ci.yml` (VERIFIED)
- ✅ Already tests with Python 3.11, 3.12, and 3.13

### 2. Runtime Version Checks

#### `nbedr.py` (UPDATED)
- Added Python version check at startup
- Exits with error message if Python < 3.11

#### `cli/main.py` (UPDATED)
- Added Python version check before imports
- Provides clear error message for version mismatch

### 3. Development Tools

#### `scripts/check_python.py` (NEW)
- Standalone script to check Python version compatibility
- Provides installation guidance for Python 3.11+

#### `scripts/setup_dev.sh` (NEW)
- Automated development environment setup
- Detects and uses Python 3.11+ automatically
- Creates virtual environment and installs dependencies
- Runs initial code quality checks

### 4. Documentation Updates

#### `docs/DEVELOPMENT.md` (NEW)
- Comprehensive development setup guide
- Python 3.11+ installation instructions for multiple platforms
- IDE configuration recommendations
- Troubleshooting guide

#### `README.md` (UPDATED)
- Updated installation instructions to mention Python 3.11
- Added reference to detailed development guide
- Added note about .python-version file

### 5. Dependencies (VERIFIED)

#### `requirements.txt` (VERIFIED)
- ✅ Already includes `types-requests>=2.31.0` for mypy

#### `pyproject.toml` (VERIFIED)
- ✅ Already includes `types-requests>=2.31.0` in dev dependencies

## Verification

All configurations have been verified to ensure:

1. **Consistent Python 3.11 requirement** across all configuration files
2. **Runtime version checks** prevent execution on incompatible Python versions
3. **Development tools** automatically detect and use Python 3.11+
4. **CI/CD pipeline** tests with Python 3.11, 3.12, and 3.13
5. **Documentation** provides clear setup instructions

## Usage

### For New Developers

1. **Check Python version:**
   ```bash
   python scripts/check_python.py
   ```

2. **Automated setup:**
   ```bash
   ./scripts/setup_dev.sh
   ```

3. **Manual setup:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -e .[dev,all]
   ```

### For Users

The application will automatically check Python version and provide clear error messages if Python 3.11+ is not available.

### For CI/CD

The GitHub Actions workflow already tests with Python 3.11+ and will catch any compatibility issues.

## Benefits

1. **Consistent Environment**: All developers and deployments use Python 3.11+
2. **Clear Error Messages**: Users get helpful guidance when using incompatible Python versions
3. **Automated Setup**: Development environment setup is streamlined
4. **Future-Proof**: Ready for Python 3.12 and 3.13 adoption
5. **Type Safety**: Full mypy compatibility with modern Python features

## Migration Notes

- Existing installations will continue to work if already using Python 3.11+
- Users with Python < 3.11 will get clear upgrade instructions
- No breaking changes to the API or functionality
- All existing configuration files remain compatible
