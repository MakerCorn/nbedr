# Development Guide

This guide covers setting up a development environment for nBedR.

## Prerequisites

### Python Version

nBedR requires **Python 3.11 or higher**. Python 3.9 and 3.10 are no longer supported.

#### Check Your Python Version

```bash
python3 --version
# or
python3.11 --version
```

#### Install Python 3.11+

**Using pyenv (Recommended):**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.0
pyenv global 3.11.0
```

**Using conda:**
```bash
conda install python=3.11
```

**Using system package manager:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS with Homebrew
brew install python@3.11

# CentOS/RHEL/Fedora
sudo dnf install python3.11 python3.11-venv python3.11-devel
```

**Download from python.org:**
Visit [https://www.python.org/downloads/](https://www.python.org/downloads/)

## Quick Setup

### Automated Setup

Use the provided setup script:

```bash
./scripts/setup_dev.sh
```

This script will:
- Check for Python 3.11+
- Create a virtual environment
- Install all dependencies
- Run initial code quality checks

### Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/nbedr.git
   cd nbedr
   ```

2. **Create virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e .[dev,all]
   ```

4. **Verify installation:**
   ```bash
   python scripts/check_python.py
   python nbedr.py --help
   ```

## Development Workflow

### Code Quality

Run these commands before committing:

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy core/ cli/ --ignore-missing-imports

# Run tests
pytest
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run checks:

```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=cli --cov-report=html

# Run specific test file
pytest tests/unit/test_embedding_utils.py
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html
```

## Project Structure

```
nbedr/
├── cli/                    # Command-line interface
├── core/                   # Core application logic
│   ├── clients/           # Embedding provider clients
│   ├── services/          # Business logic services
│   ├── sources/           # Document source handlers
│   ├── utils/             # Utility functions
│   └── vector_stores/     # Vector database implementations
├── deployment/            # Deployment configurations
├── docs/                  # Documentation
├── scripts/               # Development scripts
├── tests/                 # Test suite
├── templates/             # Prompt templates
├── nbedr.py              # Main entry point
└── pyproject.toml        # Project configuration
```

## Environment Variables

Create a `.env` file for development:

```bash
# Copy example environment file
cp deployment/docker/.env.example .env

# Edit with your settings
vim .env
```

Key environment variables:
- `EMBEDDING_PROVIDER`: openai, azure_openai, aws_bedrock, etc.
- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_DATABASE_TYPE`: faiss, pinecone, chromadb, etc.

## Troubleshooting

### Python Version Issues

If you get Python version errors:

1. Check your Python version: `python3 --version`
2. Install Python 3.11+ using one of the methods above
3. Recreate your virtual environment with the correct Python version

### Import Errors

If you get import errors:

1. Ensure you're in the virtual environment: `source venv/bin/activate`
2. Install in development mode: `pip install -e .`
3. Check your PYTHONPATH includes the project directory

### Dependency Issues

If you have dependency conflicts:

1. Delete the virtual environment: `rm -rf venv`
2. Recreate it: `python3.11 -m venv venv`
3. Reinstall dependencies: `pip install -e .[dev,all]`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests and quality checks
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## IDE Configuration

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- Flake8

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true
}
```

### PyCharm

1. Set Python interpreter to `./venv/bin/python`
2. Enable Black formatter in Settings > Tools > External Tools
3. Configure Flake8 in Settings > Tools > External Tools
4. Enable mypy in Settings > Tools > External Tools
