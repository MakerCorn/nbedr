#!/bin/bash
"""
Development environment setup script for nBedR.
Ensures Python 3.11+ and installs dependencies.
"""

set -e

echo "🚀 Setting up nBedR development environment..."

# Check if Python 3.11+ is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Found Python 3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✅ Found Python 3.12"
elif command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
    echo "✅ Found Python 3.13"
elif command -v python3 &> /dev/null; then
    # Check if python3 is 3.11+
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
        PYTHON_CMD="python3"
        echo "✅ Found Python $PYTHON_VERSION"
    else
        echo "❌ Error: Python 3.11+ required, found Python $PYTHON_VERSION"
        echo "Please install Python 3.11 or higher:"
        echo "  - Using pyenv: pyenv install 3.11.0 && pyenv global 3.11.0"
        echo "  - Using conda: conda install python=3.11"
        echo "  - Download from: https://www.python.org/downloads/"
        exit 1
    fi
else
    echo "❌ Error: Python not found"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -e .[dev,all]

# Run initial checks
echo "🔍 Running initial checks..."
black --check . || echo "⚠️ Code formatting issues found. Run 'black .' to fix."
isort --check-only . || echo "⚠️ Import sorting issues found. Run 'isort .' to fix."
flake8 . || echo "⚠️ Linting issues found."
mypy core/ cli/ --ignore-missing-imports || echo "⚠️ Type checking issues found."

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python nbedr.py --help"
echo ""
echo "To run tests:"
echo "  pytest"
