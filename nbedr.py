#!/usr/bin/env python3
"""
Entry point script for RAG Embedding Toolkit CLI.
This script provides a way to run the CLI without module import issues.
"""
import os
import sys
from pathlib import Path

# Check Python version before proceeding
if sys.version_info < (3, 11):
    print("❌ Error: nBedR requires Python 3.11 or higher.")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("Please upgrade your Python installation.")
    sys.exit(1)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def main():
    """Main entry point."""
    try:
        from nbedr.cli.main import main as cli_main

        cli_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Please install dependencies with:")
        print("  pip install -r requirements.txt")
        print("\nOr install minimal dependencies:")
        print("  pip install openai python-dotenv pathlib")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
