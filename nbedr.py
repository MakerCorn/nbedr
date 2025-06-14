#!/usr/bin/env python3
"""
Entry point script for RAG Embedding Toolkit CLI.
This script provides a way to run the CLI without module import issues.
"""
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point."""
    try:
        from cli.main import main as cli_main
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