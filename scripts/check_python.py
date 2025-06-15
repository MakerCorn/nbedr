#!/usr/bin/env python3
"""
Check if the current Python version meets nBedR requirements.
"""

import sys
from packaging import version

REQUIRED_VERSION = "3.11.0"
CURRENT_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def check_python_version():
    """Check if current Python version meets requirements."""
    print(f"Current Python version: {CURRENT_VERSION}")
    print(f"Required Python version: {REQUIRED_VERSION}+")
    
    if version.parse(CURRENT_VERSION) >= version.parse(REQUIRED_VERSION):
        print("✅ Python version is compatible with nBedR!")
        return True
    else:
        print("❌ Python version is too old for nBedR.")
        print(f"Please upgrade to Python {REQUIRED_VERSION} or higher.")
        print("\nInstallation options:")
        print("- Using pyenv: pyenv install 3.11.0 && pyenv global 3.11.0")
        print("- Using conda: conda install python=3.11")
        print("- Download from: https://www.python.org/downloads/")
        return False

if __name__ == "__main__":
    success = check_python_version()
    sys.exit(0 if success else 1)
