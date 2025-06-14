"""
Input source providers for RAG embedding database application.

This module provides an extensible system for handling different input sources
including local files, S3 buckets, and SharePoint sites.
"""

from .base import BaseInputSource, InputSourceConfig, SourceDocument, SourceValidationError
from .local import LocalInputSource
from .s3 import S3InputSource
from .sharepoint import SharePointInputSource
from .factory import InputSourceFactory

__all__ = [
    "BaseInputSource",
    "InputSourceConfig", 
    "SourceDocument",
    "SourceValidationError",
    "LocalInputSource",
    "S3InputSource", 
    "SharePointInputSource",
    "InputSourceFactory",
]