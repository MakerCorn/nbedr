"""
RAG Embeddings Database Core Module

This module provides the core functionality for creating and managing
embedding databases for Retrieval Augmented Generation (RAG) applications.
"""

__version__ = "1.0.0"
__author__ = "RAG Embeddings Team"
__email__ = "contact@example.com"

from .config import EmbeddingConfig
from .models import DocumentChunk, EmbeddingBatch, VectorSearchResult

__all__ = [
    "EmbeddingConfig",
    "DocumentChunk",
    "EmbeddingBatch",
    "VectorSearchResult",
]
