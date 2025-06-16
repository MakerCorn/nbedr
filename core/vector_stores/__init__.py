"""
Vector store implementations for different vector databases.
"""

from .base import BaseVectorStore
from .faiss_store import FAISSVectorStore

__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
]

# Optional imports with graceful fallback
try:
    from .azure_search_store import AzureAISearchVectorStore

    __all__.append("AzureAISearchVectorStore")
except ImportError:
    pass

try:
    from .elasticsearch_store import ElasticsearchVectorStore

    __all__.append("ElasticsearchVectorStore")
except ImportError:
    pass

try:
    from .pgvector_store import PGVectorStore

    __all__.append("PGVectorStore")
except ImportError:
    pass
