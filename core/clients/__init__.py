"""
Embedding client functionality with multiple provider support.
"""

from .aws_bedrock_embedding_provider import AWSBedrockEmbeddingProvider
from .azure_openai_embedding_provider import AzureOpenAIEmbeddingProvider

# New embedding provider system
from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult
from .embedding_provider_factory import EmbeddingProviderFactory, create_embedding_provider, create_provider_from_config
from .google_vertex_embedding_provider import GoogleVertexEmbeddingProvider
from .llamacpp_embedding_provider import LlamaCppEmbeddingProvider
from .lmstudio_embedding_provider import LMStudioEmbeddingProvider
from .ollama_embedding_provider import OllamaEmbeddingProvider

# Legacy OpenAI client imports for backward compatibility
from .openai_client import (
    EmbeddingClient,
    build_langchain_embeddings,
    build_openai_client,
    create_embedding_client,
    is_azure,
)

# Individual provider imports
from .openai_embedding_provider import OpenAIEmbeddingProvider

__all__ = [
    # Legacy exports
    "build_openai_client",
    "build_langchain_embeddings",
    "is_azure",
    "EmbeddingClient",
    "create_embedding_client",
    # New provider system
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingModelInfo",
    "EmbeddingProviderFactory",
    "create_embedding_provider",
    "create_provider_from_config",
    # Individual providers
    "OpenAIEmbeddingProvider",
    "AzureOpenAIEmbeddingProvider",
    "AWSBedrockEmbeddingProvider",
    "GoogleVertexEmbeddingProvider",
    "LMStudioEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "LlamaCppEmbeddingProvider",
]
