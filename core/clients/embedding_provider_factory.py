"""
Embedding provider factory for creating provider instances.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from .aws_bedrock_embedding_provider import AWSBedrockEmbeddingProvider
from .azure_openai_embedding_provider import AzureOpenAIEmbeddingProvider
from .base_embedding_provider import BaseEmbeddingProvider
from .google_vertex_embedding_provider import GoogleVertexEmbeddingProvider
from .llamacpp_embedding_provider import LlamaCppEmbeddingProvider
from .lmstudio_embedding_provider import LMStudioEmbeddingProvider
from .ollama_embedding_provider import OllamaEmbeddingProvider
from .openai_embedding_provider import OpenAIEmbeddingProvider
from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEmbeddingProvider)


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances."""

    # Registry of available providers
    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {
        "openai": OpenAIEmbeddingProvider,
        "azure_openai": AzureOpenAIEmbeddingProvider,
        "aws_bedrock": AWSBedrockEmbeddingProvider,
        "google_vertex": GoogleVertexEmbeddingProvider,
        "lmstudio": LMStudioEmbeddingProvider,
        "ollama": OllamaEmbeddingProvider,
        "llamacpp": LlamaCppEmbeddingProvider,
    }

    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> BaseEmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            provider_type: Name of the provider to create
            config: Configuration dictionary for the provider

        Returns:
            BaseEmbeddingProvider instance

        Raises:
            ValueError: If provider_type is not supported or if required config is missing
        """
        if provider_type not in cls._providers:
            valid_providers = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider type: {provider_type}. Valid providers: {valid_providers}")

        provider_class = cls._providers[provider_type]
        return provider_class(config)

    @classmethod
    def from_embedding_config(cls, embedding_config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create provider from EmbeddingConfig object.

        Args:
            embedding_config: Configuration object containing all provider settings

        Returns:
            BaseEmbeddingProvider instance

        Raises:
            ValueError: If the provider type is not supported or if required config is missing
        """
        provider_name = embedding_config.embedding_provider
        if not provider_name:
            raise ValueError("embedding_provider must be specified in config")

        config = cls._build_provider_config(provider_name, embedding_config)
        return cls.create_provider(provider_name, config)

    @classmethod
    def _build_provider_config(cls, provider_name: str, embedding_config: EmbeddingConfig) -> Dict[str, Any]:
        """Build provider-specific configuration from EmbeddingConfig.

        Args:
            provider_name: Name of the provider
            embedding_config: Configuration object containing all provider settings

        Returns:
            Provider-specific configuration dictionary

        Raises:
            ValueError: If required configuration values are missing
        """
        base_config = {
            "default_model": embedding_config.embedding_model,
            "max_batch_size": embedding_config.batch_size_embeddings,
        }

        if provider_name == "openai":
            if not embedding_config.openai_api_key:
                raise ValueError("OpenAI API key is required")

            base_config.update(
                {
                    "api_key": embedding_config.openai_api_key,
                    "organization": embedding_config.openai_organization,
                    "base_url": embedding_config.openai_base_url,
                    "timeout": embedding_config.openai_timeout,
                    "max_retries": embedding_config.openai_max_retries,
                }
            )

        elif provider_name == "azure_openai":
            if not embedding_config.azure_openai_api_key:
                raise ValueError("Azure OpenAI API key is required")
            if not embedding_config.azure_openai_endpoint:
                raise ValueError("Azure OpenAI endpoint is required")

            base_config.update(
                {
                    "api_key": embedding_config.azure_openai_api_key,
                    "azure_endpoint": embedding_config.azure_openai_endpoint,
                    "api_version": embedding_config.azure_openai_api_version,
                    "deployment_name": embedding_config.azure_openai_deployment_name,
                    "deployment_mapping": embedding_config.azure_openai_deployment_mapping or {},
                    "timeout": embedding_config.azure_openai_timeout,
                    "max_retries": embedding_config.azure_openai_max_retries,
                }
            )

        elif provider_name == "aws_bedrock":
            if not embedding_config.aws_bedrock_region:
                raise ValueError("AWS Bedrock region is required")

            base_config.update(
                {
                    "region_name": embedding_config.aws_bedrock_region,
                    "aws_access_key_id": embedding_config.aws_bedrock_access_key_id,
                    "aws_secret_access_key": embedding_config.aws_bedrock_secret_access_key,
                    "aws_session_token": embedding_config.aws_bedrock_session_token,
                    "profile_name": embedding_config.aws_bedrock_profile_name,
                    "role_arn": embedding_config.aws_bedrock_role_arn,
                    "timeout": embedding_config.aws_bedrock_timeout,
                }
            )

        elif provider_name == "google_vertex":
            if not embedding_config.google_vertex_project_id:
                raise ValueError("Google Vertex project ID is required")
            if not embedding_config.google_vertex_location:
                raise ValueError("Google Vertex location is required")

            base_config.update(
                {
                    "project_id": embedding_config.google_vertex_project_id,
                    "location": embedding_config.google_vertex_location,
                    "credentials_path": embedding_config.google_vertex_credentials_path,
                    "timeout": embedding_config.google_vertex_timeout,
                }
            )

        elif provider_name == "lmstudio":
            base_config.update(
                {
                    "base_url": embedding_config.lmstudio_base_url,
                    "api_key": embedding_config.lmstudio_api_key,
                    "timeout": embedding_config.lmstudio_timeout,
                    "verify_ssl": embedding_config.lmstudio_verify_ssl,
                }
            )

        elif provider_name == "ollama":
            base_config.update(
                {
                    "base_url": embedding_config.ollama_base_url,
                    "timeout": embedding_config.ollama_timeout,
                    "verify_ssl": embedding_config.ollama_verify_ssl,
                }
            )

        elif provider_name == "llamacpp":
            base_config.update(
                {
                    "base_url": embedding_config.llamacpp_base_url,
                    "api_key": embedding_config.llamacpp_api_key,
                    "model_name": embedding_config.llamacpp_model_name,
                    "timeout": embedding_config.llamacpp_timeout,
                    "verify_ssl": embedding_config.llamacpp_verify_ssl,
                    "dimensions": embedding_config.llamacpp_dimensions,
                }
            )

        else:
            raise ValueError(f"Unknown provider type: {provider_name}")

        return base_config

    @classmethod
    def list_providers(cls) -> List[str]:
        """Get list of available provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[T]) -> None:
        """Register a new embedding provider.

        Args:
            name: Provider name
            provider_class: Provider class that extends BaseEmbeddingProvider

        Raises:
            ValueError: If the provider class doesn't extend BaseEmbeddingProvider
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise ValueError("Provider class must extend BaseEmbeddingProvider")

        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")

    @classmethod
    def get_provider_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers.

        Returns:
            Dictionary with provider information including:
            - class_name: Name of the provider class
            - provider_name: Provider identifier
            - supports_async: Whether the provider supports async operations
            - description: Provider description
            - error: Error message if provider initialization failed
        """
        info: Dict[str, Dict[str, Any]] = {}

        for name, provider_class in cls._providers.items():
            try:
                # Create a temporary instance with minimal config
                provider = provider_class({"default_model": "test"})

                info[name] = {
                    "class_name": provider_class.__name__,
                    "provider_name": provider.provider_name,
                    "supports_async": True,  # All providers are async
                    "description": provider_class.__doc__ or f"{name.title()} embedding provider",
                }
            except Exception as e:
                logger.warning(f"Failed to get info for provider {name}: {e}")
                info[name] = {
                    "class_name": provider_class.__name__,
                    "error": str(e),
                    "description": provider_class.__doc__ or f"{name.title()} embedding provider",
                }

        return info


def create_embedding_provider(provider_name: str, config: Dict[str, Any]) -> BaseEmbeddingProvider:
    """Create an embedding provider instance.

    Args:
        provider_name: Name of the provider to create
        config: Configuration dictionary for the provider

    Returns:
        BaseEmbeddingProvider instance

    Raises:
        ValueError: If the provider is not supported or if required config is missing
    """
    return EmbeddingProviderFactory.create_provider(provider_name, config)


def create_provider_from_config(embedding_config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Create provider from EmbeddingConfig object.

    Args:
        embedding_config: Configuration object containing all provider settings

    Returns:
        BaseEmbeddingProvider instance

    Raises:
        ValueError: If the provider is not supported or if required config is missing
    """
    return EmbeddingProviderFactory.from_embedding_config(embedding_config)
