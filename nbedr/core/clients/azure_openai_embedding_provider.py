"""
Azure OpenAI embedding provider implementation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import AsyncAzureOpenAI, AzureOpenAI
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.embedding import Embedding

    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not available, using mock implementation")
    AZURE_OPENAI_AVAILABLE = False

    # Create mock types for when OpenAI is not available
    class _MockAzureOpenAI:
        pass

    class _MockAsyncAzureOpenAI:
        pass

    class _MockCreateEmbeddingResponse:
        pass

    class _MockEmbedding:
        pass

    AzureOpenAI = _MockAzureOpenAI  # type: ignore
    AsyncAzureOpenAI = _MockAsyncAzureOpenAI  # type: ignore
    CreateEmbeddingResponse = _MockCreateEmbeddingResponse  # type: ignore
    Embedding = _MockEmbedding  # type: ignore


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""

    KNOWN_MODELS = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_input_tokens": 8192,
            "cost_per_token": 0.00013 / 1000,  # $0.00013 per 1K tokens
            "description": "Highest quality embedding model with 3072 dimensions",
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_input_tokens": 8192,
            "cost_per_token": 0.00002 / 1000,  # $0.00002 per 1K tokens
            "description": "High quality embedding model with 1536 dimensions",
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_input_tokens": 8192,
            "cost_per_token": 0.0001 / 1000,  # $0.0001 per 1K tokens
            "description": "Legacy embedding model, still reliable",
        },
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Azure OpenAI embedding provider."""
        super().__init__(config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")

        self.azure_endpoint = config.get("azure_endpoint")
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        self.api_version = str(config.get("api_version", "2024-02-01"))
        self.deployment_name = config.get("deployment_name")
        self.deployment_mapping = cast(Dict[str, str], config.get("deployment_mapping", {}))
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)

        # Initialize clients
        client_kwargs = {
            "api_key": self.api_key,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        self.client: Optional[AzureOpenAI] = None
        self.async_client: Optional[AsyncAzureOpenAI] = None

        if AZURE_OPENAI_AVAILABLE:
            try:
                self.client = AzureOpenAI(**client_kwargs)
                self.async_client = AsyncAzureOpenAI(**client_kwargs)
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
                self.async_client = None

    def _get_deployment_name(self, model: Optional[str] = None) -> str:
        """Get the Azure deployment name for a model."""
        if model and model in self.config.get("deployments", {}):
            return str(self.config["deployments"][model])

        # If no specific model requested or not found, use default
        default_deployment = self.config.get("default_deployment")
        if default_deployment:
            return str(default_deployment)

        raise ValueError("No deployment configured and no model specified")

    async def _initialize_client(self) -> None:
        """Initialize the Azure OpenAI client."""
        try:
            if not self.async_client:
                endpoint = self.config.get("endpoint")
                api_key = self.config.get("api_key")
                api_version = self.config.get("api_version", "2024-02-15-preview")

                if not endpoint or not api_key:
                    raise ValueError("Azure OpenAI endpoint and API key are required")

                self.async_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version,
                )

        except Exception as e:
            self._record_error("client_initialization_error")
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        deployment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings using Azure OpenAI API."""
        if not texts:
            raise ValueError("No texts provided for embedding")

        if model is None:
            model = self.get_default_model()

        effective_deployment = deployment_name or self._get_deployment_name(model)

        if batch_size is None:
            batch_size = self.get_max_batch_size()

        if not self.async_client:
            logger.warning("Azure OpenAI client not available, returning mock embeddings")
            mock_dims = cast(int, self.KNOWN_MODELS.get(model, {}).get("dimensions", 1536))
            mock_embeddings = self._generate_mock_embeddings(texts, mock_dims)
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=f"{effective_deployment}({model})",
                dimensions=mock_dims,
                token_count=sum(len(text.split()) for text in texts),
                usage_stats={"provider": "azure_openai", "mock": True},
            )

        all_embeddings: List[List[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Prepare request parameters
                request_params: Dict[str, Any] = {
                    "input": batch_texts,
                    "model": effective_deployment,
                }

                if dimensions and model in ["text-embedding-3-large", "text-embedding-3-small"]:
                    request_params["dimensions"] = dimensions

                response = await self.async_client.embeddings.create(**request_params)

                # Extract embeddings and usage info
                batch_embeddings = [list(data.embedding) for data in response.data]
                all_embeddings.extend(batch_embeddings)

                if hasattr(response, "usage") and response.usage:
                    tokens_used = response.usage.total_tokens
                    total_tokens += tokens_used
                    self._record_response(time.time(), tokens_used)

                logger.debug(
                    f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )

            except Exception as e:
                error_type = "rate_limit" if "rate_limit" in str(e).lower() else "server_error"
                self._record_error(error_type)
                logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}")

                # Add mock embeddings for failed batch
                mock_dims = cast(int, self.KNOWN_MODELS.get(model, {}).get("dimensions", 1536))
                mock_batch = self._generate_mock_embeddings(batch_texts, mock_dims)
                all_embeddings.extend(mock_batch)

        dimensions_used = (
            len(all_embeddings[0])
            if all_embeddings
            else cast(int, self.KNOWN_MODELS.get(model, {}).get("dimensions", 1536))
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=f"{effective_deployment}({model})",
            dimensions=dimensions_used,
            token_count=total_tokens if total_tokens > 0 else None,
            usage_stats={
                "provider": "azure_openai",
                "deployment_name": effective_deployment,
                "base_url": self.azure_endpoint,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def get_embeddings(
        self, texts: List[str], model: Optional[str] = None, dimensions: Optional[int] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            dimensions: Optional dimensions override for supported models
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()
        batch_size = self.get_max_batch_size()
        all_embeddings: List[List[float]] = []

        # Ensure client is initialized
        await self._initialize_client()
        if not self.async_client:
            raise RuntimeError("Failed to initialize Azure OpenAI client")

        # Get effective deployment name
        effective_deployment = self._get_deployment_name(model)

        # Apply rate limiting
        await self._apply_rate_limiting(len(texts))

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Prepare request parameters
                request_params: Dict[str, Any] = {
                    "input": batch_texts,
                    "model": effective_deployment,
                }

                if dimensions and model in ["text-embedding-3-large", "text-embedding-3-small"]:
                    request_params["dimensions"] = dimensions

                response = await self.async_client.embeddings.create(**request_params)

                # Extract embeddings and usage info
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Record usage if available
                if hasattr(response, "usage") and response.usage:
                    tokens_used = response.usage.total_tokens
                    self._record_response(time.time() - start_time, tokens_used)

            except Exception as e:
                self._record_error(str(e))
                logger.error(f"Failed to get embeddings from Azure OpenAI {effective_deployment}: {e}")
                raise

        return all_embeddings

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an Azure OpenAI embedding model."""
        if model not in self.KNOWN_MODELS:
            # If not in known models, create basic info
            return EmbeddingModelInfo(
                model_name=model,
                dimensions=1536,  # Default dimension
                max_input_tokens=8192,  # Default max tokens
                cost_per_token=0.0,
                supports_batch=True,
                provider="azure_openai",
                description=f"Azure OpenAI deployment: {self._get_deployment_name(model)}",
            )

        model_spec = self.KNOWN_MODELS[model]

        return EmbeddingModelInfo(
            model_name=model,
            dimensions=cast(int, model_spec["dimensions"]),
            max_input_tokens=cast(int, model_spec["max_input_tokens"]),
            cost_per_token=cast(float, model_spec.get("cost_per_token", 0.0)),
            supports_batch=True,
            provider="azure_openai",
            description=f"{model_spec['description']} (Deployment: {self._get_deployment_name(model)})",
        )

    def list_models(self) -> List[str]:
        """List available Azure OpenAI embedding models."""
        # Return known models plus any custom deployments
        models = list(self.KNOWN_MODELS.keys())

        # Add custom deployment names from mapping if they map to unknown models
        for model_name in self.deployment_mapping:
            if model_name not in models:
                models.append(model_name)

        return models

    async def health_check(self) -> bool:
        """Check if Azure OpenAI API is accessible."""
        if not self.async_client:
            return False

        try:
            # Try a simple embedding request with default deployment
            deployment = self.deployment_name or self._get_deployment_name("text-embedding-3-small")
            response = await self.async_client.embeddings.create(input=["test"], model=deployment)
            return bool(response.data)
        except Exception as e:
            logger.error(f"Azure OpenAI health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default Azure OpenAI embedding model."""
        return str(self.config.get("default_model", "text-embedding-3-small"))

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Azure OpenAI API."""
        return int(self.config.get("max_batch_size", 2048))

    def get_deployment_info(self) -> Dict[str, str]:
        """Get deployment mapping information."""
        info: Dict[str, str] = {}

        # Add explicit mappings
        info.update(self.deployment_mapping)

        # Add default deployment if available
        if self.deployment_name:
            default_model = self.get_default_model()
            if default_model not in info:
                info[default_model] = self.deployment_name

        return info
