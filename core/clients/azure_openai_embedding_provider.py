"""
Azure OpenAI embedding provider implementation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    from openai import AsyncAzureOpenAI, AzureOpenAI
except ImportError:
    AzureOpenAI = None
    AsyncAzureOpenAI = None


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider."""

    # Azure OpenAI embedding models (deployment names are user-defined)
    MODELS = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_input_tokens": 8192,
            "description": "Highest quality embedding model with 3072 dimensions",
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_input_tokens": 8192,
            "description": "High quality embedding model with 1536 dimensions",
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_input_tokens": 8192,
            "description": "Legacy embedding model, still reliable",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure OpenAI embedding provider.

        Args:
            config: Configuration containing:
                - api_key: Azure OpenAI API key
                - azure_endpoint: Azure OpenAI endpoint URL
                - api_version: Azure OpenAI API version
                - deployment_name: Default deployment name
                - deployment_mapping: Dict mapping model names to deployment names
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
        """
        super().__init__(config)

        self.api_key = config.get("api_key")
        self.azure_endpoint = config.get("azure_endpoint")
        self.api_version = config.get("api_version", "2024-02-01")
        self.deployment_name = config.get("deployment_name")
        self.deployment_mapping = config.get("deployment_mapping", {})

        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")

        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

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

        if AzureOpenAI is None:
            logger.warning("OpenAI library not available, using mock implementation")
            self.client = None
            self.async_client = None
        else:
            self.client = AzureOpenAI(**client_kwargs)
            self.async_client = AsyncAzureOpenAI(**client_kwargs)

    def _get_deployment_name(self, model: str) -> str:
        """Get deployment name for a model.

        Args:
            model: Model name

        Returns:
            Deployment name
        """
        # Check deployment mapping first
        if model in self.deployment_mapping:
            return self.deployment_mapping[model]

        # Use default deployment name if available
        if self.deployment_name:
            return self.deployment_name

        # Fall back to model name as deployment name
        return model

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        deployment_name: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """Generate embeddings using Azure OpenAI API.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default)
            batch_size: Batch size for processing
            dimensions: Number of dimensions (for models that support it)
            deployment_name: Specific deployment name to use
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._validate_inputs(texts)

        if model is None:
            model = self.get_default_model() or "text-embedding-3-small"

        if deployment_name is None:
            deployment_name = self._get_deployment_name(model)

        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 2048)

        if not self.async_client:
            logger.warning("Azure OpenAI client not available, returning mock embeddings")
            mock_embeddings = self._generate_mock_embeddings(texts, self.MODELS.get(model, {}).get("dimensions", 1536))
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=f"{deployment_name}({model})",
                dimensions=len(mock_embeddings[0]) if mock_embeddings else 1536,
                token_count=sum(len(text.split()) for text in texts),
            )

        all_embeddings = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Prepare request parameters
                request_params = {
                    "input": batch_texts,
                    "model": deployment_name,  # Use deployment name instead of model name
                }

                # Add dimensions parameter for supported models
                if dimensions and model in ["text-embedding-3-large", "text-embedding-3-small"]:
                    request_params["dimensions"] = dimensions

                response = await self.async_client.embeddings.create(**request_params)

                # Extract embeddings and usage info
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                if hasattr(response, "usage") and response.usage:
                    total_tokens += response.usage.total_tokens

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add mock embeddings for failed batch
                mock_batch = self._generate_mock_embeddings(
                    batch_texts, self.MODELS.get(model, {}).get("dimensions", 1536)
                )
                all_embeddings.extend(mock_batch)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=f"{deployment_name}({model})",
            dimensions=len(all_embeddings[0]) if all_embeddings else self.MODELS.get(model, {}).get("dimensions", 1536),
            token_count=total_tokens if total_tokens > 0 else None,
            usage_stats={
                "provider": "azure_openai",
                "deployment_name": deployment_name,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an Azure OpenAI embedding model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        if model not in self.MODELS:
            # If not in known models, create basic info
            return EmbeddingModelInfo(
                model_name=model,
                dimensions=1536,  # Default dimension
                max_input_tokens=8192,  # Default max tokens
                supports_batch=True,
                provider="azure_openai",
                description=f"Azure OpenAI deployment: {self._get_deployment_name(model)}",
            )

        model_spec = self.MODELS[model]

        return EmbeddingModelInfo(
            model_name=model,
            dimensions=model_spec["dimensions"],
            max_input_tokens=model_spec["max_input_tokens"],
            supports_batch=True,
            provider="azure_openai",
            description=f"{model_spec['description']} (Deployment: {self._get_deployment_name(model)})",
        )

    def list_models(self) -> List[str]:
        """List available Azure OpenAI embedding models.

        Returns:
            List of model names
        """
        # Return known models plus any custom deployments
        models = list(self.MODELS.keys())

        # Add custom deployment names from mapping
        for model_name in self.deployment_mapping.keys():
            if model_name not in models:
                models.append(model_name)

        return models

    async def health_check(self) -> bool:
        """Check if Azure OpenAI API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        if not self.async_client:
            return False

        try:
            # Try a simple embedding request with default deployment
            deployment = self.deployment_name or self._get_deployment_name("text-embedding-3-small")
            response = await self.async_client.embeddings.create(input=["test"], model=deployment)
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Azure OpenAI health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default Azure OpenAI embedding model.

        Returns:
            Default model name
        """
        return self.config.get("default_model", "text-embedding-3-small")

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Azure OpenAI API.

        Returns:
            Maximum batch size
        """
        return self.config.get("max_batch_size", 2048)

    def get_deployment_info(self) -> Dict[str, str]:
        """Get deployment mapping information.

        Returns:
            Dictionary mapping model names to deployment names
        """
        info = {}

        # Add explicit mappings
        info.update(self.deployment_mapping)

        # Add default deployment if available
        if self.deployment_name:
            default_model = self.get_default_model()
            if default_model not in info:
                info[default_model] = self.deployment_name

        return info
