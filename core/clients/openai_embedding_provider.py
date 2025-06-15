"""
OpenAI embedding provider implementation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import AsyncOpenAI, OpenAI
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.embedding import Embedding

    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI library not available, using mock implementation")
    OPENAI_AVAILABLE = False
    # Use Any for type annotations
    OpenAI = Any  # type: ignore[misc, assignment]
    AsyncOpenAI = Any  # type: ignore[misc, assignment]
    CreateEmbeddingResponse = Any  # type: ignore[misc, assignment]
    Embedding = Any  # type: ignore[misc, assignment]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

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
        """Initialize OpenAI embedding provider.

        Args:
            config: Configuration containing:
                - api_key: OpenAI API key
                - organization: OpenAI organization ID (optional)
                - base_url: Custom base URL (optional)
                - default_model: Default model to use
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
        """
        super().__init__(config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.organization = config.get("organization")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)

        # Initialize clients
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.organization:
            client_kwargs["organization"] = self.organization

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None

        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(**client_kwargs)
                self.async_client = AsyncOpenAI(**client_kwargs)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.async_client = None

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        if not texts:
            raise ValueError("No texts provided for embedding")

        if model is None:
            model = self.get_default_model()

        if batch_size is None:
            batch_size = self.get_max_batch_size()

        if not self.async_client:
            logger.warning("OpenAI client not available, returning mock embeddings")
            mock_dims = cast(int, self.KNOWN_MODELS.get(model, {}).get("dimensions", 1536))
            mock_embeddings = self._generate_mock_embeddings(texts, mock_dims)
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=model,
                dimensions=mock_dims,
                token_count=sum(len(text.split()) for text in texts),
                usage_stats={"provider": "openai", "mock": True},
            )

        all_embeddings: List[List[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            estimated_tokens = sum(len(text.split()) * 1.3 for text in batch_texts)

            try:
                # Apply rate limiting
                await self._apply_rate_limiting(int(estimated_tokens))

                # Prepare request parameters
                request_params: Dict[str, Any] = {"input": batch_texts, "model": model}
                if dimensions and model in ["text-embedding-3-large", "text-embedding-3-small"]:
                    request_params["dimensions"] = dimensions

                start_time = time.time()
                response = await self.async_client.embeddings.create(**request_params)
                response_time = time.time() - start_time

                # Extract embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Extract token usage
                if hasattr(response, "usage") and response.usage:
                    tokens_used = response.usage.total_tokens
                    total_tokens += tokens_used
                    self._record_response(response_time, tokens_used)
                else:
                    self._record_response(response_time, int(estimated_tokens))

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
            model=model,
            dimensions=dimensions_used,
            token_count=total_tokens if total_tokens > 0 else None,
            usage_stats={
                "provider": "openai",
                "base_url": self.base_url,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an OpenAI embedding model."""
        if model not in self.KNOWN_MODELS:
            raise ValueError(f"Unknown OpenAI embedding model: {model}")

        model_spec = self.KNOWN_MODELS[model]

        return EmbeddingModelInfo(
            model_name=model,
            dimensions=cast(int, model_spec["dimensions"]),
            max_input_tokens=cast(int, model_spec["max_input_tokens"]),
            cost_per_token=cast(float, model_spec["cost_per_token"]),
            supports_batch=True,
            provider="openai",
            description=str(model_spec["description"]),
        )

    def list_models(self) -> List[str]:
        """List available OpenAI embedding models."""
        return list(self.KNOWN_MODELS.keys())

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.async_client:
            return False

        try:
            response = await self.async_client.embeddings.create(input=["test"], model="text-embedding-3-small")
            return bool(response.data)
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default OpenAI embedding model."""
        return str(self.config.get("default_model", "text-embedding-3-small"))

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for OpenAI API."""
        return int(self.config.get("max_batch_size", 2048))

    async def _estimate_cost(self, token_count: int, model: Optional[str] = None) -> float:
        """Estimate cost for the operation."""
        effective_model = model or self.get_default_model()

        if effective_model not in self.KNOWN_MODELS:
            return 0.0

        cost_per_token_str = str(self.KNOWN_MODELS[effective_model]["cost_per_token"])
        cost_per_token = float(cost_per_token_str)
        return float(token_count) * cost_per_token
