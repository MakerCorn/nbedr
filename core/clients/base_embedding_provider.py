"""
Base interface for embedding providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.rate_limiter import RateLimitConfig, RateLimiter, create_rate_limiter_from_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: List[List[float]]
    model: str
    dimensions: int
    token_count: Optional[int] = None
    usage_stats: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model."""

    model_name: str
    dimensions: int
    max_input_tokens: int
    cost_per_token: Optional[float] = None
    supports_batch: bool = True
    provider: str = ""
    description: str = ""


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider.

        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace("EmbeddingProvider", "").lower()

        # Initialize rate limiter
        self.rate_limiter = self._create_rate_limiter(config)

    @abstractmethod
    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None, **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            model: Model to use (provider-specific)
            batch_size: Number of texts to process in each batch
            **kwargs: Additional provider-specific arguments

        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        pass

    @abstractmethod
    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about a specific model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    def get_default_model(self) -> str:
        """Get the default model for this provider.

        Returns:
            Default model name
        """
        return self.config.get("default_model", "")

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for this provider.

        Returns:
            Maximum batch size
        """
        return self.config.get("max_batch_size", 100)

    def supports_async(self) -> bool:
        """Check if provider supports async operations.

        Returns:
            True if async is supported
        """
        return True

    def _validate_inputs(self, texts: List[str]) -> None:
        """Validate input texts.

        Args:
            texts: List of texts to validate

        Raises:
            ValueError: If inputs are invalid
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")

        if any(len(text.strip()) == 0 for text in texts):
            logger.warning("Some texts are empty or contain only whitespace")

    def _generate_mock_embeddings(self, texts: List[str], dimensions: int = 1536) -> List[List[float]]:
        """Generate mock embeddings for testing purposes.

        Args:
            texts: List of texts to embed
            dimensions: Number of dimensions for embeddings

        Returns:
            List of mock embedding vectors
        """
        import random

        embeddings = []
        for text in texts:
            # Generate deterministic mock embedding based on text hash
            random.seed(hash(text) % (2**32))
            embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
            embeddings.append(embedding)

        return embeddings

    def _create_rate_limiter(self, config: Dict[str, Any]) -> Optional[RateLimiter]:
        """Create rate limiter from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            RateLimiter instance or None if disabled
        """
        # Check for rate limiting configuration
        rate_limit_enabled = config.get("rate_limit_enabled", False)

        if not rate_limit_enabled:
            return None

        # Get rate limiting parameters
        rate_config = {
            "enabled": True,
            "strategy": config.get("rate_limit_strategy", "sliding_window"),
            "requests_per_minute": config.get("rate_limit_requests_per_minute"),
            "requests_per_hour": config.get("rate_limit_requests_per_hour"),
            "tokens_per_minute": config.get("rate_limit_tokens_per_minute"),
            "tokens_per_hour": config.get("rate_limit_tokens_per_hour"),
            "max_burst_requests": config.get("rate_limit_max_burst"),
            "burst_window_seconds": config.get("rate_limit_burst_window", 60.0),
            "target_response_time": config.get("rate_limit_target_response_time", 2.0),
            "max_response_time": config.get("rate_limit_max_response_time", 10.0),
            "max_retries": config.get("rate_limit_max_retries", 3),
            "base_retry_delay": config.get("rate_limit_base_delay", 1.0),
        }

        # Remove None values
        rate_config = {k: v for k, v in rate_config.items() if v is not None}

        try:
            return create_rate_limiter_from_config(**rate_config)
        except Exception as e:
            logger.warning(f"Failed to create rate limiter: {e}")
            return None

    async def _apply_rate_limiting(self, estimated_tokens: Optional[int] = None) -> float:
        """Apply rate limiting before making a request.

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            Delay time applied in seconds
        """
        if not self.rate_limiter:
            return 0.0

        start_time = time.time()
        delay = self.rate_limiter.acquire(estimated_tokens)

        if delay > 0:
            logger.debug(f"Rate limited: waited {delay:.2f}s for {self.provider_name}")

        return delay

    def _record_response(self, response_time: float, actual_tokens: Optional[int] = None):
        """Record response information for rate limiting.

        Args:
            response_time: Response time in seconds
            actual_tokens: Actual tokens used
        """
        if self.rate_limiter:
            self.rate_limiter.record_response(response_time, actual_tokens)

    def _record_error(self, error_type: str, retry_after: Optional[float] = None):
        """Record an error for rate limiting adjustments.

        Args:
            error_type: Type of error
            retry_after: Retry-After header value if available
        """
        if self.rate_limiter:
            self.rate_limiter.record_error(error_type, retry_after)

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics.

        Returns:
            Dictionary with rate limiting stats
        """
        if not self.rate_limiter:
            return {"enabled": False}

        return self.rate_limiter.get_statistics()

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.provider_name.title()}EmbeddingProvider"

    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return f"{self.__class__.__name__}(provider_name='{self.provider_name}')"
