"""
LMStudio embedding provider implementation.
"""

import asyncio
import json
import logging
import ssl
from typing import Any, Dict, List, Optional, Union, cast

import aiohttp
from aiohttp import ClientTimeout

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult
from ..utils.embedding_utils import normalize_embedding

logger = logging.getLogger(__name__)


class LMStudioEmbeddingProvider(BaseEmbeddingProvider):
    """LMStudio embedding provider."""

    KNOWN_MODELS = {
        "default": {
            "dimensions": 768,
            "max_input_tokens": 8192,
            "cost_per_token": 0.0,
            "description": "Default LMStudio model",
        }
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the provider.

        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        super().__init__(config)
        self.model_name = config.get("model", "default")
        self.base_url = str(config.get("base_url", "http://localhost:1234"))
        self.timeout = int(config.get("timeout", 30))
        self.verify_ssl = bool(config.get("verify_ssl", True))
        self._models_cache: Optional[List[str]] = None
        self._model_info_cache: Dict[str, EmbeddingModelInfo] = {}

    def _get_ssl_context(self) -> Union[ssl.SSLContext, bool]:
        """Get SSL context based on configuration."""
        if not self.verify_ssl:
            return False
        ssl_context = ssl.create_default_context()
        return ssl_context

    async def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to LMStudio API."""
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}{endpoint}"
        json_data = data or {}

        try:
            timeout = ClientTimeout(total=float(self.timeout))
            ssl_context = self._get_ssl_context()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=json_data,
                    headers=headers,
                    timeout=timeout,
                    ssl=ssl_context,
                ) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    result: Dict[str, Any] = await response.json()
                    return result
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            return {}

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for the provider."""
        return 100

    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None, **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings using LMStudio.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default or available model)
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            raise ValueError("No texts provided for embedding")

        if model is None:
            model = await self._get_available_model()

        if batch_size is None:
            batch_size = self.get_max_batch_size()

        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Try batch processing first
                batch_embeddings = await self._generate_batch_embeddings(batch_texts, model)
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )

            except Exception as e:
                logger.warning(f"Batch processing failed, falling back to individual processing: {e}")

                # Fall back to individual processing
                for text in batch_texts:
                    try:
                        embedding = await self._generate_single_embedding(text, model)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.error(f"Failed to generate individual embedding: {e2}")
                        # Add mock embedding for failed text
                        model_info = self.KNOWN_MODELS.get(model, self.KNOWN_MODELS["default"])
                        dimensions = cast(int, model_info["dimensions"])
                        mock_embedding = self._generate_mock_embeddings([text], dimensions)[0]
                        all_embeddings.append(mock_embedding)

        dimensions = cast(int, self.KNOWN_MODELS.get(model, self.KNOWN_MODELS["default"])["dimensions"])

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=dimensions,
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={
                "provider": "lmstudio",
                "base_url": self.base_url,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def _generate_batch_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single request."""
        data = {"input": texts, "model": model}

        try:
            response = await self._make_request("/v1/embeddings", data)

            if "data" in response:
                return [[float(x) for x in item["embedding"]] for item in response["data"]]
            raise ValueError("Unexpected response format")

        except Exception as e:
            logger.error(f"Batch embedding request failed: {e}")
            raise

    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self._make_request("/api/embeddings", {"text": text, "model": model})
            if response and "embedding" in response:
                embedding_data = response["embedding"]
                if isinstance(embedding_data, list):
                    return [float(x) for x in embedding_data]
            raise ValueError("Invalid response format")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            model_info = self.KNOWN_MODELS.get(self.model_name, self.KNOWN_MODELS["default"])
            dimensions = cast(int, model_info["dimensions"])
            return self._generate_mock_embeddings([text], dimensions)[0]

    async def _get_available_model(self) -> str:
        """Get an available model from LMStudio."""
        if not self._models_cache:
            try:
                response = await self._make_request("/api/models/available")
                models = response.get("models", [])
                if isinstance(models, list):
                    self._models_cache = [str(m) for m in models if m]  # Filter out empty names
            except Exception as e:
                logger.error(f"Failed to get models: {e}")
                return str(self.model_name)
        
        if not self._models_cache or len(self._models_cache) == 0:
            return str(self.model_name)
            
        return str(self._models_cache[0])

    def _get_model_name(self, model_id: str) -> str:
        """Get canonical model name."""
        return model_id.split(":")[0]

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about a LMStudio model."""
        if model in self._model_info_cache:
            return self._model_info_cache[model]

        # Check if it's a known model
        if model in self.KNOWN_MODELS:
            model_spec = self.KNOWN_MODELS[model]
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=cast(int, model_spec["dimensions"]),
                max_input_tokens=cast(int, model_spec["max_input_tokens"]),
                cost_per_token=cast(float, model_spec.get("cost_per_token", 0.0)),
                supports_batch=True,
                provider="lmstudio",
                description=str(model_spec.get("description", f"LMStudio model: {model}"))
            )
            self._model_info_cache[model] = info
            return info

        # For unknown models, use default values
        info = EmbeddingModelInfo(
            model_name=model,
            dimensions=768,  # Default
            max_input_tokens=8192,  # Default
            cost_per_token=0.0,
            supports_batch=True,
            provider="lmstudio",
            description=f"LMStudio model: {model}"
        )
        self._model_info_cache[model] = info
        return info

    def list_models(self) -> List[str]:
        """List available models for this provider."""
        return list(self.KNOWN_MODELS.keys())

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            response = await self._make_request("/v1/models")
            return "data" in response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
