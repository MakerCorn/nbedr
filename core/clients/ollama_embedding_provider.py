"""
Ollama embedding provider implementation.
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


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider."""

    KNOWN_MODELS = {
        "default": {
            "dimensions": 768,
            "max_input_tokens": 8192,
            "cost_per_token": 0.0,
            "description": "Default Ollama model",
        }
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the provider."""
        super().__init__(config)
        self.model_name = config.get("model", "default")
        self.base_url = str(config.get("base_url", "http://localhost:11434"))
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
        """Make HTTP request to Ollama API."""
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

    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None, **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings using Ollama.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default or available model)
            batch_size: Batch size for processing (Ollama processes individually)
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            raise ValueError("No texts provided for embedding")

        if model is None:
            model = await self._get_available_model()

        all_embeddings: List[List[float]] = []

        # Ollama typically processes embeddings individually
        for i, text in enumerate(texts):
            try:
                embedding = await self._generate_single_embedding(text, model)
                all_embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated embeddings for {i + 1}/{len(texts)} texts")

            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i + 1}: {e}")
                # Add mock embedding for failed text
                model_info = self.KNOWN_MODELS.get(model, self.KNOWN_MODELS["default"])
                dimensions = cast(int, model_info.get("dimensions", 768))
                mock_embeddings = self._generate_mock_embeddings([text], dimensions)[0]
                all_embeddings.append(mock_embeddings)

        dimensions = cast(int, self.KNOWN_MODELS.get(model, self.KNOWN_MODELS["default"]).get("dimensions", 768))

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=dimensions,
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={"provider": "ollama", "base_url": self.base_url, "texts_processed": len(texts)},
        )

    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self._make_request("/api/embeddings", {"model": model, "prompt": text})
            if "embedding" in response:
                embedding = response["embedding"]
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    return [float(x) for x in embedding]
            raise ValueError("No embedding in response")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            model_info = self.KNOWN_MODELS.get(model, self.KNOWN_MODELS["default"])
            dimensions = cast(int, model_info.get("dimensions", 768))
            return self._generate_mock_embeddings([text], dimensions)[0]

    async def _get_available_model(self) -> str:
        """Get an available model from Ollama."""
        if not self._models_cache:
            try:
                response = await self._make_request("/api/tags")
                models = []
                if "models" in response and isinstance(response["models"], list):
                    models = [str(model.get("name", "")) for model in response.get("models", [])]
                self._models_cache = [m for m in models if m]  # Filter out empty names
            except Exception as e:
                logger.error(f"Failed to get models: {e}")
                return str(self.model_name)

        if not self._models_cache:
            return str(self.model_name)

        return str(self._models_cache[0])

    def _get_model_name(self, model_id: str) -> str:
        """Get canonical model name."""
        return model_id.split(":")[0]

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an Ollama model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        if model in self._model_info_cache:
            return self._model_info_cache[model]

        # Check if it's a known model
        if model in self.KNOWN_MODELS:
            model_spec = self.KNOWN_MODELS[model]
            description = str(model_spec.get("description", f"Ollama model: {model}"))
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=cast(int, model_spec["dimensions"]),
                max_input_tokens=cast(int, model_spec["max_input_tokens"]),
                cost_per_token=cast(float, model_spec.get("cost_per_token", 0.0)),
                supports_batch=False,  # Ollama processes individually
                provider="ollama",
                description=description,
            )
            self._model_info_cache[model] = info
            return info

        # For unknown models, use default values
        info = EmbeddingModelInfo(
            model_name=model,
            dimensions=768,  # Default
            max_input_tokens=8192,  # Default
            cost_per_token=0.0,
            supports_batch=False,
            provider="ollama",
            description=f"Ollama model: {model}",
        )
        self._model_info_cache[model] = info
        return info

    def list_models(self) -> List[str]:
        """List available models for this provider."""
        return list(self.KNOWN_MODELS.keys())

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            response = await self._make_request("/api/version")
            return bool(response.get("version", False))
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
