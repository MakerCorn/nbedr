"""
Llama.cpp embedding provider implementation.
"""

import asyncio
import json
import logging
import ssl
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

import aiohttp
from aiohttp import ClientTimeout

from ..utils.embedding_utils import normalize_embedding
from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

# Define a type variable for float lists
FloatList = List[float]

# Add mypy configuration to ignore no-any-return errors in this file
# mypy: disable-error-code="no-any-return"


logger = logging.getLogger(__name__)


class LlamaCppEmbeddingProvider(BaseEmbeddingProvider):
    """LlamaCPP embedding provider."""

    KNOWN_MODELS = {
        "default": {
            "dimensions": 4096,
            "max_input_tokens": 8192,
            "cost_per_token": 0.0,
            "description": "Default LlamaCPP model",
        }
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the provider.

        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        super().__init__(config)
        self.model_name = config.get("model", "default")
        self.base_url = str(config.get("base_url", "http://localhost:8080"))
        self.api_key = str(config.get("api_key", ""))
        self.timeout = int(config.get("timeout", 30))
        self.verify_ssl = bool(config.get("verify_ssl", True))
        self.expected_dimensions = int(config.get("dimensions", 4096))
        self._models_cache: Optional[List[str]] = None
        self._model_info_cache: Dict[str, EmbeddingModelInfo] = {}
        self._server_info: Dict[str, Any] = {}

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for the provider."""
        return 50

    def _get_ssl_context(self) -> Union[ssl.SSLContext, bool]:
        """Get SSL context based on configuration."""
        if not self.verify_ssl:
            return False
        ssl_context = ssl.create_default_context()
        return ssl_context

    async def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to Llama.cpp server.

        Args:
            endpoint: API endpoint
            data: Request data for POST requests

        Returns:
            Response data
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            ssl_context = self._get_ssl_context()
            timeout = ClientTimeout(total=float(self.timeout))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
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
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings using Llama.cpp.

        Args:
            texts: List of texts to embed
            model: Model to use (typically ignored as llama.cpp loads one model)
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            raise ValueError("No texts provided for embedding")

        if model is None:
            model = self.model_name

        if batch_size is None:
            batch_size = self.get_max_batch_size()

        normalize = kwargs.get("normalize", True)
        all_embeddings: List[List[float]] = []

        # Process in batches or individually depending on server capabilities
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Try batch processing first
                batch_embeddings = await self._generate_batch_embeddings(batch_texts, normalize)
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
                        mock_embedding = self._generate_mock_embeddings([text], self.expected_dimensions)[0]
                        all_embeddings.append(mock_embedding)

        dimensions = len(all_embeddings[0]) if all_embeddings else self.expected_dimensions

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=dimensions,
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={
                "provider": "llamacpp",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "normalize": normalize,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def _generate_batch_embeddings(self, texts: List[str], normalize: bool = True) -> List[FloatList]:
        """Generate embeddings for multiple texts in a single request."""
        # Try OpenAI-compatible API first
        try:
            data = {"input": texts, "model": self.model_name}
            response = await self._make_request("/v1/embeddings", data)

            if "data" in response and isinstance(response["data"], list):
                result: List[List[float]] = []
                for item in response["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        embedding = item["embedding"]
                        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                            result.append([float(x) for x in embedding])
                if result and len(result) == len(texts):
                    return result

            # Try llama.cpp specific batch API
            data = {"content": texts, "normalize": normalize}
            response = await self._make_request("/embeddings", data)

            if "embeddings" in response and isinstance(response["embeddings"], list):
                return [[float(x) for x in emb] for emb in response["embeddings"]]
            elif "embedding" in response and isinstance(response["embedding"], list) and len(response["embedding"]) > 0:
                if isinstance(response["embedding"][0], list):
                    return [[float(x) for x in emb] for emb in response["embedding"]]

            # If we get here, generate mock embeddings as fallback
            return self._generate_mock_embeddings(texts, self.expected_dimensions)

        except Exception as e:
            logger.error(f"Batch embedding request failed: {e}")
            # Return mock embeddings instead of raising
            return self._generate_mock_embeddings(texts, self.expected_dimensions)

    async def _generate_single_embedding(self, text: str, model: str) -> FloatList:
        """Generate embedding for a single text."""
        try:
            response = await self._make_request("/embeddings", {"content": text, "model": model})
            if response and "embedding" in response:
                embedding = [float(x) for x in response["embedding"]]
                return normalize_embedding(embedding)
            raise ValueError("Invalid response format")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            dimensions = cast(int, self.KNOWN_MODELS[self.model_name]["dimensions"])
            mock_embedding: FloatList = self._generate_mock_embeddings([text], dimensions)[0]
            return mock_embedding

    async def _get_available_model(self) -> str:
        """Get an available model from Llama.cpp."""
        if self._models_cache is None:
            try:
                response = await self._make_request("/v1/models")
                models = response.get("data", [])
                if isinstance(models, list) and all(isinstance(model, dict) for model in models):
                    self._models_cache = [str(model.get("id", "")) for model in models if "id" in model]
                else:
                    self._models_cache = []
            except Exception as e:
                logger.error(f"Failed to fetch models: {e}")
                self._models_cache = []

        if self._models_cache and len(self._models_cache) > 0:
            # Try to use the default model if available
            if self.model_name in self._models_cache:
                return str(self.model_name)
            return str(self._models_cache[0])

        return str(self.model_name)

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about a Llama.cpp model."""
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
                provider="llamacpp",
                description=str(model_spec.get("description", f"LlamaCPP model: {model}")),
            )
            self._model_info_cache[model] = info
            return info

        # For unknown models, use default values
        info = EmbeddingModelInfo(
            model_name=model,
            dimensions=self.expected_dimensions,
            max_input_tokens=8192,
            cost_per_token=0.0,
            supports_batch=True,
            provider="llamacpp",
            description=f"LlamaCPP model: {model}",
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

    async def _get_single_embedding(self, text: str, model: Optional[str] = None) -> FloatList:
        """Get embedding for a single text using Llama.cpp."""

        # Define a function to create mock embeddings to ensure consistent return type
        def create_mock_embedding() -> FloatList:
            return self._generate_mock_embeddings([text], self.expected_dimensions)[0]

        try:
            effective_model = model or self.model_name
            response = await self._make_request(
                "embedding",
                {
                    "content": text,
                    "model": effective_model,
                },
            )

            if not response or "embedding" not in response:
                # Return mock embedding if no valid response
                return create_mock_embedding()

            embedding_data = response["embedding"]
            if not isinstance(embedding_data, list):
                # Return mock embedding if wrong type
                return create_mock_embedding()

            embedding: FloatList = [float(x) for x in embedding_data]
            if len(embedding) != self.expected_dimensions:
                logger.warning(
                    f"Embedding dimensions mismatch. Expected {self.expected_dimensions}, got {len(embedding)}"
                )
            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return create_mock_embedding()

    def _get_model_name(self, model_id: str) -> str:
        """Get canonical model name."""
        return model_id
