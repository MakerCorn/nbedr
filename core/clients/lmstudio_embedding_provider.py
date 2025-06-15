"""
LMStudio embedding provider implementation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)


class LMStudioEmbeddingProvider(BaseEmbeddingProvider):
    """LMStudio embedding provider for local models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LMStudio embedding provider.

        Args:
            config: Configuration containing:
                - base_url: LMStudio server URL (default: http://localhost:1234)
                - api_key: API key if required (optional)
                - default_model: Default model to use
                - timeout: Request timeout in seconds
                - verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(config)

        self.base_url = config.get("base_url", "http://localhost:1234").rstrip("/")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 60)
        self.verify_ssl = config.get("verify_ssl", True)

        # Cache for discovered models
        self._models_cache = None
        self._model_info_cache: Dict[str, Any] = {}

    async def _make_request(self, endpoint: str, data: Optional[Dict] = None, method: str = "GET") -> Dict[str, Any]:
        """Make HTTP request to LMStudio server.

        Args:
            endpoint: API endpoint
            data: Request data for POST requests
            method: HTTP method

        Returns:
            Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if method.upper() == "POST":
                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()

    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None, **kwargs
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
        self._validate_inputs(texts)

        if model is None:
            model = await self._get_available_model()

        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 100)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Try batch processing first
                batch_embeddings = await self._generate_batch_embeddings(batch_texts, model)
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

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
                        mock_embedding = self._generate_mock_embeddings([text], 1536)[0]
                        all_embeddings.append(mock_embedding)

        dimensions = len(all_embeddings[0]) if all_embeddings else 1536

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
        """Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed
            model: Model to use

        Returns:
            List of embedding vectors
        """
        data = {"input": texts, "model": model}

        try:
            response = await self._make_request("/v1/embeddings", data, "POST")

            if "data" in response:
                return [item["embedding"] for item in response["data"]]
            else:
                raise ValueError("Unexpected response format")

        except Exception as e:
            logger.error(f"Batch embedding request failed: {e}")
            raise

    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector
        """
        data = {"input": text, "model": model}

        response = await self._make_request("/v1/embeddings", data, "POST")

        if "data" in response and len(response["data"]) > 0:
            return response["data"][0]["embedding"]
        else:
            raise ValueError("Unexpected response format")

    async def _get_available_model(self) -> str:
        """Get an available model from LMStudio.

        Returns:
            Model name
        """
        if self._models_cache is None:
            try:
                response = await self._make_request("/v1/models")
                if "data" in response:
                    self._models_cache = [model["id"] for model in response["data"]]
                else:
                    self._models_cache = []
            except Exception as e:
                logger.error(f"Failed to fetch models from LMStudio: {e}")
                self._models_cache = []

        if self._models_cache:
            default = self.get_default_model()
            if default and default in self._models_cache:
                return default
            return self._models_cache[0]

        # If no models available, return a default name
        return self.get_default_model() or "embedding-model"

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an LMStudio model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        if model in self._model_info_cache:
            return self._model_info_cache[model]

        try:
            # Try to get model info from LMStudio
            response = await self._make_request("/v1/models")

            if "data" in response:
                for model_data in response["data"]:
                    if model_data["id"] == model:
                        # Extract what info we can
                        info = EmbeddingModelInfo(
                            model_name=model,
                            dimensions=1536,  # Default, will be updated after first embedding
                            max_input_tokens=8192,  # Conservative default
                            supports_batch=True,
                            provider="lmstudio",
                            description=f"LMStudio model: {model}",
                        )

                        self._model_info_cache[model] = info
                        return info

            # If model not found in list, create basic info
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=1536,  # Default
                max_input_tokens=8192,  # Conservative default
                supports_batch=True,
                provider="lmstudio",
                description=f"LMStudio model: {model}",
            )

            self._model_info_cache[model] = info
            return info

        except Exception as e:
            logger.error(f"Failed to get model info for {model}: {e}")

            # Return basic info
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=1536,
                max_input_tokens=8192,
                supports_batch=True,
                provider="lmstudio",
                description=f"LMStudio model: {model} (info unavailable)",
            )

            return info

    def list_models(self) -> List[str]:
        """List available LMStudio models.

        Returns:
            List of model names
        """
        if self._models_cache is not None:
            return self._models_cache.copy()

        # Return empty list if not cached yet
        return []

    async def health_check(self) -> bool:
        """Check if LMStudio server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = await self._make_request("/v1/models")
            return "data" in response
        except Exception as e:
            logger.error(f"LMStudio health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default LMStudio embedding model.

        Returns:
            Default model name
        """
        return self.config.get("default_model", "")

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for LMStudio.

        Returns:
            Maximum batch size
        """
        return self.config.get("max_batch_size", 100)

    def get_server_info(self) -> Dict[str, Any]:
        """Get LMStudio server information.

        Returns:
            Dictionary with server info
        """
        return {
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
            "models_cached": self._models_cache is not None,
            "cached_models_count": len(self._models_cache) if self._models_cache else 0,
        }
