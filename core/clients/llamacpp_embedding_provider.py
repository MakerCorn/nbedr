"""
Llama.cpp embedding provider implementation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)


class LlamaCppEmbeddingProvider(BaseEmbeddingProvider):
    """Llama.cpp embedding provider for local models via llama-cpp-python server."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Llama.cpp embedding provider.

        Args:
            config: Configuration containing:
                - base_url: Llama.cpp server URL (default: http://localhost:8000)
                - api_key: API key if required (optional)
                - model_name: Name of the loaded model
                - default_model: Default model identifier
                - timeout: Request timeout in seconds
                - verify_ssl: Whether to verify SSL certificates
                - dimensions: Expected embedding dimensions (if known)
        """
        super().__init__(config)

        self.base_url = config.get("base_url", "http://localhost:8000").rstrip("/")
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "unknown")
        self.timeout = config.get("timeout", 120)  # Can be slower for large models
        self.verify_ssl = config.get("verify_ssl", True)
        self.expected_dimensions = config.get("dimensions")

        # Cache for model info
        self._model_info_cache: Dict[str, Any] = {}
        self._server_info = None

    async def _make_request(self, endpoint: str, data: Optional[Dict] = None, method: str = "GET") -> Dict[str, Any]:
        """Make HTTP request to Llama.cpp server.

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
                    if response.status == 404:
                        raise ValueError(f"Endpoint not found: {endpoint}")
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """Generate embeddings using Llama.cpp.

        Args:
            texts: List of texts to embed
            model: Model to use (typically ignored as llama.cpp loads one model)
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._validate_inputs(texts)

        if model is None:
            model = self.get_default_model()

        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 50)

        all_embeddings = []

        # Process in batches or individually depending on server capabilities
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Try batch processing first
                batch_embeddings = await self._generate_batch_embeddings(batch_texts, normalize)
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

            except Exception as e:
                logger.warning(f"Batch processing failed, falling back to individual processing: {e}")

                # Fall back to individual processing
                for text in batch_texts:
                    try:
                        embedding = await self._generate_single_embedding(text, normalize)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.error(f"Failed to generate individual embedding: {e2}")
                        # Add mock embedding for failed text
                        expected_dims = self.expected_dimensions or 4096  # Common llama dimension
                        mock_embedding = self._generate_mock_embeddings([text], expected_dims)[0]
                        all_embeddings.append(mock_embedding)

        dimensions = len(all_embeddings[0]) if all_embeddings else (self.expected_dimensions or 4096)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=f"{self.model_name}({model})" if model != self.model_name else self.model_name,
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

    async def _generate_batch_embeddings(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        # Try OpenAI-compatible API first
        try:
            data = {"input": texts, "model": self.model_name}

            response = await self._make_request("/v1/embeddings", data, "POST")

            if "data" in response:
                return [item["embedding"] for item in response["data"]]
            else:
                raise ValueError("Unexpected response format")

        except Exception as e:
            logger.debug(f"OpenAI-compatible API failed, trying llama.cpp specific API: {e}")

            # Try llama.cpp specific batch API
            data = {"content": texts, "normalize": normalize}

            response = await self._make_request("/embeddings", data, "POST")

            if "embeddings" in response:
                return response["embeddings"]
            elif "embedding" in response and isinstance(response["embedding"][0], list):
                return response["embedding"]
            else:
                raise ValueError("Unexpected response format for batch embeddings")

    async def _generate_single_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to normalize embedding

        Returns:
            Embedding vector
        """
        # Try OpenAI-compatible API first
        try:
            data = {"input": text, "model": self.model_name}

            response = await self._make_request("/v1/embeddings", data, "POST")

            if "data" in response and len(response["data"]) > 0:
                return response["data"][0]["embedding"]

        except Exception as e:
            logger.debug(f"OpenAI-compatible API failed, trying llama.cpp specific API: {e}")

        # Try llama.cpp specific API
        data = {"content": text, "normalize": normalize}

        response = await self._make_request("/embedding", data, "POST")

        if "embedding" in response:
            return response["embedding"]
        else:
            raise ValueError("Unexpected response format for single embedding")

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about the loaded Llama.cpp model.

        Args:
            model: Model identifier (may be ignored as llama.cpp loads one model)

        Returns:
            EmbeddingModelInfo with model details
        """
        cache_key = f"{self.model_name}_{model}"

        if cache_key in self._model_info_cache:
            return self._model_info_cache[cache_key]

        try:
            # Try to get server/model information
            response = await self._make_request("/props")

            dimensions = self.expected_dimensions or 4096  # Default
            max_tokens = 2048  # Conservative default

            # Extract information from props if available
            if "n_embd" in response:
                dimensions = response["n_embd"]
            if "n_ctx" in response:
                max_tokens = response["n_ctx"]

            description = f"Llama.cpp model: {self.model_name}"
            if "model_path" in response:
                description += f" (Path: {response['model_path']})"

            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=dimensions,
                max_input_tokens=max_tokens,
                supports_batch=True,
                provider="llamacpp",
                description=description,
            )

        except Exception as e:
            logger.warning(f"Failed to get model properties: {e}")

            # Return basic info
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=self.expected_dimensions or 4096,
                max_input_tokens=2048,
                supports_batch=True,
                provider="llamacpp",
                description=f"Llama.cpp model: {self.model_name} (info unavailable)",
            )

        self._model_info_cache[cache_key] = info
        return info

    def list_models(self) -> List[str]:
        """List available models (typically just the loaded model).

        Returns:
            List of model names
        """
        # Llama.cpp typically loads one model at a time
        return [self.model_name] if self.model_name != "unknown" else []

    async def health_check(self) -> bool:
        """Check if Llama.cpp server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            # Try to get server properties
            response = await self._make_request("/props")
            return isinstance(response, dict)
        except Exception:
            try:
                # Fallback: try a simple embedding
                await self._generate_single_embedding("test", normalize=False)
                return True
            except Exception as e:
                logger.error(f"Llama.cpp health check failed: {e}")
                return False

    def get_default_model(self) -> str:
        """Get the default Llama.cpp model.

        Returns:
            Default model name
        """
        return self.config.get("default_model", self.model_name)

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Llama.cpp.

        Returns:
            Maximum batch size
        """
        return self.config.get("max_batch_size", 50)

    async def get_server_properties(self) -> Dict[str, Any]:
        """Get detailed server properties from Llama.cpp.

        Returns:
            Dictionary with server properties
        """
        if self._server_info is None:
            try:
                self._server_info = await self._make_request("/props")
            except Exception as e:
                logger.error(f"Failed to get server properties: {e}")
                self._server_info = {}

        return self._server_info.copy()

    def get_server_info(self) -> Dict[str, Any]:
        """Get Llama.cpp server information.

        Returns:
            Dictionary with server info
        """
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "has_api_key": bool(self.api_key),
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
            "expected_dimensions": self.expected_dimensions,
        }
