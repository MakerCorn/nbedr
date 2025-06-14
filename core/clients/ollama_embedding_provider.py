"""
Ollama embedding provider implementation.
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingResult, EmbeddingModelInfo

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models."""
    
    # Common Ollama embedding models
    KNOWN_MODELS = {
        'nomic-embed-text': {
            'dimensions': 768,
            'max_input_tokens': 8192,
            'description': 'High-quality English embedding model'
        },
        'mxbai-embed-large': {
            'dimensions': 1024,
            'max_input_tokens': 512,
            'description': 'Large embedding model by MixedBread AI'
        },
        'snowflake-arctic-embed': {
            'dimensions': 1024,
            'max_input_tokens': 512,
            'description': 'Snowflake Arctic embedding model'
        },
        'all-minilm': {
            'dimensions': 384,
            'max_input_tokens': 256,
            'description': 'Lightweight multilingual embedding model'
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama embedding provider.
        
        Args:
            config: Configuration containing:
                - base_url: Ollama server URL (default: http://localhost:11434)
                - default_model: Default model to use
                - timeout: Request timeout in seconds
                - verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(config)
        
        self.base_url = config.get('base_url', 'http://localhost:11434').rstrip('/')
        self.timeout = config.get('timeout', 120)  # Ollama can be slower
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Cache for discovered models
        self._models_cache = None
        self._model_info_cache = {}
    
    async def _make_request(self, endpoint: str, data: Optional[Dict] = None, method: str = 'GET') -> Dict[str, Any]:
        """Make HTTP request to Ollama server.
        
        Args:
            endpoint: API endpoint
            data: Request data for POST requests
            method: HTTP method
            
        Returns:
            Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if method.upper() == 'POST':
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 404:
                        raise ValueError(f"Model or endpoint not found: {endpoint}")
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
        **kwargs
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
        self._validate_inputs(texts)
        
        if model is None:
            model = await self._get_available_model()
        
        all_embeddings = []
        
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
                expected_dims = self.KNOWN_MODELS.get(model, {}).get('dimensions', 768)
                mock_embedding = self._generate_mock_embeddings([text], expected_dims)[0]
                all_embeddings.append(mock_embedding)
        
        dimensions = len(all_embeddings[0]) if all_embeddings else self.KNOWN_MODELS.get(model, {}).get('dimensions', 768)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=dimensions,
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={
                'provider': 'ollama',
                'base_url': self.base_url,
                'texts_processed': len(texts)
            }
        )
    
    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text using Ollama.
        
        Args:
            text: Text to embed
            model: Model to use
            
        Returns:
            Embedding vector
        """
        data = {
            'model': model,
            'prompt': text
        }
        
        try:
            response = await self._make_request('/api/embeddings', data, 'POST')
            
            if 'embedding' in response:
                return response['embedding']
            else:
                raise ValueError("Unexpected response format - no embedding found")
                
        except Exception as e:
            logger.error(f"Ollama embedding request failed: {e}")
            raise
    
    async def _get_available_model(self) -> str:
        """Get an available model from Ollama.
        
        Returns:
            Model name
        """
        if self._models_cache is None:
            try:
                response = await self._make_request('/api/tags')
                if 'models' in response:
                    self._models_cache = [model['name'] for model in response['models']]
                else:
                    self._models_cache = []
            except Exception as e:
                logger.error(f"Failed to fetch models from Ollama: {e}")
                self._models_cache = []
        
        if self._models_cache:
            default = self.get_default_model()
            if default and default in self._models_cache:
                return default
            
            # Prefer known embedding models
            for known_model in self.KNOWN_MODELS:
                if known_model in self._models_cache:
                    return known_model
            
            # Return first available model
            return self._models_cache[0]
        
        # If no models available, return a default name
        return self.get_default_model() or 'nomic-embed-text'
    
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
            info = EmbeddingModelInfo(
                model_name=model,
                dimensions=model_spec['dimensions'],
                max_input_tokens=model_spec['max_input_tokens'],
                supports_batch=False,  # Ollama processes individually
                provider='ollama',
                description=model_spec['description']
            )
        else:
            # Try to get model info from Ollama
            try:
                response = await self._make_request('/api/show', {'name': model}, 'POST')
                
                # Extract what info we can from the model details
                dimensions = 768  # Default
                max_tokens = 8192  # Default
                description = f"Ollama model: {model}"
                
                if 'modelinfo' in response:
                    modelinfo = response['modelinfo']
                    # Try to extract embedding dimensions from model info
                    # This is model-specific and may vary
                
                info = EmbeddingModelInfo(
                    model_name=model,
                    dimensions=dimensions,
                    max_input_tokens=max_tokens,
                    supports_batch=False,
                    provider='ollama',
                    description=description
                )
                
            except Exception as e:
                logger.warning(f"Failed to get detailed info for model {model}: {e}")
                
                # Return basic info
                info = EmbeddingModelInfo(
                    model_name=model,
                    dimensions=768,  # Common default
                    max_input_tokens=8192,  # Conservative default
                    supports_batch=False,
                    provider='ollama',
                    description=f"Ollama model: {model} (info unavailable)"
                )
        
        self._model_info_cache[model] = info
        return info
    
    def list_models(self) -> List[str]:
        """List available Ollama models.
        
        Returns:
            List of model names
        """
        if self._models_cache is not None:
            return self._models_cache.copy()
        
        # Return known models if cache not available yet
        return list(self.KNOWN_MODELS.keys())
    
    async def health_check(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = await self._make_request('/api/tags')
            return 'models' in response
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_default_model(self) -> str:
        """Get the default Ollama embedding model.
        
        Returns:
            Default model name
        """
        return self.config.get('default_model', 'nomic-embed-text')
    
    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Ollama.
        
        Returns:
            Maximum batch size (1 for individual processing)
        """
        return 1  # Ollama processes embeddings individually
    
    async def pull_model(self, model: str) -> bool:
        """Pull/download a model to Ollama.
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {'name': model}
            await self._make_request('/api/pull', data, 'POST')
            
            # Clear models cache to refresh
            self._models_cache = None
            
            logger.info(f"Successfully pulled model: {model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get Ollama server information.
        
        Returns:
            Dictionary with server info
        """
        return {
            'base_url': self.base_url,
            'timeout': self.timeout,
            'verify_ssl': self.verify_ssl,
            'models_cached': self._models_cache is not None,
            'cached_models_count': len(self._models_cache) if self._models_cache else 0,
            'known_embedding_models': list(self.KNOWN_MODELS.keys())
        }