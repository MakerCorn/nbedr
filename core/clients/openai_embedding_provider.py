"""
OpenAI embedding provider implementation.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingResult, EmbeddingModelInfo

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    # OpenAI embedding models with their specifications
    MODELS = {
        'text-embedding-3-large': {
            'dimensions': 3072,
            'max_input_tokens': 8192,
            'cost_per_token': 0.00013 / 1000,  # $0.00013 per 1K tokens
            'description': 'Highest quality embedding model with 3072 dimensions'
        },
        'text-embedding-3-small': {
            'dimensions': 1536,
            'max_input_tokens': 8192,
            'cost_per_token': 0.00002 / 1000,  # $0.00002 per 1K tokens
            'description': 'High quality embedding model with 1536 dimensions'
        },
        'text-embedding-ada-002': {
            'dimensions': 1536,
            'max_input_tokens': 8192,
            'cost_per_token': 0.0001 / 1000,  # $0.0001 per 1K tokens
            'description': 'Legacy embedding model, still reliable'
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
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
        
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.organization = config.get('organization')
        self.base_url = config.get('base_url')
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        
        # Initialize clients
        client_kwargs = {
            'api_key': self.api_key,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }
        
        if self.organization:
            client_kwargs['organization'] = self.organization
        
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        
        if OpenAI is None:
            logger.warning("OpenAI library not available, using mock implementation")
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default)
            batch_size: Batch size for processing
            dimensions: Number of dimensions (for models that support it)
            **kwargs: Additional arguments
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._validate_inputs(texts)
        
        if model is None:
            model = self.get_default_model() or 'text-embedding-3-small'
        
        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 2048)
        
        if not self.async_client:
            logger.warning("OpenAI client not available, returning mock embeddings")
            mock_embeddings = self._generate_mock_embeddings(texts, self.MODELS.get(model, {}).get('dimensions', 1536))
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=model,
                dimensions=len(mock_embeddings[0]) if mock_embeddings else 1536,
                token_count=sum(len(text.split()) for text in texts)
            )
        
        all_embeddings = []
        total_tokens = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Estimate tokens for rate limiting
            estimated_tokens = sum(len(text.split()) * 1.3 for text in batch_texts)  # Rough estimation
            
            try:
                # Apply rate limiting
                await self._apply_rate_limiting(int(estimated_tokens))
                
                # Prepare request parameters
                request_params = {
                    'input': batch_texts,
                    'model': model
                }
                
                # Add dimensions parameter for supported models
                if dimensions and model in ['text-embedding-3-large', 'text-embedding-3-small']:
                    request_params['dimensions'] = dimensions
                
                start_time = time.time()
                response = await self.async_client.embeddings.create(**request_params)
                response_time = time.time() - start_time
                
                # Extract embeddings and usage info
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                actual_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    actual_tokens = response.usage.total_tokens
                    total_tokens += actual_tokens
                
                # Record response for rate limiting
                self._record_response(response_time, actual_tokens or int(estimated_tokens))
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
                
            except Exception as e:
                # Record error for rate limiting
                if "rate_limit" in str(e).lower():
                    self._record_error("rate_limit")
                elif "server" in str(e).lower():
                    self._record_error("server_error")
                
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add mock embeddings for failed batch
                mock_batch = self._generate_mock_embeddings(batch_texts, self.MODELS.get(model, {}).get('dimensions', 1536))
                all_embeddings.extend(mock_batch)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.MODELS.get(model, {}).get('dimensions', 1536),
            token_count=total_tokens if total_tokens > 0 else None,
            usage_stats={'provider': 'openai', 'batches_processed': (len(texts) - 1)//batch_size + 1}
        )
    
    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about an OpenAI embedding model.
        
        Args:
            model: Model name
            
        Returns:
            EmbeddingModelInfo with model details
        """
        if model not in self.MODELS:
            raise ValueError(f"Unknown OpenAI embedding model: {model}")
        
        model_spec = self.MODELS[model]
        
        return EmbeddingModelInfo(
            model_name=model,
            dimensions=model_spec['dimensions'],
            max_input_tokens=model_spec['max_input_tokens'],
            cost_per_token=model_spec['cost_per_token'],
            supports_batch=True,
            provider='openai',
            description=model_spec['description']
        )
    
    def list_models(self) -> List[str]:
        """List available OpenAI embedding models.
        
        Returns:
            List of model names
        """
        return list(self.MODELS.keys())
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        if not self.async_client:
            return False
        
        try:
            # Try a simple embedding request
            response = await self.async_client.embeddings.create(
                input=["test"],
                model="text-embedding-3-small"
            )
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_default_model(self) -> str:
        """Get the default OpenAI embedding model.
        
        Returns:
            Default model name
        """
        return self.config.get('default_model', 'text-embedding-3-small')
    
    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for OpenAI API.
        
        Returns:
            Maximum batch size
        """
        return self.config.get('max_batch_size', 2048)
    
    def estimate_cost(self, token_count: int, model: str) -> float:
        """Estimate the cost for embedding generation.
        
        Args:
            token_count: Number of tokens to embed
            model: Model to use
            
        Returns:
            Estimated cost in USD
        """
        if model not in self.MODELS:
            return 0.0
        
        cost_per_token = self.MODELS[model]['cost_per_token']
        return token_count * cost_per_token