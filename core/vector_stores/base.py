"""
Base vector store interface for different vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from ..models import DocumentChunk, VectorSearchResult
from ..utils.rate_limiter import RateLimiter, create_rate_limiter_from_config

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store with configuration."""
        self.config = config
        
        # Initialize rate limiter for vector store operations
        self.rate_limiter = self._create_rate_limiter(config)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create indices, collections, etc.)."""
        pass
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            List of vector IDs for the added documents
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of search results with similarity scores
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """
        Delete documents by their vector IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the vector store connection."""
        pass
    
    def _create_rate_limiter(self, config: Dict[str, Any]) -> Optional[RateLimiter]:
        """Create rate limiter for vector store operations.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RateLimiter instance or None if disabled
        """
        # Check for vector store rate limiting configuration
        rate_limit_enabled = config.get('vector_store_rate_limit_enabled', False)
        
        if not rate_limit_enabled:
            return None
        
        # Get rate limiting parameters with defaults for vector stores
        rate_config = {
            'enabled': True,
            'strategy': config.get('vector_store_rate_limit_strategy', 'sliding_window'),
            'requests_per_minute': config.get('vector_store_rate_limit_requests_per_minute', 300),
            'requests_per_hour': config.get('vector_store_rate_limit_requests_per_hour'),
            'max_burst_requests': config.get('vector_store_rate_limit_max_burst', 50),
            'burst_window_seconds': config.get('vector_store_rate_limit_burst_window', 60.0),
            'target_response_time': config.get('vector_store_rate_limit_target_response_time', 1.0),
            'max_response_time': config.get('vector_store_rate_limit_max_response_time', 5.0),
            'max_retries': config.get('vector_store_rate_limit_max_retries', 3),
            'base_retry_delay': config.get('vector_store_rate_limit_base_delay', 0.5)
        }
        
        # Remove None values
        rate_config = {k: v for k, v in rate_config.items() if v is not None}
        
        try:
            return create_rate_limiter_from_config(**rate_config)
        except Exception as e:
            logger.warning(f"Failed to create vector store rate limiter: {e}")
            return None
    
    async def _apply_rate_limiting(self, operation_type: str = "operation") -> float:
        """Apply rate limiting before vector store operations.
        
        Args:
            operation_type: Type of operation for logging
            
        Returns:
            Delay time applied in seconds
        """
        if not self.rate_limiter:
            return 0.0
        
        delay = self.rate_limiter.acquire()
        
        if delay > 0:
            logger.debug(f"Vector store rate limited: waited {delay:.2f}s for {operation_type}")
        
        return delay
    
    def _record_operation_response(self, response_time: float, operation_type: str = "operation"):
        """Record response information for rate limiting.
        
        Args:
            response_time: Response time in seconds
            operation_type: Type of operation
        """
        if self.rate_limiter:
            self.rate_limiter.record_response(response_time)
    
    def _record_operation_error(self, error_type: str, retry_after: Optional[float] = None):
        """Record an error for rate limiting adjustments.
        
        Args:
            error_type: Type of error
            retry_after: Retry-After header value if available
        """
        if self.rate_limiter:
            self.rate_limiter.record_error(error_type, retry_after)
    
    def get_vector_store_rate_limit_stats(self) -> Dict[str, Any]:
        """Get vector store rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting stats
        """
        if not self.rate_limiter:
            return {'enabled': False}
        
        stats = self.rate_limiter.get_statistics()
        stats['component'] = 'vector_store'
        return stats