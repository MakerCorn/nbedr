"""
Base vector store interface for different vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models import DocumentChunk, VectorSearchResult


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store with configuration."""
        self.config = config
    
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