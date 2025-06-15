"""
Tests for vector store modules.
"""

import pytest


class TestVectorStoresModule:
    """Test cases for vector stores module imports and basic functionality."""

    def test_vector_stores_init_import(self):
        """Test that vector_stores module can be imported with optional dependencies."""
        try:
            import core.vector_stores
            assert hasattr(core.vector_stores, '__name__')
        except ImportError as e:
            # Some vector stores have optional dependencies, skip if missing
            if "azure.search" in str(e) or "elasticsearch" in str(e) or "psycopg" in str(e):
                pytest.skip(f"Optional dependency missing: {e}")
            else:
                pytest.fail(f"Failed to import core.vector_stores: {e}")

    def test_base_import(self):
        """Test that base module can be imported."""
        try:
            from core.vector_stores.base import BaseVectorStore
            assert BaseVectorStore is not None
            # Verify it's a class
            assert isinstance(BaseVectorStore, type)
        except ImportError as e:
            if "azure.search" in str(e) or "elasticsearch" in str(e) or "psycopg" in str(e):
                pytest.skip(f"Optional dependency missing: {e}")
            else:
                pytest.fail(f"Failed to import BaseVectorStore: {e}")

    def test_faiss_store_import(self):
        """Test that faiss_store module can be imported."""
        try:
            from core.vector_stores.faiss_store import FAISSVectorStore
            assert FAISSVectorStore is not None
            # Verify it's a class
            assert isinstance(FAISSVectorStore, type)
        except ImportError as e:
            if "faiss" in str(e).lower() or "azure.search" in str(e):
                pytest.skip(f"Optional dependency missing: {e}")
            else:
                pytest.fail(f"Failed to import FAISSVectorStore: {e}")
    
    def test_azure_search_store_import(self):
        """Test that azure_search_store module can be imported."""
        try:
            from core.vector_stores.azure_search_store import AzureAISearchVectorStore
            assert AzureAISearchVectorStore is not None
            assert isinstance(AzureAISearchVectorStore, type)
        except ImportError as e:
            if "azure.search" in str(e):
                pytest.skip(f"Optional Azure Search dependency missing: {e}")
            else:
                pytest.fail(f"Failed to import AzureAISearchVectorStore: {e}")
    
    def test_pgvector_store_import(self):
        """Test that pgvector_store module can be imported."""
        try:
            from core.vector_stores.pgvector_store import PGVectorStore
            assert PGVectorStore is not None
            assert isinstance(PGVectorStore, type)
        except ImportError as e:
            if "psycopg" in str(e) or "azure.search" in str(e):
                pytest.skip(f"Optional PostgreSQL dependency missing: {e}")
            else:
                pytest.fail(f"Failed to import PGVectorStore: {e}")
            
    def test_all_vector_stores_inherit_from_base(self):
        """Test that all vector store classes inherit from BaseVectorStore."""
        try:
            from core.vector_stores.base import BaseVectorStore
            from core.vector_stores.faiss_store import FAISSVectorStore
            
            # Test inheritance for FAISS (which should always be available)
            assert issubclass(FAISSVectorStore, BaseVectorStore)
            
            # Try to test other stores if their dependencies are available
            try:
                from core.vector_stores.azure_search_store import AzureAISearchVectorStore
                assert issubclass(AzureAISearchVectorStore, BaseVectorStore)
            except ImportError:
                pass  # Azure Search dependency not available
                
            try:
                from core.vector_stores.pgvector_store import PGVectorStore
                assert issubclass(PGVectorStore, BaseVectorStore)
            except ImportError:
                pass  # PostgreSQL dependency not available
                
        except ImportError as e:
            if "azure.search" in str(e) or "psycopg" in str(e):
                pytest.skip(f"Optional dependencies missing: {e}")
            else:
                pytest.fail(f"Failed to import vector store classes: {e}")