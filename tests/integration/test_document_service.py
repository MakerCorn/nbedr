"""
Integration tests for document service functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.clients import BaseEmbeddingProvider, EmbeddingResult, create_provider_from_config
from core.config import EmbeddingConfig, get_config
from core.models import DocumentChunk, VectorDatabaseConfig, VectorDatabaseType
from core.services.document_service import DocumentService
from core.sources.local import LocalDocumentSource


class TestDocumentServiceIntegration:
    """Integration tests for document service with real components."""

    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary directory with test documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)

            # Create test documents
            (docs_dir / "test1.txt").write_text("This is the first test document with some content.")
            (docs_dir / "test2.txt").write_text("This is the second test document with different content.")
            (docs_dir / "test3.md").write_text("# Markdown Document\n\nThis is a markdown test document.")

            yield docs_dir

    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            provider="openai",
            api_key="test-key-12345",
            model="text-embedding-3-small",
            dimensions=1536,
            batch_size=10,
            max_workers=2,
            rate_limit_enabled=False,
        )

    @pytest.fixture
    def vector_db_config(self):
        """Create vector database configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VectorDatabaseConfig(
                type=VectorDatabaseType.FAISS,
                connection_params={"index_path": str(Path(temp_dir) / "test_index")},
            )

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider that returns consistent embeddings."""
        provider = Mock(spec=BaseEmbeddingProvider)
        provider.provider_name = "openai"
        provider.model_name = "text-embedding-3-small"
        provider.dimensions = 1536
        provider.max_batch_size = 2048

        async def mock_generate_embeddings(texts):
            # Generate deterministic embeddings based on text content
            embeddings = []
            for i, text in enumerate(texts):
                # Create a simple embedding based on text hash and index
                base_value = hash(text) % 100 / 100.0
                embedding = [base_value + (j * 0.001) for j in range(1536)]
                embeddings.append(embedding)

            return EmbeddingResult(
                embeddings=embeddings,
                model="text-embedding-3-small",
                usage={"prompt_tokens": len(texts) * 10, "total_tokens": len(texts) * 10},
            )

        provider.generate_embeddings = mock_generate_embeddings
        return provider

    @pytest.fixture
    def document_service(self, embedding_config, vector_db_config, mock_embedding_provider):
        """Create document service with mocked embedding provider."""
        with patch("core.services.document_service.create_provider_from_config", return_value=mock_embedding_provider):
            service = DocumentService(embedding_config, vector_db_config)
            return service

    @pytest.mark.asyncio
    async def test_process_local_documents(self, document_service, temp_docs_dir):
        """Test processing documents from local directory."""
        # Create local document source
        source = LocalDocumentSource(str(temp_docs_dir))

        # Process documents
        result = await document_service.process_documents_from_source(source)

        # Verify results
        assert result is not None
        assert len(result.chunks) > 0
        assert len(result.embeddings) == len(result.chunks)

        # Check that all chunks have embeddings
        for chunk in result.chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1536

    @pytest.mark.asyncio
    async def test_chunking_strategies(self, document_service, temp_docs_dir):
        """Test different chunking strategies."""
        source = LocalDocumentSource(str(temp_docs_dir))

        # Test with different chunk sizes
        for chunk_size in [256, 512, 1024]:
            document_service.config.chunk_size = chunk_size
            result = await document_service.process_documents_from_source(source)

            assert result is not None
            assert len(result.chunks) > 0

            # Verify chunk sizes are reasonable
            for chunk in result.chunks:
                assert len(chunk.content) <= chunk_size * 1.5  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_batch_processing(self, document_service, temp_docs_dir):
        """Test batch processing of documents."""
        # Create more test documents
        for i in range(10):
            (temp_docs_dir / f"batch_test_{i}.txt").write_text(f"This is batch test document number {i}.")

        source = LocalDocumentSource(str(temp_docs_dir))

        # Set small batch size to test batching
        document_service.config.batch_size = 3

        result = await document_service.process_documents_from_source(source)

        assert result is not None
        assert len(result.chunks) > 10  # Should have many chunks
        assert len(result.embeddings) == len(result.chunks)

    @pytest.mark.asyncio
    async def test_error_handling(self, embedding_config, vector_db_config):
        """Test error handling in document processing."""
        # Create service with failing embedding provider
        failing_provider = Mock(spec=BaseEmbeddingProvider)
        failing_provider.generate_embeddings.side_effect = Exception("API Error")

        with patch("core.services.document_service.create_provider_from_config", return_value=failing_provider):
            service = DocumentService(embedding_config, vector_db_config)

            # Create source with non-existent directory
            source = LocalDocumentSource("/non/existent/path")

            # Should handle errors gracefully
            with pytest.raises(Exception):
                await service.process_documents_from_source(source)

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, document_service, temp_docs_dir):
        """Test that document metadata is preserved through processing."""
        # Create document with specific metadata
        test_file = temp_docs_dir / "metadata_test.txt"
        test_file.write_text("This document should preserve metadata.")

        source = LocalDocumentSource(str(temp_docs_dir))
        result = await document_service.process_documents_from_source(source)

        # Find chunks from our test file
        test_chunks = [chunk for chunk in result.chunks if "metadata_test.txt" in chunk.source]
        assert len(test_chunks) > 0

        # Verify metadata is preserved
        for chunk in test_chunks:
            assert chunk.source.endswith("metadata_test.txt")
            assert chunk.metadata is not None
            assert isinstance(chunk.metadata, dict)

    @pytest.mark.asyncio
    async def test_document_filtering(self, document_service, temp_docs_dir):
        """Test document filtering by file type."""
        # Create files of different types
        (temp_docs_dir / "test.txt").write_text("Text file content")
        (temp_docs_dir / "test.md").write_text("# Markdown content")
        (temp_docs_dir / "test.py").write_text("# Python code")
        (temp_docs_dir / "test.log").write_text("Log file content")

        source = LocalDocumentSource(str(temp_docs_dir))
        result = await document_service.process_documents_from_source(source)

        # Check that appropriate files were processed
        processed_extensions = set()
        for chunk in result.chunks:
            file_path = Path(chunk.source)
            processed_extensions.add(file_path.suffix)

        # Should process common text formats
        expected_extensions = {".txt", ".md"}
        assert expected_extensions.issubset(processed_extensions)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, document_service, temp_docs_dir):
        """Test concurrent document processing."""
        # Create multiple documents
        for i in range(20):
            (temp_docs_dir / f"concurrent_test_{i}.txt").write_text(
                f"This is concurrent test document {i} with unique content for testing."
            )

        source = LocalDocumentSource(str(temp_docs_dir))

        # Enable concurrent processing
        document_service.config.max_workers = 4

        result = await document_service.process_documents_from_source(source)

        assert result is not None
        assert len(result.chunks) > 20
        assert len(result.embeddings) == len(result.chunks)

        # Verify all embeddings are valid
        for embedding in result.embeddings:
            assert len(embedding) == 1536
            assert all(isinstance(x, (int, float)) for x in embedding)


class TestDocumentServiceConfiguration:
    """Test document service configuration and setup."""

    def test_service_initialization(self):
        """Test service initialization with different configurations."""
        config = EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        vector_config = VectorDatabaseConfig(
            type=VectorDatabaseType.FAISS,
            connection_params={"index_path": "./test_index"},
        )

        with patch("core.services.document_service.create_provider_from_config"):
            service = DocumentService(config, vector_config)

            assert service.config == config
            assert service.vector_db_config == vector_config

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        # Test with invalid provider
        with pytest.raises((ValueError, TypeError)):
            config = EmbeddingConfig(
                provider="invalid_provider",
                api_key="test-key",
            )

            vector_config = VectorDatabaseConfig(
                type=VectorDatabaseType.FAISS,
                connection_params={"index_path": "./test_index"},
            )

            DocumentService(config, vector_config)


class TestRealProviderIntegration:
    """Integration tests with real provider implementations (mocked API calls)."""

    @pytest.mark.asyncio
    async def test_openai_provider_integration(self):
        """Test integration with OpenAI provider (mocked)."""
        config = EmbeddingConfig(
            provider="openai",
            api_key="test-key-12345",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        # Mock the OpenAI API call
        with patch("core.clients.openai_embedding_provider.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_response.usage = Mock(prompt_tokens=10, total_tokens=10)
            mock_client.embeddings.create.return_value = mock_response

            # Create provider and test
            provider = create_provider_from_config(config)
            result = await provider.generate_embeddings(["test text"])

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 1536

    @pytest.mark.asyncio
    async def test_provider_fallback_behavior(self):
        """Test provider fallback when API is unavailable."""
        config = EmbeddingConfig(
            provider="openai",
            api_key="test-key-12345",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        # Mock API failure
        with patch("core.clients.openai_embedding_provider.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.side_effect = Exception("API Error")

            # Create provider and test fallback
            provider = create_provider_from_config(config)
            result = await provider.generate_embeddings(["test text"])

            # Should still return embeddings (mock fallback)
            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 1536
