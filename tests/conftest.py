"""
Test configuration and fixtures for RAG embedding database tests.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest

# Import core modules with new API structure
from core.clients import (
    BaseEmbeddingProvider,
    EmbeddingProviderFactory,
    EmbeddingResult,
)
from core.config import EmbeddingConfig
from core.models import (
    DocumentChunk,
    EmbeddingBatch,
    VectorDatabaseConfig,
    VectorDatabaseType,
)
from core.services.document_service import DocumentService
from core.utils.rate_limiter import RateLimitConfig, RateLimiter, RateLimitStrategy


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create test PDF content (mock)
        (test_dir / "test.pdf").write_bytes(b"Mock PDF content")

        # Create test text file
        (test_dir / "test.txt").write_text("This is a test document.\nIt has multiple lines.\nFor testing purposes.")

        # Create test JSON file
        test_json = {
            "text": "This is JSON content for testing",
            "metadata": {"type": "test"},
        }
        (test_dir / "test.json").write_text(json.dumps(test_json))

        # Create test API documentation JSON
        api_docs = [
            {
                "user_name": "test_user",
                "api_name": "test_api",
                "api_call": "GET /api/test",
                "api_version": "v1",
                "api_arguments": {"param1": "value1"},
                "functionality": "Test API endpoint",
            }
        ]
        (test_dir / "api_docs.json").write_text(json.dumps(api_docs))

        yield test_dir


@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables for testing."""
    # Remove common environment variables that might interfere with tests
    env_vars_to_remove = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENABLED",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "VECTOR_DATABASE_TYPE",
        "FAISS_INDEX_PATH",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "CHROMA_HOST",
        "CHROMA_PORT",
    ]

    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_embedding_config():
    """Create a mock embedding configuration for testing."""
    return EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        api_key="test-key-12345",
        batch_size=10,
        max_workers=2,
        rate_limit_enabled=False,
    )


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider for testing."""
    provider = Mock(spec=BaseEmbeddingProvider)
    provider.provider_name = "mock"
    provider.model_name = "mock-model"
    provider.dimensions = 1536
    provider.max_batch_size = 100

    # Mock the generate_embeddings method
    async def mock_generate_embeddings(texts: List[str]) -> EmbeddingResult:
        embeddings = [[0.1] * 1536 for _ in texts]
        return EmbeddingResult(
            embeddings=embeddings,
            model="mock-model",
            dimensions=1536,
            usage_stats={"prompt_tokens": len(texts) * 10, "total_tokens": len(texts) * 10},
        )

    provider.generate_embeddings = mock_generate_embeddings
    return provider


@pytest.fixture
def mock_vector_db_config():
    """Create a mock vector database configuration."""
    return VectorDatabaseConfig(
        type=VectorDatabaseType.FAISS,
        connection_params={"index_path": "./test_embeddings"},
    )


@pytest.fixture
def sample_document_chunks():
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            id="chunk_1",
            content="This is the first test chunk with some content.",
            source="test_doc_1.txt",
            metadata={"page": 1, "section": "intro"},
            embedding=None,
        ),
        DocumentChunk(
            id="chunk_2",
            content="This is the second test chunk with different content.",
            source="test_doc_1.txt",
            metadata={"page": 1, "section": "body"},
            embedding=None,
        ),
        DocumentChunk(
            id="chunk_3",
            content="This is the third test chunk from another document.",
            source="test_doc_2.txt",
            metadata={"page": 1, "section": "conclusion"},
            embedding=None,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536 dimensions
        [0.4, 0.5, 0.6] + [0.0] * 1533,
        [0.7, 0.8, 0.9] + [0.0] * 1533,
    ]


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter for testing."""
    limiter = Mock(spec=RateLimiter)
    limiter.can_proceed.return_value = True
    limiter.record_request.return_value = None
    limiter.get_stats.return_value = {
        "total_requests": 0,
        "rate_limit_hits": 0,
        "average_response_time": 0.0,
    }
    return limiter


@pytest.fixture
def mock_document_service(mock_embedding_provider, mock_vector_db_config):
    """Create a mock document service for testing."""
    service = Mock(spec=DocumentService)
    service.embedding_provider = mock_embedding_provider
    service.vector_db_config = mock_vector_db_config
    return service


@pytest.fixture
def rate_limit_config():
    """Create a rate limit configuration for testing."""
    return RateLimitConfig(
        enabled=True,
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        requests_per_minute=60,
        tokens_per_minute=50000,
        max_burst=10,
    )


@pytest.fixture
def embedding_batch(sample_document_chunks, sample_embeddings):
    """Create an embedding batch for testing."""
    # Add embeddings to chunks
    for chunk, embedding in zip(sample_document_chunks, sample_embeddings):
        chunk.embedding = embedding

    return EmbeddingBatch(
        id="test-batch-123",
        chunks=sample_document_chunks,
        model="text-embedding-3-small",
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for legacy tests."""
    client = Mock()
    client.embeddings = Mock()
    client.embeddings.create = Mock()

    # Mock response
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536),
    ]
    mock_response.usage = Mock(prompt_tokens=20, total_tokens=20)
    client.embeddings.create.return_value = mock_response

    return client


@pytest.fixture
def mock_faiss_index():
    """Create a mock FAISS index for testing."""
    index = Mock()
    index.ntotal = 0
    index.d = 1536
    index.add = Mock()
    index.search = Mock(return_value=([[[0.9, 0.8, 0.7]]], [[[0, 1, 2]]]))
    return index


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Set default test environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("VECTOR_DATABASE_TYPE", "faiss")
    monkeypatch.setenv("FAISS_INDEX_PATH", "./test_embeddings")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNKING_STRATEGY", "semantic")
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("MAX_WORKERS", "2")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("NBEDR_DISABLE_COORDINATION", "true")


# Async test utilities
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Mock provider factory for testing
@pytest.fixture
def mock_provider_factory():
    """Create a mock provider factory."""
    factory = Mock(spec=EmbeddingProviderFactory)

    def create_provider(provider_type: str, **kwargs):
        provider = Mock(spec=BaseEmbeddingProvider)
        provider.provider_name = provider_type
        provider.model_name = kwargs.get("model", "mock-model")
        provider.dimensions = kwargs.get("dimensions", 1536)
        return provider

    factory.create_provider = create_provider
    return factory


# Test data fixtures
@pytest.fixture
def test_texts():
    """Sample texts for embedding tests."""
    return [
        "This is a test document about machine learning.",
        "Natural language processing is a subset of AI.",
        "Vector databases are used for similarity search.",
        "Embeddings represent text as numerical vectors.",
    ]


@pytest.fixture
def test_metadata():
    """Sample metadata for testing."""
    return [
        {"source": "doc1.txt", "page": 1, "section": "intro"},
        {"source": "doc1.txt", "page": 2, "section": "methods"},
        {"source": "doc2.txt", "page": 1, "section": "results"},
        {"source": "doc2.txt", "page": 2, "section": "conclusion"},
    ]
