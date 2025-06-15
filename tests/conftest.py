"""
Test configuration and fixtures for RAG embedding database tests.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from core.clients.openai_client import EmbeddingClient

# Import core modules
from core.config import EmbeddingConfig
from core.models import DocumentChunk, EmbeddingBatch, VectorDatabaseConfig, VectorDatabaseType
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
        import json

        test_json = {"text": "This is JSON content for testing", "metadata": {"type": "test"}}
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
def sample_config():
    """Create a sample EmbeddingConfig for testing."""
    return EmbeddingConfig(
        datapath=Path("/tmp/test"),
        output="./test_output",
        output_format="jsonl",
        source_type="local",
        chunk_size=512,
        doctype="txt",
        chunking_strategy="fixed",
        openai_api_key="test_key_for_testing",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        batch_size_embeddings=100,
        vector_db_type="faiss",
        workers=1,
        embed_workers=1,
        pace=False,
        rate_limit_enabled=False,
    )


@pytest.fixture
def sample_config_with_env_vars(monkeypatch):
    """Create a config using environment variables."""
    monkeypatch.setenv("EMBEDDING_DATAPATH", "/tmp/env_test")
    monkeypatch.setenv("EMBEDDING_OUTPUT", "./env_output")
    monkeypatch.setenv("EMBEDDING_CHUNK_SIZE", "256")
    monkeypatch.setenv("OPENAI_API_KEY", "env_test_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("VECTOR_DB_TYPE", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "test_pinecone_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "test-env")

    config = EmbeddingConfig.from_env()
    return config


@pytest.fixture
def sample_document_chunk():
    """Create a sample DocumentChunk for testing."""
    return DocumentChunk.create(
        content="This is a test document chunk with some content for testing.",
        source="/path/to/test/document.txt",
        metadata={"type": "test", "chunk_index": 0},
    )


@pytest.fixture
def sample_document_chunks():
    """Create multiple sample DocumentChunks for testing."""
    chunks = []
    for i in range(5):
        chunk = DocumentChunk.create(
            content=f"This is test chunk number {i + 1} with some content.",
            source=f"/path/to/test/document_{i + 1}.txt",
            metadata={"type": "test", "chunk_index": i},
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_embedding_batch(sample_document_chunks):
    """Create a sample EmbeddingBatch for testing."""
    return EmbeddingBatch.create(sample_document_chunks, "text-embedding-3-small")


@pytest.fixture
def mock_embedding_client():
    """Create a mock EmbeddingClient for testing."""
    client = Mock(spec=EmbeddingClient)
    client.model = "text-embedding-3-small"
    client.azure_enabled = False

    # Mock embedding generation
    def mock_generate_embeddings(texts, batch_size=100):
        import random

        embeddings = []
        for text in texts:
            random.seed(hash(text) % (2**32))
            embedding = [random.uniform(-1, 1) for _ in range(1536)]
            embeddings.append(embedding)
        return embeddings

    client.generate_embeddings.side_effect = mock_generate_embeddings
    client.generate_single_embedding.side_effect = lambda text: mock_generate_embeddings([text])[0]
    client.get_model_info.return_value = {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "max_input_tokens": 8191,
        "azure_enabled": False,
    }

    return client


@pytest.fixture
def rate_limit_config():
    """Create a sample RateLimitConfig for testing."""
    return RateLimitConfig(
        enabled=True,
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        requests_per_minute=60,
        tokens_per_minute=1000,
        max_burst_requests=10,
        max_retries=3,
        base_retry_delay=1.0,
    )


@pytest.fixture
def mock_rate_limiter(rate_limit_config):
    """Create a mock RateLimiter for testing."""
    limiter = Mock(spec=RateLimiter)
    limiter.config = rate_limit_config
    limiter.acquire.return_value = 0.0
    limiter.record_response.return_value = None
    limiter.record_error.return_value = None
    limiter.get_statistics.return_value = {
        "enabled": True,
        "strategy": "sliding_window",
        "total_requests": 0,
        "total_tokens": 0,
        "total_wait_time": 0.0,
        "rate_limit_hits": 0,
        "average_response_time": 0.0,
        "current_rate_limit": 60,
        "requests_in_last_minute": 0,
        "tokens_in_last_minute": 0,
    }
    return limiter


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    client = MagicMock()

    # Mock embeddings response
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]  # 1536 dimensions
    client.embeddings.create.return_value = mock_response

    return client


@pytest.fixture
def document_service(sample_config):
    """Create a DocumentService instance for testing."""
    return DocumentService(sample_config)


@pytest.fixture
def vector_db_config():
    """Create a sample VectorDatabaseConfig for testing."""
    return VectorDatabaseConfig(
        db_type=VectorDatabaseType.FAISS,
        connection_params={"index_path": "/tmp/test_index"},
        index_params={"nlist": 100},
        dimension=1536,
        metric="cosine",
    )


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("This is temporary test content.")
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        import json

        test_data = {"test_key": "test_value", "nested": {"key": "value"}}
        json.dump(test_data, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables for testing."""
    # Remove common environment variables that might interfere with tests
    env_vars_to_clean = [
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "AZURE_OPENAI_ENABLED",
        "EMBEDDING_DATAPATH",
        "EMBEDDING_OUTPUT",
        "EMBEDDING_CHUNK_SIZE",
        "EMBEDDING_MODEL",
        "VECTOR_DB_TYPE",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "CHROMA_HOST",
        "CHROMA_PORT",
    ]

    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_file_operations(monkeypatch):
    """Mock file operations for testing."""

    def mock_exists(path):
        return True

    def mock_is_file(path):
        return True

    def mock_is_dir(path):
        return False

    def mock_stat():
        stat_result = MagicMock()
        stat_result.st_size = 1024
        return stat_result

    monkeypatch.setattr("pathlib.Path.exists", mock_exists)
    monkeypatch.setattr("pathlib.Path.is_file", mock_is_file)
    monkeypatch.setattr("pathlib.Path.is_dir", mock_is_dir)
    monkeypatch.setattr("pathlib.Path.stat", lambda self: mock_stat())


# Async test support fixtures
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Parametrized fixtures for testing different configurations
@pytest.fixture(params=["local", "s3", "sharepoint"])
def source_type(request):
    """Parametrized fixture for different source types."""
    return request.param


@pytest.fixture(params=["pdf", "txt", "json", "pptx"])
def doctype(request):
    """Parametrized fixture for different document types."""
    return request.param


@pytest.fixture(params=["semantic", "fixed", "sentence"])
def chunking_strategy(request):
    """Parametrized fixture for different chunking strategies."""
    return request.param


@pytest.fixture(params=["faiss", "pinecone", "chroma"])
def vector_db_type(request):
    """Parametrized fixture for different vector database types."""
    return request.param


# Test data constants
TEST_DOCUMENTS = [
    "This is the first test document with some content.",
    "Here is another test document with different content.",
    "A third document for comprehensive testing purposes.",
]

TEST_EMBEDDINGS = [
    [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
    [0.4, 0.5, 0.6] * 512,
    [0.7, 0.8, 0.9] * 512,
]

TEST_API_DOCS = [
    {
        "user_name": "test_user",
        "api_name": "test_api",
        "api_call": "GET /api/test",
        "api_version": "v1",
        "api_arguments": {"param1": "value1"},
        "functionality": "Test API endpoint",
    },
    {
        "user_name": "test_user2",
        "api_name": "another_api",
        "api_call": "POST /api/another",
        "api_version": "v2",
        "api_arguments": {"param2": "value2"},
        "functionality": "Another test API endpoint",
    },
]


# Helper functions for tests
def create_test_config(**overrides):
    """Helper function to create test configurations with overrides."""
    defaults = {
        "datapath": Path("/tmp/test"),
        "output": "./test_output",
        "output_format": "jsonl",
        "source_type": "local",
        "chunk_size": 512,
        "doctype": "txt",
        "chunking_strategy": "fixed",
        "openai_api_key": "test_key_for_testing",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "batch_size_embeddings": 100,
        "vector_db_type": "faiss",
        "workers": 1,
        "embed_workers": 1,
        "pace": False,
        "rate_limit_enabled": False,
    }
    defaults.update(overrides)
    config_dict = {k: v for k, v in defaults.items()}
    return EmbeddingConfig(**config_dict)


def create_test_chunks(count=3, with_embeddings=False):
    """Helper function to create test document chunks."""
    chunks = []
    for i in range(count):
        chunk = DocumentChunk.create(
            content=f"Test chunk {i + 1} content", source=f"/test/doc_{i + 1}.txt", metadata={"index": i}
        )
        if with_embeddings:
            chunk.embedding = [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] * 512
            chunk.embedding_model = "text-embedding-3-small"
        chunks.append(chunk)
    return chunks
