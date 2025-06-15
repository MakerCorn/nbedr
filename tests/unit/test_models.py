"""
Unit tests for data models and types in the RAG embedding database application.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from core.models import (
    ChunkingStrategy,
    DocType,
    DocumentChunk,
    EmbeddingBatch,
    EmbeddingStats,
    JobStatus,
    OutputFormat,
    ProcessingJob,
    ProcessingResult,
    VectorDatabaseConfig,
    VectorDatabaseType,
    VectorSearchResult,
)


class TestEnums:
    """Test cases for enum types."""

    def test_doctype_enum(self):
        """Test DocType enum values."""
        assert DocType.PDF.value == "pdf"
        assert DocType.TXT.value == "txt"
        assert DocType.JSON.value == "json"
        assert DocType.API.value == "api"
        assert DocType.PPTX.value == "pptx"

    def test_output_format_enum(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.JSONL.value == "jsonl"
        assert OutputFormat.PARQUET.value == "parquet"

    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.FIXED.value == "fixed"
        assert ChunkingStrategy.SENTENCE.value == "sentence"

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    def test_vector_database_type_enum(self):
        """Test VectorDatabaseType enum values."""
        assert VectorDatabaseType.FAISS.value == "faiss"
        assert VectorDatabaseType.PINECONE.value == "pinecone"
        assert VectorDatabaseType.CHROMA.value == "chroma"


class TestDocumentChunk:
    """Test cases for the DocumentChunk class."""

    def test_basic_creation(self):
        """Test basic DocumentChunk creation."""
        chunk = DocumentChunk(
            id="test-123", content="Test content", source="/path/to/test.txt", metadata={"type": "test"}
        )

        assert chunk.id == "test-123"
        assert chunk.content == "Test content"
        assert chunk.source == "/path/to/test.txt"
        assert chunk.metadata == {"type": "test"}
        assert isinstance(chunk.created_at, datetime)
        assert chunk.embedding is None
        assert chunk.embedding_model is None
        assert chunk.vector_id is None

    def test_create_class_method(self):
        """Test DocumentChunk.create class method."""
        chunk = DocumentChunk.create(content="Test content", source="/path/to/test.txt", metadata={"type": "test"})

        assert len(chunk.id) > 0  # Should have generated UUID
        assert chunk.content == "Test content"
        assert chunk.source == "/path/to/test.txt"
        assert chunk.metadata == {"type": "test"}
        assert isinstance(chunk.created_at, datetime)

    def test_create_with_custom_id(self):
        """Test DocumentChunk.create with custom ID."""
        custom_id = "custom-chunk-id"
        chunk = DocumentChunk.create(content="Test content", source="/path/to/test.txt", chunk_id=custom_id)

        assert chunk.id == custom_id

    def test_create_with_embedding(self):
        """Test DocumentChunk.create with embedding data."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = DocumentChunk.create(
            content="Test content",
            source="/path/to/test.txt",
            embedding=embedding,
            embedding_model="text-embedding-3-small",
        )

        assert chunk.embedding == embedding
        assert chunk.embedding_model == "text-embedding-3-small"

    def test_to_dict(self):
        """Test DocumentChunk.to_dict serialization."""
        created_time = datetime.now()
        chunk = DocumentChunk(
            id="test-123",
            content="Test content",
            source="/path/to/test.txt",
            metadata={"type": "test"},
            created_at=created_time,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            vector_id="vec-123",
        )

        result = chunk.to_dict()
        expected = {
            "id": "test-123",
            "content": "Test content",
            "source": "/path/to/test.txt",
            "metadata": {"type": "test"},
            "created_at": created_time.isoformat(),
            "embedding": [0.1, 0.2, 0.3],
            "embedding_model": "test-model",
            "vector_id": "vec-123",
        }

        assert result == expected

    def test_from_dict(self):
        """Test DocumentChunk.from_dict deserialization."""
        created_time = datetime.now()
        data = {
            "id": "test-123",
            "content": "Test content",
            "source": "/path/to/test.txt",
            "metadata": {"type": "test"},
            "created_at": created_time.isoformat(),
            "embedding": [0.1, 0.2, 0.3],
            "embedding_model": "test-model",
            "vector_id": "vec-123",
        }

        chunk = DocumentChunk.from_dict(data)

        assert chunk.id == "test-123"
        assert chunk.content == "Test content"
        assert chunk.source == "/path/to/test.txt"
        assert chunk.metadata == {"type": "test"}
        assert chunk.created_at == created_time
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.embedding_model == "test-model"
        assert chunk.vector_id == "vec-123"

    def test_from_dict_minimal(self):
        """Test DocumentChunk.from_dict with minimal data."""
        data = {
            "id": "test-123",
            "content": "Test content",
            "source": "/path/to/test.txt",
            "metadata": {"type": "test"},
        }

        chunk = DocumentChunk.from_dict(data)

        assert chunk.id == "test-123"
        assert chunk.content == "Test content"
        assert chunk.source == "/path/to/test.txt"
        assert chunk.metadata == {"type": "test"}
        assert isinstance(chunk.created_at, datetime)
        assert chunk.embedding is None
        assert chunk.embedding_model is None
        assert chunk.vector_id is None

    def test_has_embedding(self):
        """Test DocumentChunk.has_embedding method."""
        # Chunk without embedding
        chunk = DocumentChunk.create("content", "source")
        assert not chunk.has_embedding()

        # Chunk with empty embedding
        chunk.embedding = []
        assert not chunk.has_embedding()

        # Chunk with embedding
        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.has_embedding()

    def test_get_embedding_array(self):
        """Test DocumentChunk.get_embedding_array method."""
        chunk = DocumentChunk.create("content", "source")

        # No embedding
        assert chunk.get_embedding_array() is None

        # With embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk.embedding = embedding

        result = chunk.get_embedding_array()
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array(embedding))

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization are consistent."""
        original = DocumentChunk.create(
            content="Test content for roundtrip",
            source="/path/to/test.txt",
            metadata={"type": "test", "index": 42},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            embedding_model="test-model",
        )
        original.vector_id = "vec-test-123"

        # Serialize to dict
        data = original.to_dict()

        # Deserialize back
        restored = DocumentChunk.from_dict(data)

        # Compare all fields
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.source == original.source
        assert restored.metadata == original.metadata
        assert restored.created_at == original.created_at
        assert restored.embedding == original.embedding
        assert restored.embedding_model == original.embedding_model
        assert restored.vector_id == original.vector_id


class TestEmbeddingBatch:
    """Test cases for the EmbeddingBatch class."""

    def test_basic_creation(self, sample_document_chunks):
        """Test basic EmbeddingBatch creation."""
        batch = EmbeddingBatch(id="batch-123", chunks=sample_document_chunks, model="text-embedding-3-small")

        assert batch.id == "batch-123"
        assert batch.chunks == sample_document_chunks
        assert batch.model == "text-embedding-3-small"
        assert batch.status == "pending"
        assert isinstance(batch.created_at, datetime)
        assert batch.processed_at is None
        assert batch.error is None

    def test_create_class_method(self, sample_document_chunks):
        """Test EmbeddingBatch.create class method."""
        batch = EmbeddingBatch.create(sample_document_chunks, "test-model")

        assert len(batch.id) > 0  # Should have generated UUID
        assert batch.chunks == sample_document_chunks
        assert batch.model == "test-model"
        assert batch.status == "pending"
        assert isinstance(batch.created_at, datetime)

    def test_mark_completed(self, embedding_batch):
        """Test EmbeddingBatch.mark_completed method."""
        start_time = datetime.now()
        embedding_batch.mark_completed()

        assert embedding_batch.status == "completed"
        assert embedding_batch.processed_at is not None
        assert embedding_batch.processed_at >= start_time
        assert embedding_batch.error is None

    def test_mark_failed(self, embedding_batch):
        """Test EmbeddingBatch.mark_failed method."""
        error_msg = "Test error message"
        start_time = datetime.now()

        embedding_batch.mark_failed(error_msg)

        assert embedding_batch.status == "failed"
        assert embedding_batch.processed_at is not None
        assert embedding_batch.processed_at >= start_time
        assert embedding_batch.error == error_msg


class TestVectorSearchResult:
    """Test cases for the VectorSearchResult class."""

    def test_basic_creation(self, sample_document_chunks):
        """Test basic VectorSearchResult creation."""
        sample_document_chunk = sample_document_chunks[0]
        result = VectorSearchResult(
            id="test-id",
            content="Test content",
            source="test-source",
            metadata={},
            similarity_score=0.85,
            embedding_model="test-model",
            chunk=sample_document_chunk,
            score=0.85,
            rank=1,
        )

        assert result.chunk == sample_document_chunk
        assert result.score == 0.85
        assert result.rank == 1

    def test_to_dict(self, sample_document_chunks):
        """Test VectorSearchResult.to_dict serialization."""
        sample_document_chunk = sample_document_chunks[0]
        result = VectorSearchResult(
            id="test-id",
            content="Test content",
            source="test-source",
            metadata={},
            similarity_score=0.85,
            embedding_model="test-model",
            chunk=sample_document_chunk,
            score=0.85,
            rank=1,
        )

        serialized = result.to_dict()

        assert "chunk" in serialized
        assert "score" in serialized
        assert "rank" in serialized
        assert serialized["score"] == 0.85
        assert serialized["rank"] == 1
        assert serialized["chunk"] == sample_document_chunk.to_dict()


class TestVectorDatabaseConfig:
    """Test cases for the VectorDatabaseConfig class."""

    def test_basic_creation(self):
        """Test basic VectorDatabaseConfig creation."""
        config = VectorDatabaseConfig(
            db_type=VectorDatabaseType.FAISS,
            connection_params={"host": "localhost", "port": 8000},
            index_params={"nlist": 100},
            dimension=1536,
            metric="cosine",
        )

        assert config.db_type == VectorDatabaseType.FAISS
        assert config.connection_params == {"host": "localhost", "port": 8000}
        assert config.index_params == {"nlist": 100}
        assert config.dimension == 1536
        assert config.metric == "cosine"

    def test_default_values(self):
        """Test VectorDatabaseConfig with default values."""
        config = VectorDatabaseConfig(db_type=VectorDatabaseType.PINECONE)

        assert config.db_type == VectorDatabaseType.PINECONE
        assert config.connection_params == {}
        assert config.index_params == {}
        assert config.dimension == 1536
        assert config.metric == "cosine"

    def test_to_dict(self):
        """Test VectorDatabaseConfig.to_dict serialization."""
        config = VectorDatabaseConfig(
            db_type=VectorDatabaseType.CHROMA,
            connection_params={"host": "localhost"},
            index_params={"ef": 200},
            dimension=768,
            metric="euclidean",
        )

        result = config.to_dict()
        expected = {
            "db_type": "chroma",
            "connection_params": {"host": "localhost"},
            "index_params": {"ef": 200},
            "dimension": 768,
            "metric": "euclidean",
        }

        assert result == expected


class TestProcessingJob:
    """Test cases for the ProcessingJob class."""

    def test_basic_creation(self, sample_document_chunks):
        """Test basic ProcessingJob creation."""
        sample_document_chunk = sample_document_chunks[0]
        job = ProcessingJob(id="job-123", chunk=sample_document_chunk, embedding_model="text-embedding-3-small")

        assert job.id == "job-123"
        assert job.chunk == sample_document_chunk
        assert job.embedding_model == "text-embedding-3-small"
        assert job.status == "pending"
        assert isinstance(job.created_at, datetime)
        assert job.processed_at is None

    def test_create_class_method(self, sample_document_chunks):
        """Test ProcessingJob.create class method."""
        sample_document_chunk = sample_document_chunks[0]
        job = ProcessingJob.create(sample_document_chunk, "test-model")

        assert len(job.id) > 0  # Should have generated UUID
        assert job.chunk == sample_document_chunk
        assert job.embedding_model == "test-model"
        assert job.status == "pending"
        assert isinstance(job.created_at, datetime)


class TestProcessingResult:
    """Test cases for the ProcessingResult class."""

    def test_basic_creation(self, sample_document_chunks):
        """Test basic ProcessingResult creation."""
        result = ProcessingResult(
            job_id="job-123",
            success=True,
            embedded_chunks=sample_document_chunks,
            processing_time=2.5,
            token_usage={"total_tokens": 1000, "prompt_tokens": 800, "completion_tokens": 200},
            error=None,
        )

        assert result.job_id == "job-123"
        assert result.success is True
        assert result.embedded_chunks == sample_document_chunks
        assert result.processing_time == 2.5
        assert result.token_usage == {"total_tokens": 1000, "prompt_tokens": 800, "completion_tokens": 200}
        assert result.error is None

    def test_default_values(self):
        """Test ProcessingResult with default values."""
        result = ProcessingResult(job_id="job-123", success=False)

        assert result.job_id == "job-123"
        assert result.success is False
        assert result.embedded_chunks == []
        assert result.processing_time == 0.0
        assert result.token_usage == {}
        assert result.error is None

    def test_to_dict(self, sample_document_chunks):
        """Test ProcessingResult.to_dict serialization."""
        result = ProcessingResult(
            job_id="job-123",
            success=True,
            embedded_chunks=sample_document_chunks,
            processing_time=1.5,
            token_usage={"total_tokens": 500},
            error=None,
        )

        serialized = result.to_dict()

        assert serialized["job_id"] == "job-123"
        assert serialized["success"] is True
        assert len(serialized["embedded_chunks"]) == len(sample_document_chunks)
        assert serialized["processing_time"] == 1.5
        assert serialized["token_usage"] == {"total_tokens": 500}
        assert serialized["error"] is None

        # Check that chunks are properly serialized
        for i, chunk_dict in enumerate(serialized["embedded_chunks"]):
            assert chunk_dict == sample_document_chunks[i].to_dict()

    def test_failed_result(self):
        """Test ProcessingResult for failed processing."""
        result = ProcessingResult(job_id="job-456", success=False, error="Processing failed due to API error")

        assert result.job_id == "job-456"
        assert result.success is False
        assert result.embedded_chunks == []
        assert result.error == "Processing failed due to API error"


class TestEmbeddingStats:
    """Test cases for the EmbeddingStats class."""

    def test_basic_creation(self):
        """Test basic EmbeddingStats creation."""
        stats = EmbeddingStats()

        assert stats.total_chunks == 0
        assert stats.embedded_chunks == 0
        assert stats.failed_chunks == 0
        assert stats.total_tokens == 0
        assert stats.total_processing_time == 0.0
        assert stats.average_embedding_time == 0.0

    def test_custom_initialization(self):
        """Test EmbeddingStats with custom initial values."""
        stats = EmbeddingStats(
            total_chunks=100,
            embedded_chunks=80,
            failed_chunks=20,
            total_tokens=50000,
            total_processing_time=120.0,
            average_embedding_time=1.2,
        )

        assert stats.total_chunks == 100
        assert stats.embedded_chunks == 80
        assert stats.failed_chunks == 20
        assert stats.total_tokens == 50000
        assert stats.total_processing_time == 120.0
        assert stats.average_embedding_time == 1.2

    def test_update_with_successful_result(self, sample_document_chunks):
        """Test EmbeddingStats.update with successful result."""
        stats = EmbeddingStats()

        result = ProcessingResult(
            job_id="job-1",
            success=True,
            embedded_chunks=sample_document_chunks,
            processing_time=2.0,
            token_usage={"total_tokens": 1000},
        )

        stats.update(result)

        assert stats.embedded_chunks == len(sample_document_chunks)
        assert stats.failed_chunks == 0
        assert stats.total_tokens == 1000
        assert stats.total_processing_time == 2.0
        assert stats.average_embedding_time == 2.0

    def test_update_with_failed_result(self):
        """Test EmbeddingStats.update with failed result."""
        stats = EmbeddingStats()

        result = ProcessingResult(
            job_id="job-1", success=False, processing_time=1.0, token_usage={"total_tokens": 0}, error="API error"
        )

        stats.update(result)

        assert stats.embedded_chunks == 0
        assert stats.failed_chunks == 1
        assert stats.total_tokens == 0
        assert stats.total_processing_time == 1.0
        assert stats.average_embedding_time == 1.0

    def test_update_multiple_results(self, sample_document_chunks):
        """Test EmbeddingStats.update with multiple results."""
        stats = EmbeddingStats()

        # First successful result
        result1 = ProcessingResult(
            job_id="job-1",
            success=True,
            embedded_chunks=sample_document_chunks[:2],  # 2 chunks
            processing_time=1.5,
            token_usage={"total_tokens": 500},
        )

        # Second successful result
        result2 = ProcessingResult(
            job_id="job-2",
            success=True,
            embedded_chunks=sample_document_chunks[2:4],  # 2 more chunks
            processing_time=2.5,
            token_usage={"total_tokens": 800},
        )

        # Failed result
        result3 = ProcessingResult(job_id="job-3", success=False, processing_time=0.5, token_usage={"total_tokens": 0})

        stats.update(result1)
        stats.update(result2)
        stats.update(result3)

        assert stats.embedded_chunks == 4
        assert stats.failed_chunks == 1
        assert stats.total_tokens == 1300
        assert stats.total_processing_time == 4.5
        assert stats.average_embedding_time == 1.5  # 4.5 / 3 results

    def test_to_dict(self):
        """Test EmbeddingStats.to_dict serialization."""
        stats = EmbeddingStats(
            total_chunks=100,
            embedded_chunks=80,
            failed_chunks=20,
            total_tokens=50000,
            total_processing_time=120.0,
            average_embedding_time=1.5,
        )

        result = stats.to_dict()

        assert result["total_chunks"] == 100
        assert result["embedded_chunks"] == 80
        assert result["failed_chunks"] == 20
        assert result["total_tokens"] == 50000
        assert result["total_processing_time"] == 120.0
        assert result["average_embedding_time"] == 1.5
        assert result["success_rate"] == 80.0  # 80/100 * 100

    def test_success_rate_calculation(self):
        """Test success rate calculation in to_dict."""
        # Test with no chunks
        stats = EmbeddingStats()
        result = stats.to_dict()
        assert result["success_rate"] == 0.0

        # Test with some chunks
        stats = EmbeddingStats(total_chunks=50, embedded_chunks=40)
        result = stats.to_dict()
        assert result["success_rate"] == 80.0

        # Test with perfect success rate
        stats = EmbeddingStats(total_chunks=10, embedded_chunks=10)
        result = stats.to_dict()
        assert result["success_rate"] == 100.0

    def test_average_embedding_time_calculation(self, sample_document_chunks):
        """Test that average embedding time is calculated correctly."""
        stats = EmbeddingStats()

        # Add results with different processing times
        results = [
            ProcessingResult("job-1", True, sample_document_chunks[:1], 1.0, {"total_tokens": 100}),
            ProcessingResult("job-2", True, sample_document_chunks[:1], 3.0, {"total_tokens": 200}),
            ProcessingResult("job-3", False, [], 2.0, {"total_tokens": 0}),
        ]

        for result in results:
            stats.update(result)

        # Average should be (1.0 + 3.0 + 2.0) / 3 = 2.0
        assert stats.average_embedding_time == 2.0
        assert stats.total_processing_time == 6.0
        assert stats.embedded_chunks == 2
        assert stats.failed_chunks == 1


class TestModelIntegration:
    """Integration tests for model interactions."""

    def test_chunk_to_batch_workflow(self, sample_document_chunks):
        """Test workflow from chunks to embedding batch."""
        # Create batch from chunks
        batch = EmbeddingBatch.create(sample_document_chunks, "text-embedding-3-small")

        assert len(batch.chunks) == len(sample_document_chunks)
        assert batch.model == "text-embedding-3-small"
        assert batch.status == "pending"

        # Simulate adding embeddings to chunks
        for i, chunk in enumerate(batch.chunks):
            chunk.embedding = [0.1 * (i + 1)] * 1536
            chunk.embedding_model = batch.model

        # Mark batch as completed
        batch.mark_completed()

        assert batch.status == "completed"
        assert batch.processed_at is not None

        # Verify all chunks have embeddings
        for chunk in batch.chunks:
            assert chunk.has_embedding()
            assert chunk.embedding_model == "text-embedding-3-small"

    def test_processing_job_to_result_workflow(self, sample_document_chunks):
        """Test workflow from processing job to result."""
        sample_document_chunk = sample_document_chunks[0]
        # Create processing job
        job = ProcessingJob.create(sample_document_chunk, "text-embedding-3-small")

        assert job.status == "pending"
        assert job.chunk == sample_document_chunk

        # Simulate processing
        sample_document_chunk.embedding = [0.1, 0.2, 0.3] * 512
        sample_document_chunk.embedding_model = job.embedding_model

        # Create result
        result = ProcessingResult(
            job_id=job.id,
            success=True,
            embedded_chunks=[sample_document_chunk],
            processing_time=1.5,
            token_usage={"total_tokens": 100},
        )

        assert result.success
        assert len(result.embedded_chunks) == 1
        assert result.embedded_chunks[0].has_embedding()

    def test_stats_aggregation_workflow(self, sample_document_chunks):
        """Test aggregating statistics from multiple processing results."""
        stats = EmbeddingStats()
        stats.total_chunks = len(sample_document_chunks)

        # Create multiple processing results
        results = []
        for i, chunk in enumerate(sample_document_chunks):
            chunk.embedding = [0.1 * (i + 1)] * 100
            chunk.embedding_model = "test-model"

            success = i < 3  # First 3 succeed, rest fail
            result = ProcessingResult(
                job_id=f"job-{i}",
                success=success,
                embedded_chunks=[chunk] if success else [],
                processing_time=1.0 + i * 0.5,
                token_usage={"total_tokens": 100 * (i + 1)} if success else {"total_tokens": 0},
            )
            results.append(result)

        # Update stats with all results
        for result in results:
            stats.update(result)

        assert stats.total_chunks == len(sample_document_chunks)
        assert stats.embedded_chunks == 3  # First 3 succeeded
        assert stats.failed_chunks == 2  # Last 2 failed
        assert stats.total_tokens == 600  # 100 + 200 + 300

        # Check success rate
        stats_dict = stats.to_dict()
        assert stats_dict["success_rate"] == 60.0  # 3/5 * 100

    def test_json_serialization_roundtrip(self, sample_document_chunks):
        """Test JSON serialization and deserialization of complex model structures."""
        # Create a complex structure
        batch = EmbeddingBatch.create(sample_document_chunks, "text-embedding-3-small")

        # Add embeddings to chunks
        for i, chunk in enumerate(batch.chunks):
            chunk.embedding = [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)]
            chunk.embedding_model = "test-model"
            chunk.vector_id = f"vec-{i}"

        batch.mark_completed()

        # Serialize each chunk to dict
        chunk_dicts = [chunk.to_dict() for chunk in batch.chunks]

        # Convert to JSON and back
        json_str = json.dumps(chunk_dicts)
        restored_dicts = json.loads(json_str)

        # Restore chunks from dicts
        restored_chunks = [DocumentChunk.from_dict(d) for d in restored_dicts]

        # Verify everything is preserved
        assert len(restored_chunks) == len(batch.chunks)
        for original, restored in zip(batch.chunks, restored_chunks):
            assert original.id == restored.id
            assert original.content == restored.content
            assert original.source == restored.source
            assert original.metadata == restored.metadata
            assert original.embedding == restored.embedding
            assert original.embedding_model == restored.embedding_model
            assert original.vector_id == restored.vector_id
