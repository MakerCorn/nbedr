"""
Data models and types for the RAG embedding database application.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np


# Enum types
class DocType(Enum):
    PDF = "pdf"
    TXT = "txt"
    JSON = "json"
    API = "api"
    PPTX = "pptx"


class OutputFormat(Enum):
    JSONL = "jsonl"
    PARQUET = "parquet"


class ChunkingStrategy(Enum):
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SENTENCE = "sentence"


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VectorDatabaseType(Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    AZURE_AI_SEARCH = "azure_ai_search"
    AWS_ELASTICSEARCH = "aws_elasticsearch"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with optional embedding vector."""

    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    # New fields for embedding functionality
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    vector_id: Optional[str] = None  # ID in vector database

    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        embedding_model: Optional[str] = None,
    ) -> "DocumentChunk":
        """Create a new document chunk with generated ID."""
        return cls(
            id=chunk_id or str(uuid.uuid4()),
            content=content,
            source=source,
            metadata=metadata or {},
            embedding=embedding,
            embedding_model=embedding_model,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "vector_id": self.vector_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        return cls(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            metadata=data["metadata"],
            created_at=created_at,
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model"),
            vector_id=data.get("vector_id"),
        )

    def has_embedding(self) -> bool:
        """Check if chunk has an embedding vector."""
        return self.embedding is not None and len(self.embedding) > 0

    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Get embedding as numpy array."""
        if self.embedding:
            return np.array(self.embedding)
        return None


@dataclass
class EmbeddingBatch:
    """Represents a batch of chunks to be embedded."""

    id: str
    chunks: List[DocumentChunk]
    model: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error: Optional[str] = None

    @classmethod
    def create(cls, chunks: List[DocumentChunk], model: str) -> "EmbeddingBatch":
        """Create a new embedding batch."""
        return cls(id=str(uuid.uuid4()), chunks=chunks, model=model)

    def mark_completed(self):
        """Mark batch as completed."""
        self.status = "completed"
        self.processed_at = datetime.now()

    def mark_failed(self, error: str):
        """Mark batch as failed with error."""
        self.status = "failed"
        self.processed_at = datetime.now()
        self.error = error


@dataclass
class VectorSearchResult:
    """Represents a search result from vector database."""

    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    similarity_score: float
    embedding_model: str
    created_at: Optional[str] = None
    # For backward compatibility with tests
    chunk: Optional[DocumentChunk] = None
    score: Optional[float] = None
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }
        if self.chunk:
            result["chunk"] = self.chunk.to_dict()
        if self.score is not None:
            result["score"] = self.score
        if self.rank is not None:
            result["rank"] = self.rank
        return result


@dataclass
class VectorDatabaseConfig:
    """Configuration for vector database operations."""

    db_type: VectorDatabaseType
    connection_params: Dict[str, Any] = field(default_factory=dict)
    index_params: Dict[str, Any] = field(default_factory=dict)

    # Common parameters
    dimension: int = 1536
    metric: str = "cosine"  # cosine, euclidean, dot_product

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "db_type": self.db_type.value,
            "connection_params": self.connection_params,
            "index_params": self.index_params,
            "dimension": self.dimension,
            "metric": self.metric,
        }


@dataclass
class ProcessingJob:
    """Represents a processing job for document chunks embedding."""

    id: str
    chunk: DocumentChunk
    embedding_model: str
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None

    @classmethod
    def create(cls, chunk: DocumentChunk, embedding_model: str) -> "ProcessingJob":
        """Create a new processing job."""
        return cls(id=str(uuid.uuid4()), chunk=chunk, embedding_model=embedding_model)


@dataclass
class ProcessingResult:
    """Results from processing an embedding job."""

    job_id: str
    success: bool
    embedded_chunks: List[DocumentChunk] = field(default_factory=list)
    processing_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "embedded_chunks": [chunk.to_dict() for chunk in self.embedded_chunks],
            "processing_time": self.processing_time,
            "token_usage": self.token_usage,
            "error": self.error,
        }


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""

    total_chunks: int = 0
    embedded_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_processing_time: float = 0.0
    average_embedding_time: float = 0.0

    def update(self, result: ProcessingResult):
        """Update stats with processing result."""
        if result.success:
            self.embedded_chunks += len(result.embedded_chunks)
        else:
            self.failed_chunks += 1

        self.total_tokens += result.token_usage.get("total_tokens", 0)
        self.total_processing_time += result.processing_time

        # Recalculate average
        total_processed = self.embedded_chunks + self.failed_chunks
        if total_processed > 0:
            self.average_embedding_time = self.total_processing_time / total_processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "failed_chunks": self.failed_chunks,
            "total_tokens": self.total_tokens,
            "total_processing_time": self.total_processing_time,
            "average_embedding_time": self.average_embedding_time,
            "success_rate": self.embedded_chunks / max(1, self.total_chunks) * 100,
        }
