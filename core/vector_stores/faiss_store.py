"""
FAISS vector store implementation.
"""
# mypy: disable-error-code="attr-defined,unreachable"

import fcntl
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..models import DocumentChunk, VectorSearchResult
from ..utils.instance_coordinator import InstanceCoordinator
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation of vector store."""

    def __init__(self, config: Dict[str, Any], coordinator: Optional[InstanceCoordinator] = None):
        """Initialize FAISS vector store."""
        super().__init__(config)

        self.index_path = Path(config.get("faiss_index_path", "./faiss_index"))
        self.index_type = config.get("faiss_index_type", "IndexFlatIP")
        self.embedding_dim = config.get("embedding_dimensions", 1536)
        self.coordinator = coordinator

        # Use instance-specific path if coordinator is provided
        if self.coordinator:
            suggested_paths = self.coordinator.suggest_instance_specific_paths()
            if "faiss_index_path" in suggested_paths:
                self.index_path = Path(suggested_paths["faiss_index_path"])
                logger.info(f"Using instance-specific FAISS path: {self.index_path}")

        self.index = None
        self.document_map: Dict[int, Dict[str, Any]] = {}  # Maps index positions to document metadata
        self.next_id = 0

        # File locking for concurrent access
        self.lock_file = self.index_path.parent / f"{self.index_path.name}.lock"

    async def initialize(self) -> None:
        """Initialize the FAISS index."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)

            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"

            # Load existing index if it exists
            if index_file.exists() and metadata_file.exists():
                logger.info(f"Loading existing FAISS index from {index_file}")

                # Use file locking when loading to prevent conflicts
                try:
                    with self._acquire_file_lock():
                        self.index = faiss.read_index(str(index_file))

                        with open(metadata_file, "rb") as f:
                            data = pickle.load(f)
                            self.document_map = data.get("document_map", {})
                            self.next_id = data.get("next_id", 0)
                except (FileNotFoundError, pickle.PickleError) as e:
                    logger.warning(f"Failed to load existing index, creating new one: {e}")
                    self._create_new_index()
            else:
                logger.info(f"Creating new FAISS index: {self.index_type}")
                self._create_new_index()

        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise

    def _create_new_index(self):
        """Create a new FAISS index based on the configured type."""
        if self.index_type == "IndexFlatIP":
            # Inner Product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "IndexHNSW":
            # Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to FAISS index."""
        try:
            # Apply rate limiting for vector store operations
            await self._apply_rate_limiting("add_documents")

            start_time = time.time()

            vectors = []
            vector_ids = []

            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue

                vector_id = chunk.vector_id or chunk.id
                vector_ids.append(vector_id)

                # Convert embedding to numpy array
                embedding = np.array(chunk.embedding, dtype=np.float32).reshape(1, -1)
                vectors.append(embedding)

                # Store document metadata
                self.document_map[self.next_id] = {
                    "id": vector_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                    "embedding_model": chunk.embedding_model,
                    "created_at": chunk.created_at.isoformat(),
                }
                self.next_id += 1

            if vectors:
                # Concatenate all vectors
                embeddings_matrix = np.vstack(vectors)

                # Train index if needed (for IVF indices)
                if hasattr(self.index, "is_trained") and not self.index.is_trained:
                    self.index.train(embeddings_matrix)

                # Add vectors to index
                self.index.add(embeddings_matrix)

                # Save index and metadata
                await self._save_index()

                logger.info(f"Added {len(vectors)} documents to FAISS index")

            # Record operation response time
            response_time = time.time() - start_time
            self._record_operation_response(response_time, "add_documents")

            return vector_ids

        except Exception as e:
            self._record_operation_error("add_documents_error")
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise

    async def search(
        self, query_embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in FAISS index."""
        try:
            # Apply rate limiting for vector store operations
            await self._apply_rate_limiting("search")

            start_time = time.time()

            # Convert query to numpy array
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            # Perform search
            scores, indices = self.index.search(query_vector, top_k)

            search_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break

                # Get document metadata
                doc_data = self.document_map.get(idx)
                if not doc_data:
                    logger.warning(f"No metadata found for index {idx}")
                    continue

                # Apply filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key == "source" and doc_data.get("source") != value:
                            skip = True
                            break
                        elif key == "embedding_model" and doc_data.get("embedding_model") != value:
                            skip = True
                            break
                        # Add more filter logic as needed

                    if skip:
                        continue

                search_result = VectorSearchResult(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    source=doc_data["source"],
                    metadata=doc_data.get("metadata", {}),
                    similarity_score=float(score),
                    embedding_model=doc_data.get("embedding_model"),
                    created_at=doc_data.get("created_at"),
                )
                search_results.append(search_result)

            # Record operation response time
            response_time = time.time() - start_time
            self._record_operation_response(response_time, "search")

            logger.info(f"Found {len(search_results)} results in FAISS index")
            return search_results

        except Exception as e:
            self._record_operation_error("search_error")
            logger.error(f"Failed to search FAISS index: {e}")
            raise

    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """Delete documents from FAISS index."""
        # FAISS doesn't support deletion directly
        # We would need to rebuild the index without the deleted documents
        logger.warning("FAISS doesn't support direct deletion. Consider rebuilding the index.")
        return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        try:
            return {
                "index_path": str(self.index_path),
                "index_type": self.index_type,
                "document_count": self.index.ntotal if self.index else 0,
                "embedding_dimension": self.embedding_dim,
                "is_trained": getattr(self.index, "is_trained", True) if self.index else False,
            }

        except Exception as e:
            logger.error(f"Failed to get FAISS stats: {e}")
            return {"error": str(e)}

    async def _save_index(self):
        """Save FAISS index and metadata to disk with file locking."""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"

            # Acquire exclusive lock for saving
            with self._acquire_file_lock():
                # Save FAISS index
                faiss.write_index(self.index, str(index_file))

                # Save metadata
                with open(metadata_file, "wb") as f:
                    pickle.dump({"document_map": self.document_map, "next_id": self.next_id}, f)

                logger.debug(f"Saved FAISS index to {index_file}")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def _acquire_file_lock(self):
        """Context manager for file locking."""

        class FileLock:
            def __init__(self, lock_file):
                self.lock_file = lock_file
                self.lock_fd = None

            def __enter__(self):
                self.lock_fd = open(self.lock_file, "w")
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
                logger.debug(f"Acquired file lock: {self.lock_file}")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.lock_fd:
                    fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                    self.lock_fd.close()
                try:
                    os.unlink(self.lock_file)
                except FileNotFoundError:
                    pass
                logger.debug(f"Released file lock: {self.lock_file}")

        return FileLock(self.lock_file)

    async def close(self) -> None:
        """Close FAISS vector store."""
        # Save index before closing
        if self.index:
            await self._save_index()

        logger.info("FAISS vector store closed")
