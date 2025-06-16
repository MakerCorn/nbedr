"""
PostgreSQL with pgvector extension vector store implementation.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

import asyncpg
from asyncpg import Connection, Pool
from asyncpg.pool import PoolConnectionProxy

from ..models import DocumentChunk, VectorSearchResult
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector extension implementation of vector store."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PGVector store.

        Args:
            config: Configuration dictionary containing:
                - pgvector_host: PostgreSQL host (default: localhost)
                - pgvector_port: PostgreSQL port (default: 5432)
                - pgvector_database: Database name (default: vectordb)
                - pgvector_user: Database user (default: postgres)
                - pgvector_password: Database password (required)
                - pgvector_table_name: Table name (default: rag_embeddings)
                - embedding_dimensions: Embedding dimensions (default: 1536)

        Raises:
            ValueError: If required configuration values are missing
        """
        super().__init__(config)

        self.host = str(config.get("pgvector_host", "localhost"))
        self.port = int(config.get("pgvector_port", 5432))
        self.database = str(config.get("pgvector_database", "vectordb"))
        self.user = str(config.get("pgvector_user", "postgres"))
        self.password = config.get("pgvector_password")
        self.table_name = str(config.get("pgvector_table_name", "rag_embeddings"))
        self.embedding_dimensions = int(config.get("embedding_dimensions", 1536))

        if not self.password:
            raise ValueError("PostgreSQL password is required for pgvector")

        self.pool: Optional[Pool] = None
        self._validate_table_name()

    def _validate_table_name(self) -> None:
        """Validate table name to prevent SQL injection."""
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.table_name):
            raise ValueError(f"Invalid table name: {self.table_name}. Must be alphanumeric with underscores.")

    async def _get_connection_pool(self) -> Pool:
        """Get or create connection pool.

        Returns:
            asyncpg connection pool

        Raises:
            RuntimeError: If connection pool creation fails
        """
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=2,
                    max_size=10,
                    command_timeout=60,
                )
                if not self.pool:
                    raise RuntimeError("Failed to create connection pool")
            except Exception as e:
                raise RuntimeError(f"Failed to create connection pool: {e}")

        return self.pool

    async def initialize(self) -> None:
        """Initialize the pgvector table and extension.

        Creates the pgvector extension if not exists and sets up the required table
        with appropriate indexes.

        Raises:
            RuntimeError: If table initialization fails
        """
        try:
            pool = await self._get_connection_pool()

            async with pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Enabled pgvector extension")

                # Check if table exists
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = $1
                    );
                    """,
                    self.table_name,
                )

                if table_exists:
                    logger.info(f"Using existing pgvector table: {self.table_name}")
                    return

                logger.info(f"Creating new pgvector table: {self.table_name}")

                # Create table with vector column
                await conn.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        embedding_model TEXT,
                        content_vector vector({self.embedding_dimensions}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    """
                )

                # Create vector similarity index
                await conn.execute(
                    f"""
                    CREATE INDEX ON {self.table_name}
                    USING ivfflat (content_vector vector_cosine_ops)
                    WITH (lists = 100);
                    """
                )

                # Create additional indexes
                await conn.execute(
                    f"""
                    CREATE INDEX idx_{self.table_name}_source ON {self.table_name}(source);
                    CREATE INDEX idx_{self.table_name}_metadata ON {self.table_name} USING GIN(metadata);
                    """
                )

                logger.info(f"Created pgvector table: {self.table_name}")

        except Exception as e:
            logger.error(f"Failed to initialize pgvector table: {e}")
            raise RuntimeError(f"Failed to initialize pgvector table: {e}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to pgvector.

        Args:
            chunks: List of document chunks to add

        Returns:
            List of vector IDs for the added documents

        Raises:
            RuntimeError: If document addition fails
        """
        try:
            # Apply rate limiting for vector store operations
            await self._apply_rate_limiting("add_documents")

            start_time = time.time()

            pool = await self._get_connection_pool()
            vector_ids: List[str] = []

            async with pool.acquire() as conn:
                # Create a prepared statement for better performance
                # Table name is validated in constructor to prevent injection
                insert_stmt = await conn.prepare(
                    f"""
                    INSERT INTO {self.table_name}
                    (id, content, source, metadata, embedding_model, content_vector, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata,
                        embedding_model = EXCLUDED.embedding_model,
                        content_vector = EXCLUDED.content_vector,
                        created_at = EXCLUDED.created_at;
                    """  # nosec B608
                )

                for chunk in chunks:
                    if not chunk.embedding:
                        logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                        continue

                    vector_id = chunk.vector_id or chunk.id
                    vector_ids.append(vector_id)

                    # Insert document using prepared statement
                    await insert_stmt.fetch(
                        vector_id,
                        chunk.content,
                        chunk.source,
                        json.dumps(chunk.metadata) if chunk.metadata else "{}",
                        chunk.embedding_model,
                        chunk.embedding,
                        chunk.created_at or datetime.utcnow(),
                    )

                logger.info(f"Successfully added {len(vector_ids)} documents to pgvector")

            # Record operation response time
            response_time = time.time() - start_time
            self._record_operation_response(response_time, "add_documents")

            return vector_ids

        except Exception as e:
            self._record_operation_error("add_documents_error")
            logger.error(f"Failed to add documents to pgvector: {e}")
            raise RuntimeError(f"Failed to add documents to pgvector: {e}")

    async def search(
        self, query_embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in pgvector.

        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of search results

        Raises:
            RuntimeError: If search fails
        """
        try:
            # Apply rate limiting for vector store operations
            await self._apply_rate_limiting("search")

            start_time = time.time()

            pool = await self._get_connection_pool()

            # Build WHERE clause for filters
            where_clause = ""
            filter_params = []
            param_idx = 2  # Start from $2 since $1 is the query vector

            if filters:
                conditions = []
                for key, value in filters.items():
                    if key == "source":
                        conditions.append(f"source = ${param_idx}")
                        filter_params.append(value)
                        param_idx += 1
                    elif key.startswith("metadata."):
                        # Handle metadata filters
                        metadata_key = key[9:]  # Remove "metadata." prefix
                        conditions.append(f"metadata->>${param_idx} = ${param_idx + 1}")
                        filter_params.extend([metadata_key, str(value)])
                        param_idx += 2

                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"

            # Table name is validated in constructor to prevent injection
            query = f"""
                SELECT id, content, source, metadata, embedding_model, created_at,
                       1 - (content_vector <=> $1) as similarity_score
                FROM {self.table_name}
                {where_clause}
                ORDER BY content_vector <=> $1
                LIMIT {top_k};
            """  # nosec B608

            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *filter_params)

                search_results: List[VectorSearchResult] = []
                for row in rows:
                    search_results.append(
                        VectorSearchResult(
                            id=str(row["id"]),
                            content=str(row["content"]),
                            source=str(row["source"]),
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                            similarity_score=float(row["similarity_score"]),
                            embedding_model=(
                                str(row["embedding_model"]) if row["embedding_model"] else "unknown"
                            ),  # Default value when None
                            created_at=row["created_at"].isoformat() if row["created_at"] else None,
                        )
                    )

                response_time = time.time() - start_time
                self._record_operation_response(response_time, "search")

                logger.info(f"Found {len(search_results)} results in pgvector")
                return search_results

        except Exception as e:
            self._record_operation_error("search_error")
            logger.error(f"Failed to search pgvector: {e}")
            raise RuntimeError(f"Failed to search pgvector: {e}")

    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """Delete documents from pgvector.

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            True if any documents were deleted, False otherwise
        """
        try:
            pool = await self._get_connection_pool()

            async with pool.acquire() as conn:
                # Table name is validated in constructor to prevent injection
                result = await conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE id = ANY($1::text[]);
                    """,  # nosec B608
                    vector_ids,
                )

                deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
                logger.info(f"Deleted {deleted_count} documents from pgvector")

                return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to delete documents from pgvector: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pgvector table.

        Returns:
            Dictionary containing table statistics
        """
        try:
            pool = await self._get_connection_pool()

            async with pool.acquire() as conn:
                # Table name is validated in constructor to prevent injection
                stats = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) as document_count,
                        pg_total_relation_size($1::text) as storage_size,
                        MIN(created_at) as earliest_document,
                        MAX(created_at) as latest_document
                    FROM {self.table_name};
                    """,  # nosec B608
                    self.table_name,
                )

                table_info = await conn.fetchrow(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        tableowner
                    FROM pg_tables
                    WHERE tablename = $1;
                    """,
                    self.table_name,
                )

                return {
                    "table_name": self.table_name,
                    "document_count": int(stats["document_count"]),
                    "storage_size": int(stats["storage_size"]),
                    "vector_dimensions": self.embedding_dimensions,
                    "earliest_document": stats["earliest_document"].isoformat() if stats["earliest_document"] else None,
                    "latest_document": stats["latest_document"].isoformat() if stats["latest_document"] else None,
                    "schema": str(table_info["schemaname"]) if table_info else None,
                    "owner": str(table_info["tableowner"]) if table_info else None,
                }

        except Exception as e:
            logger.error(f"Failed to get pgvector stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close pgvector connection pool."""
        if self.pool:
            try:
                await self.pool.close()
                logger.info("PGVector connection pool closed")
            except Exception as e:
                logger.error(f"Error closing pgvector connection pool: {e}")
            finally:
                self.pool = None
