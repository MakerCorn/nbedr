"""
PostgreSQL with pgvector extension vector store implementation.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
import asyncpg
from datetime import datetime

from .base import BaseVectorStore
from ..models import DocumentChunk, VectorSearchResult

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector extension implementation of vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PGVector store."""
        super().__init__(config)
        
        self.host = config.get('pgvector_host', 'localhost')
        self.port = config.get('pgvector_port', 5432)
        self.database = config.get('pgvector_database', 'vectordb')
        self.user = config.get('pgvector_user', 'postgres')
        self.password = config.get('pgvector_password')
        self.table_name = config.get('pgvector_table_name', 'rag_embeddings')
        self.embedding_dimensions = config.get('embedding_dimensions', 1536)
        
        if not self.password:
            raise ValueError("PostgreSQL password is required for pgvector")
        
        self.pool = None
    
    async def _get_connection_pool(self):
        """Get or create connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
        return self.pool
    
    async def initialize(self) -> None:
        """Initialize the pgvector table and extension."""
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
                    self.table_name
                )
                
                if table_exists:
                    logger.info(f"Using existing pgvector table: {self.table_name}")
                    return
                
                logger.info(f"Creating new pgvector table: {self.table_name}")
                
                # Create table with vector column
                await conn.execute(f"""
                    CREATE TABLE {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        embedding_model TEXT,
                        content_vector vector({self.embedding_dimensions}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                
                # Create vector similarity index
                await conn.execute(f"""
                    CREATE INDEX ON {self.table_name} 
                    USING ivfflat (content_vector vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                # Create additional indexes
                await conn.execute(f"""
                    CREATE INDEX idx_{self.table_name}_source ON {self.table_name}(source);
                """)
                
                await conn.execute(f"""
                    CREATE INDEX idx_{self.table_name}_metadata ON {self.table_name} USING GIN(metadata);
                """)
                
                logger.info(f"Created pgvector table: {self.table_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize pgvector table: {e}")
            raise
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to pgvector."""
        try:
            # Apply rate limiting for vector store operations
            await self._apply_rate_limiting("add_documents")
            
            start_time = time.time()
            
            pool = await self._get_connection_pool()
            vector_ids = []
            
            async with pool.acquire() as conn:
                for chunk in chunks:
                    if not chunk.embedding:
                        logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                        continue
                    
                    vector_id = chunk.vector_id or chunk.id
                    vector_ids.append(vector_id)
                    
                    # Insert document
                    await conn.execute(f"""
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
                    """, 
                    vector_id,
                    chunk.content,
                    chunk.source,
                    json.dumps(chunk.metadata) if chunk.metadata else "{}",
                    chunk.embedding_model,
                    chunk.embedding,
                    chunk.created_at
                    )
                
                logger.info(f"Successfully added {len(vector_ids)} documents to pgvector")
            
            # Record operation response time
            response_time = time.time() - start_time
            self._record_operation_response(response_time, "add_documents")
            
            return vector_ids
            
        except Exception as e:
            self._record_operation_error("add_documents_error")
            logger.error(f"Failed to add documents to pgvector: {e}")
            raise
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in pgvector."""
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
            
            query = f"""
                SELECT id, content, source, metadata, embedding_model, created_at,
                       1 - (content_vector <=> $1) as similarity_score
                FROM {self.table_name}
                {where_clause}
                ORDER BY content_vector <=> $1
                LIMIT {top_k};
            """
            
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, query_embedding, *filter_params)
                
                search_results = []
                for row in rows:
                    search_result = VectorSearchResult(
                        id=row['id'],
                        content=row['content'],
                        source=row['source'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        similarity_score=float(row['similarity_score']),
                        embedding_model=row['embedding_model'],
                        created_at=row['created_at'].isoformat() if row['created_at'] else None
                    )
                    search_results.append(search_result)
                
                # Record operation response time
                response_time = time.time() - start_time
                self._record_operation_response(response_time, "search")
                
                logger.info(f"Found {len(search_results)} results in pgvector")
                return search_results
                
        except Exception as e:
            self._record_operation_error("search_error")
            logger.error(f"Failed to search pgvector: {e}")
            raise
    
    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """Delete documents from pgvector."""
        try:
            pool = await self._get_connection_pool()
            
            async with pool.acquire() as conn:
                # Delete documents
                result = await conn.execute(f"""
                    DELETE FROM {self.table_name} 
                    WHERE id = ANY($1);
                """, vector_ids)
                
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                logger.info(f"Deleted {deleted_count} documents from pgvector")
                
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete documents from pgvector: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pgvector table."""
        try:
            pool = await self._get_connection_pool()
            
            async with pool.acquire() as conn:
                # Get table stats
                stats = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as document_count,
                        pg_total_relation_size('{self.table_name}') as storage_size,
                        MIN(created_at) as earliest_document,
                        MAX(created_at) as latest_document
                    FROM {self.table_name};
                """)
                
                # Get table info
                table_info = await conn.fetchrow(f"""
                    SELECT 
                        schemaname,
                        tablename,
                        tableowner
                    FROM pg_tables 
                    WHERE tablename = '{self.table_name}';
                """)
                
                return {
                    "table_name": self.table_name,
                    "document_count": stats['document_count'],
                    "storage_size": stats['storage_size'],
                    "vector_dimensions": self.embedding_dimensions,
                    "earliest_document": stats['earliest_document'].isoformat() if stats['earliest_document'] else None,
                    "latest_document": stats['latest_document'].isoformat() if stats['latest_document'] else None,
                    "schema": table_info['schemaname'] if table_info else None,
                    "owner": table_info['tableowner'] if table_info else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get pgvector stats: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close pgvector connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PGVector connection pool closed")