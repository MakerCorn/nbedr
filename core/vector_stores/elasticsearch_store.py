"""
AWS Elasticsearch (OpenSearch) vector store implementation.
"""

# mypy: disable-error-code="attr-defined,dict-item,unreachable"

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from ..models import DocumentChunk, VectorSearchResult
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore(BaseVectorStore):
    """AWS Elasticsearch (OpenSearch) implementation of vector store."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Elasticsearch vector store."""
        super().__init__(config)

        self.endpoint = config.get("aws_elasticsearch_endpoint")
        self.region = config.get("aws_elasticsearch_region", "us-east-1")
        self.index_name = config.get("aws_elasticsearch_index_name", "rag-embeddings")
        self.access_key = config.get("aws_access_key_id")
        self.secret_key = config.get("aws_secret_access_key")

        if not self.endpoint:
            raise ValueError("AWS Elasticsearch endpoint is required")

        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Setup Elasticsearch client with AWS authentication."""
        try:
            # Setup AWS authentication if credentials are provided
            if self.access_key and self.secret_key:
                # Use explicit credentials
                auth = (self.access_key, self.secret_key)
            else:
                # Use IAM role or default credentials
                auth = None

            # Create Elasticsearch client
            self.client = AsyncElasticsearch(
                hosts=[self.endpoint],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=None,  # Will use default connection
                timeout=30,
            )

        except Exception as e:
            logger.error(f"Failed to setup Elasticsearch client: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize the Elasticsearch index."""
        try:
            # Check if index exists
            if await self.client.indices.exists(index=self.index_name):
                logger.info(f"Using existing Elasticsearch index: {self.index_name}")
                return

            logger.info(f"Creating new Elasticsearch index: {self.index_name}")

            # Define index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": True},
                        "embedding_model": {"type": "keyword"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": self.config.get("embedding_dimensions", 1536),
                            "index": True,
                            "similarity": "cosine",
                        },
                        "created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                    }
                },
            }

            # Create index
            await self.client.indices.create(index=self.index_name, body=mapping)

            logger.info(f"Created Elasticsearch index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch index: {e}")
            raise

    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to Elasticsearch."""
        try:
            actions = []
            vector_ids = []

            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue

                vector_id = chunk.vector_id or chunk.id
                vector_ids.append(vector_id)

                document = {
                    "_index": self.index_name,
                    "_id": vector_id,
                    "_source": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                        "embedding_model": chunk.embedding_model,
                        "content_vector": chunk.embedding,
                        "created_at": chunk.created_at.isoformat(),
                    },
                }
                actions.append(document)

            if actions:
                # Bulk index documents
                success, failed = await async_bulk(self.client, actions, index=self.index_name, chunk_size=100)

                logger.info(f"Successfully indexed {success} documents to Elasticsearch")

                if failed:
                    logger.error(f"Failed to index {len(failed)} documents")
                    for failure in failed:
                        logger.error(f"Failed document: {failure}")

            return vector_ids

        except Exception as e:
            logger.error(f"Failed to add documents to Elasticsearch: {e}")
            raise

    async def search(
        self, query_embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Elasticsearch."""
        try:
            # Build the query
            query = {"knn": {"content_vector": {"vector": query_embedding, "k": top_k}}}

            # Add filters if provided
            if filters:
                filter_terms = []
                for key, value in filters.items():
                    filter_terms.append({"term": {key: value}})

                if filter_terms:
                    query = {"bool": {"must": [query], "filter": filter_terms}}

            # Perform search
            response = await self.client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": top_k,
                    "_source": {"excludes": ["content_vector"]},  # Exclude vector from results
                },
            )

            search_results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]

                search_result = VectorSearchResult(
                    id=hit["_id"],
                    content=source["content"],
                    source=source["source"],
                    metadata=source.get("metadata", {}),
                    similarity_score=hit["_score"],
                    embedding_model=source.get("embedding_model"),
                    created_at=source.get("created_at"),
                )
                search_results.append(search_result)

            logger.info(f"Found {len(search_results)} results in Elasticsearch")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search Elasticsearch: {e}")
            raise

    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """Delete documents from Elasticsearch."""
        try:
            actions = []
            for vector_id in vector_ids:
                actions.append({"_op_type": "delete", "_index": self.index_name, "_id": vector_id})

            if actions:
                success, failed = await async_bulk(self.client, actions, index=self.index_name)

                if failed:
                    logger.error(f"Failed to delete {len(failed)} documents")
                    return False

                logger.info(f"Deleted {success} documents from Elasticsearch")

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents from Elasticsearch: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Elasticsearch index."""
        try:
            # Get index stats
            stats = await self.client.indices.stats(index=self.index_name)
            index_stats = stats["indices"][self.index_name]

            # Get index settings and mappings
            settings = await self.client.indices.get_settings(index=self.index_name)
            mappings = await self.client.indices.get_mapping(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": index_stats["total"]["docs"]["count"],
                "storage_size": index_stats["total"]["store"]["size_in_bytes"],
                "segments": index_stats["total"]["segments"]["count"],
                "vector_dimensions": mappings[self.index_name]["mappings"]["properties"]["content_vector"].get(
                    "dims", "unknown"
                ),
                "shards": len(index_stats["shards"]),
            }

        except Exception as e:
            logger.error(f"Failed to get Elasticsearch stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Elasticsearch connection."""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed")
