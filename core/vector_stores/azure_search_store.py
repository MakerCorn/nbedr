"""
Azure AI Search vector store implementation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ..models import DocumentChunk, VectorSearchResult
from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class AzureAISearchVectorStore(BaseVectorStore):
    """Azure AI Search implementation of vector store."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure AI Search vector store."""
        super().__init__(config)

        self.service_name = config.get("azure_search_service_name")
        self.api_key = config.get("azure_search_api_key")
        self.index_name = config.get("azure_search_index_name", "rag-embeddings")
        self.api_version = config.get("azure_search_api_version", "2023-11-01")

        if not self.service_name or not self.api_key:
            raise ValueError("Azure Search service name and API key are required")

        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.credential = AzureKeyCredential(self.api_key)

        # Initialize clients
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.search_client = SearchClient(
            endpoint=self.endpoint, index_name=self.index_name, credential=self.credential
        )

    async def initialize(self) -> None:
        """Initialize the Azure AI Search index."""
        try:
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                logger.info(f"Using existing Azure AI Search index: {self.index_name}")
                return
            except Exception:
                logger.info(f"Creating new Azure AI Search index: {self.index_name}")

            # Define the search index schema
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True, searchable=False, filterable=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True, filterable=False),
                SearchField(
                    name="source", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True
                ),
                SearchField(name="metadata", type=SearchFieldDataType.String, searchable=False, filterable=True),
                SearchField(
                    name="embedding_model",
                    type=SearchFieldDataType.String,
                    searchable=False,
                    filterable=True,
                    facetable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.config.get("embedding_dimensions", 1536),
                    vector_search_profile_name="content-vector-profile",
                ),
                SearchField(
                    name="created_at",
                    type=SearchFieldDataType.DateTimeOffset,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                ),
            ]

            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="content-vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="content-vector-profile", algorithm_configuration_name="content-vector-config"
                    )
                ],
            )

            # Create the search index
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)

            self.index_client.create_index(index)
            logger.info(f"Created Azure AI Search index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Azure AI Search index: {e}")
            raise

    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to Azure AI Search."""
        try:
            documents = []
            vector_ids = []

            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue

                vector_id = chunk.vector_id or chunk.id
                vector_ids.append(vector_id)

                document = {
                    "id": vector_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "metadata": json.dumps(chunk.metadata),
                    "embedding_model": chunk.embedding_model,
                    "content_vector": chunk.embedding,
                    "created_at": chunk.created_at.isoformat(),
                }
                documents.append(document)

            if documents:
                result = self.search_client.upload_documents(documents)
                logger.info(f"Uploaded {len(documents)} documents to Azure AI Search")

                # Check for any failures
                failed_docs = [r for r in result if not r.succeeded]
                if failed_docs:
                    logger.error(f"Failed to upload {len(failed_docs)} documents")
                    for failed_doc in failed_docs:
                        logger.error(f"Failed document: {failed_doc.key}, Error: {failed_doc.error_message}")

            return vector_ids

        except Exception as e:
            logger.error(f"Failed to add documents to Azure AI Search: {e}")
            raise

    async def search(
        self, query_embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Azure AI Search."""
        try:
            # Create vector query
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="content_vector")

            # Build filter string if provided
            filter_expr = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} eq '{value}'")
                    else:
                        filter_parts.append(f"{key} eq {value}")
                filter_expr = " and ".join(filter_parts)

            # Perform search
            results = self.search_client.search(
                search_text="*",
                vector_queries=[vector_query],
                filter=filter_expr,
                top=top_k,
                select=["id", "content", "source", "metadata", "embedding_model", "created_at"],
            )

            search_results = []
            for result in results:
                # Parse metadata back from JSON
                metadata = {}
                if result.get("metadata"):
                    try:
                        metadata = json.loads(result["metadata"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata for document {result['id']}")

                search_result = VectorSearchResult(
                    id=result["id"],
                    content=result["content"],
                    source=result["source"],
                    metadata=metadata,
                    similarity_score=result.get("@search.score", 0.0),
                    embedding_model=result.get("embedding_model"),
                    created_at=result.get("created_at"),
                )
                search_results.append(search_result)

            logger.info(f"Found {len(search_results)} results in Azure AI Search")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search Azure AI Search: {e}")
            raise

    async def delete_documents(self, vector_ids: List[str]) -> bool:
        """Delete documents from Azure AI Search."""
        try:
            documents = [{"id": vector_id} for vector_id in vector_ids]
            result = self.search_client.delete_documents(documents)

            # Check for failures
            failed_docs = [r for r in result if not r.succeeded]
            if failed_docs:
                logger.error(f"Failed to delete {len(failed_docs)} documents")
                return False

            logger.info(f"Deleted {len(vector_ids)} documents from Azure AI Search")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents from Azure AI Search: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Azure AI Search index."""
        try:
            # Get index statistics
            index_stats = self.index_client.get_index_statistics(self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": index_stats.document_count,
                "storage_size": index_stats.storage_size,
                "vector_index_size": getattr(index_stats, "vector_index_size", "N/A"),
            }

        except Exception as e:
            logger.error(f"Failed to get Azure AI Search stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Azure AI Search connections."""
        # Azure SDK clients don't require explicit closing
        logger.info("Azure AI Search connections closed")
