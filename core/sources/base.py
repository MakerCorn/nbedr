"""
Base classes and interfaces for input sources.
"""

import logging
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class SourceValidationError(Exception):
    """Raised when input source validation fails."""

    pass


@dataclass
class SourceDocument:
    """Represents a document from an input source."""

    # Core identification
    name: str  # Document name/filename
    source_path: str  # Original path/URL in the source
    content_type: str  # MIME type or document type

    # Content access
    content: Optional[bytes] = None  # Raw content if loaded
    size: Optional[int] = None  # Size in bytes

    # Metadata
    last_modified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing hints
    encoding: Optional[str] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.name:
            raise ValueError("Document name cannot be empty")

        if not self.source_path:
            raise ValueError("Source path cannot be empty")

        # Try to infer content type if not provided
        if not self.content_type:
            self.content_type = self._infer_content_type()

    def _infer_content_type(self) -> str:
        """Infer content type from filename extension."""
        content_type, _ = mimetypes.guess_type(self.name)
        if content_type:
            return content_type

        # Fallback to extension-based mapping
        ext = Path(self.name).suffix.lower()
        extension_mapping = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".json": "application/json",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return extension_mapping.get(ext, "application/octet-stream")

    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.name).suffix.lower()

    def is_supported_type(self, supported_types: List[str]) -> bool:
        """Check if document type is supported."""
        return self.extension.lstrip(".") in supported_types


@dataclass
class InputSourceConfig:
    """Configuration for input sources."""

    # Source identification
    source_type: str  # 'local', 's3', 'sharepoint'
    source_uri: str  # Path, S3 URI, SharePoint URL

    # Authentication (optional)
    credentials: Dict[str, Any] = field(default_factory=dict)

    # Filtering options
    include_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(default_factory=list)
    supported_types: List[str] = field(default_factory=lambda: ["pdf", "txt", "json", "pptx"])

    # Processing options
    recursive: bool = True
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    batch_size: int = 100

    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.source_type:
            raise ValueError("source_type is required")
        if not self.source_uri:
            raise ValueError("source_uri is required")
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class BaseInputSource(ABC):
    """Abstract base class for input sources."""

    def __init__(self, config: InputSourceConfig):
        """Initialize input source with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validated = False

    @abstractmethod
    async def validate(self) -> None:
        """
        Validate the input source configuration and connectivity.

        Raises:
            SourceValidationError: If validation fails
        """
        pass

    @abstractmethod
    async def list_documents(self) -> List[SourceDocument]:
        """
        List all documents in the input source.

        Returns:
            List of SourceDocument objects (metadata only, no content)

        Raises:
            SourceValidationError: If source is not validated or accessible
        """
        pass

    @abstractmethod
    async def get_document(self, document: SourceDocument) -> SourceDocument:
        """
        Retrieve the full content of a specific document.

        Args:
            document: SourceDocument with metadata (from list_documents)

        Returns:
            SourceDocument with content loaded

        Raises:
            SourceValidationError: If document cannot be retrieved
        """
        pass

    async def get_documents(self, documents: List[SourceDocument]) -> Iterator[SourceDocument]:
        """
        Retrieve multiple documents efficiently.

        Args:
            documents: List of SourceDocument objects to retrieve

        Yields:
            SourceDocument objects with content loaded
        """
        for doc in documents:
            try:
                yield await self.get_document(doc)
            except Exception as e:
                self.logger.error(f"Failed to retrieve document {doc.name}: {e}")
                continue

    async def get_processing_preview(self) -> Dict[str, Any]:
        """
        Get a preview of what would be processed without loading content.

        Returns:
            Dictionary with processing statistics and file information
        """
        if not self._validated:
            await self.validate()

        documents = await self.list_documents()

        # Filter by supported types
        supported_docs = [doc for doc in documents if doc.is_supported_type(self.config.supported_types)]

        # Calculate statistics
        total_size = sum(doc.size or 0 for doc in supported_docs)
        type_counts = {}
        for doc in supported_docs:
            ext = doc.extension.lstrip(".")
            type_counts[ext] = type_counts.get(ext, 0) + 1

        return {
            "source_type": self.config.source_type,
            "source_uri": self.config.source_uri,
            "total_documents": len(documents),
            "supported_documents": len(supported_docs),
            "unsupported_documents": len(documents) - len(supported_docs),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "document_types": type_counts,
            "supported_types": self.config.supported_types,
        }

    def _filter_documents(self, documents: List[SourceDocument]) -> List[SourceDocument]:
        """Apply include/exclude patterns and type filtering."""
        filtered = []

        for doc in documents:
            # Check file size
            if doc.size and doc.size > self.config.max_file_size:
                self.logger.warning(f"Skipping {doc.name}: size {doc.size} exceeds limit {self.config.max_file_size}")
                continue

            # Check supported types
            if not doc.is_supported_type(self.config.supported_types):
                self.logger.debug(f"Skipping {doc.name}: unsupported type {doc.extension}")
                continue

            # Apply include/exclude patterns (basic implementation)
            # TODO: Implement proper glob pattern matching
            include_match = any(pattern in doc.source_path for pattern in self.config.include_patterns)
            exclude_match = any(pattern in doc.source_path for pattern in self.config.exclude_patterns)

            if include_match and not exclude_match:
                filtered.append(doc)
            else:
                self.logger.debug(f"Skipping {doc.name}: filtered by patterns")

        return filtered

    async def __aenter__(self):
        """Async context manager entry."""
        await self.validate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
