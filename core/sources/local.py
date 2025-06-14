"""
Local filesystem input source implementation.
"""

import os
from pathlib import Path
from typing import List
from datetime import datetime

from .base import BaseInputSource, SourceDocument, SourceValidationError


class LocalInputSource(BaseInputSource):
    """Input source for local filesystem paths."""
    
    async def validate(self) -> None:
        """Validate local path exists and is accessible."""
        path = Path(self.config.source_uri)
        
        if not path.exists():
            raise SourceValidationError(f"Path does not exist: {path}")
        
        if not (path.is_file() or path.is_dir()):
            raise SourceValidationError(f"Path is not a file or directory: {path}")
        
        if not os.access(path, os.R_OK):
            raise SourceValidationError(f"Path is not readable: {path}")
        
        self._validated = True
        self.logger.info(f"Validated local source: {path}")
    
    async def list_documents(self) -> List[SourceDocument]:
        """List all documents in the local path."""
        if not self._validated:
            await self.validate()
        
        path = Path(self.config.source_uri)
        documents = []
        
        if path.is_file():
            # Single file
            documents.append(self._create_document_from_path(path))
        else:
            # Directory - recursively find files
            if self.config.recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    try:
                        documents.append(self._create_document_from_path(file_path))
                    except Exception as e:
                        self.logger.warning(f"Error processing {file_path}: {e}")
                        continue
        
        # Apply filtering
        filtered_documents = self._filter_documents(documents)
        
        self.logger.info(f"Found {len(documents)} total files, {len(filtered_documents)} after filtering")
        return filtered_documents
    
    async def get_document(self, document: SourceDocument) -> SourceDocument:
        """Retrieve document content from local filesystem."""
        file_path = Path(document.source_path)
        
        if not file_path.exists():
            raise SourceValidationError(f"Document no longer exists: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Update document with content
            document.content = content
            document.size = len(content)
            
            return document
            
        except Exception as e:
            raise SourceValidationError(f"Failed to read document {file_path}: {e}")
    
    def _create_document_from_path(self, file_path: Path) -> SourceDocument:
        """Create a SourceDocument from a local file path."""
        try:
            stat = file_path.stat()
            
            return SourceDocument(
                name=file_path.name,
                source_path=str(file_path.absolute()),
                content_type="",  # Will be inferred in __post_init__
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                metadata={
                    'full_path': str(file_path.absolute()),
                    'relative_path': str(file_path.relative_to(Path(self.config.source_uri).parent)),
                    'permissions': oct(stat.st_mode)[-3:],
                }
            )
        except Exception as e:
            raise SourceValidationError(f"Failed to get file info for {file_path}: {e}")