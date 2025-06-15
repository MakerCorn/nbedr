"""
Factory for creating input source instances.
"""

from typing import Any, Dict, List, Optional, Type

from .base import BaseInputSource, InputSourceConfig, SourceValidationError
from .local import LocalInputSource
from .s3 import S3InputSource
from .sharepoint import SharePointInputSource


class InputSourceFactory:
    """Factory for creating input source instances."""

    # Registry of available source types
    _source_types: Dict[str, Type[BaseInputSource]] = {
        "local": LocalInputSource,
        "s3": S3InputSource,
        "sharepoint": SharePointInputSource,
    }

    @classmethod
    def create_source(cls, config: InputSourceConfig) -> BaseInputSource:
        """
        Create an input source instance based on configuration.

        Args:
            config: Input source configuration

        Returns:
            Configured input source instance

        Raises:
            SourceValidationError: If source type is not supported
        """
        source_type = config.source_type.lower()

        if source_type not in cls._source_types:
            available = list(cls._source_types.keys())
            raise SourceValidationError(f"Unsupported source type '{source_type}'. Available types: {available}")

        source_class = cls._source_types[source_type]
        return source_class(config)

    @classmethod
    def create_from_uri(cls, source_uri: str, source_type: Optional[str] = None, **kwargs) -> BaseInputSource:
        """
        Create an input source from a URI, auto-detecting type if not specified.

        Args:
            source_uri: URI of the input source
            source_type: Explicit source type (optional)
            **kwargs: Additional configuration options

        Returns:
            Configured input source instance
        """
        # Auto-detect source type if not provided
        if not source_type:
            source_type = cls._detect_source_type(source_uri)

        # Create configuration
        config = InputSourceConfig(source_type=source_type, source_uri=source_uri, **kwargs)

        return cls.create_source(config)

    @classmethod
    def _detect_source_type(cls, uri: str) -> str:
        """
        Auto-detect source type from URI.

        Args:
            uri: Source URI

        Returns:
            Detected source type
        """
        uri_lower = uri.lower()

        # S3 detection
        if uri_lower.startswith("s3://") or (
            "." in uri and any(s3_indicator in uri_lower for s3_indicator in ["amazonaws.com", "s3.", ".s3-"])
        ):
            return "s3"

        # SharePoint detection
        if "sharepoint.com" in uri_lower:
            return "sharepoint"

        # HTTP/HTTPS URLs (could be SharePoint or other web sources)
        if uri_lower.startswith(("http://", "https://")):
            if "sharepoint" in uri_lower:
                return "sharepoint"
            # Default web sources to local for now
            # Could be extended to support generic HTTP sources
            raise SourceValidationError(
                f"Cannot auto-detect source type for URL: {uri}. " "Please specify source_type explicitly."
            )

        # Default to local filesystem
        return "local"

    @classmethod
    def register_source_type(cls, name: str, source_class: Type[BaseInputSource]) -> None:
        """
        Register a new source type.

        Args:
            name: Source type name
            source_class: Source implementation class
        """
        if not issubclass(source_class, BaseInputSource):
            raise ValueError("Source class must inherit from BaseInputSource")

        cls._source_types[name.lower()] = source_class

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported source types."""
        return list(cls._source_types.keys())

    @classmethod
    def create_local_source(cls, path: str, **kwargs) -> LocalInputSource:
        """Convenience method to create a local source."""
        config = InputSourceConfig(source_type="local", source_uri=path, **kwargs)
        return LocalInputSource(config)

    @classmethod
    def create_s3_source(
        cls, bucket: str, prefix: str = "", credentials: Optional[Dict[str, Any]] = None, **kwargs
    ) -> S3InputSource:
        """Convenience method to create an S3 source."""
        if prefix:
            source_uri = f"s3://{bucket}/{prefix}"
        else:
            source_uri = f"s3://{bucket}"

        config = InputSourceConfig(source_type="s3", source_uri=source_uri, credentials=credentials or {}, **kwargs)
        return S3InputSource(config)

    @classmethod
    def create_sharepoint_source(
        cls,
        site_url: str,
        library_path: str = "Shared Documents",
        credentials: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SharePointInputSource:
        """Convenience method to create a SharePoint source."""
        if not site_url.endswith("/"):
            site_url += "/"

        source_uri = f"{site_url}{library_path}"

        config = InputSourceConfig(
            source_type="sharepoint", source_uri=source_uri, credentials=credentials or {}, **kwargs
        )
        return SharePointInputSource(config)
