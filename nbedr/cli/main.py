"""
CLI interface for RAG embedding database application.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Check Python version early
if sys.version_info < (3, 11):
    print("❌ Error: nBedR requires Python 3.11 or higher.")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("Please upgrade your Python installation.")
    sys.exit(1)

from nbedr.core.config import EmbeddingConfig, get_config
from nbedr.core.models import DocumentChunk
from nbedr.core.services.document_service import DocumentService


# Basic logging setup
def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG Embedding Database - Document embedding and vector storage system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings from local PDF documents
  %(prog)s create-embeddings --datapath ./documents --doctype pdf

  # Search for similar documents
  %(prog)s search --query "machine learning algorithms" --top-k 5

  # List all available sources
  %(prog)s list-sources

  # Check system status
  %(prog)s status

  # Process from S3 bucket
  %(prog)s create-embeddings --source-type s3 --source-uri s3://my-bucket/docs/

  # Use custom embedding model and vector database
  %(prog)s create-embeddings --datapath ./docs --embedding-model text-embedding-3-large --vector-db-type pinecone
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create embeddings command
    create_parser = subparsers.add_parser("create-embeddings", help="Create embeddings from documents")
    add_create_embedding_args(create_parser)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar documents")
    add_search_args(search_parser)

    # List sources command
    list_parser = subparsers.add_parser("list-sources", help="List available document sources")
    add_list_sources_args(list_parser)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status and configuration")
    add_status_args(status_parser)

    # Common arguments for all commands
    for subparser in [create_parser, search_parser, list_parser, status_parser]:
        add_common_args(subparser)

    return parser


def add_create_embedding_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for create-embeddings command."""
    # I/O Arguments
    parser.add_argument("--datapath", type=Path, help="Path to the input document or directory (for local sources)")
    parser.add_argument(
        "--output",
        type=str,
        default="./embeddings_output",
        help="Path to save the generated embeddings (default: ./embeddings_output)",
    )

    # Input Source Arguments
    parser.add_argument(
        "--source-type",
        type=str,
        default="local",
        choices=["local", "s3", "sharepoint"],
        help="Type of input source (default: local)",
    )
    parser.add_argument(
        "--source-uri", type=str, help="URI for the input source (S3 bucket, SharePoint URL, or local path)"
    )
    parser.add_argument(
        "--source-credentials", type=str, help="JSON string containing credentials for the input source"
    )
    parser.add_argument(
        "--source-include-patterns", type=str, help="JSON array of glob patterns to include (default: ['**/*'])"
    )
    parser.add_argument("--source-exclude-patterns", type=str, help="JSON array of glob patterns to exclude")
    parser.add_argument(
        "--source-max-file-size", type=int, default=50 * 1024 * 1024, help="Maximum file size in bytes (default: 50MB)"
    )
    parser.add_argument(
        "--source-batch-size", type=int, default=100, help="Batch size for processing source files (default: 100)"
    )

    # Document Processing Arguments
    parser.add_argument("--chunk-size", type=int, default=512, help="Size of each chunk in tokens (default: 512)")
    parser.add_argument(
        "--doctype",
        type=str,
        default="pdf",
        choices=["pdf", "txt", "json", "api", "pptx"],
        help="Type of the input document (default: pdf)",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="semantic",
        choices=["semantic", "fixed", "sentence"],
        help="Chunking algorithm to use (default: semantic)",
    )
    parser.add_argument("--chunking-params", type=str, help="JSON string of extra chunker parameters")

    # Embedding Configuration
    parser.add_argument("--openai-key", type=str, help="OpenAI API key (can also use OPENAI_API_KEY env var)")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=1536, help="Embedding dimensions (default: 1536)")
    parser.add_argument(
        "--batch-size-embeddings", type=int, default=100, help="Batch size for embedding generation (default: 100)"
    )

    # Vector Database Configuration
    parser.add_argument(
        "--vector-db-type",
        type=str,
        default="faiss",
        choices=["faiss", "pinecone", "chroma"],
        help="Vector database type (default: faiss)",
    )
    parser.add_argument("--vector-db-config", type=str, help="JSON string of vector database configuration")

    # Pinecone specific
    parser.add_argument("--pinecone-api-key", type=str, help="Pinecone API key")
    parser.add_argument("--pinecone-environment", type=str, help="Pinecone environment")
    parser.add_argument(
        "--pinecone-index-name",
        type=str,
        default="rag-embeddings",
        help="Pinecone index name (default: rag-embeddings)",
    )

    # Chroma specific
    parser.add_argument("--chroma-host", type=str, default="localhost", help="Chroma host (default: localhost)")
    parser.add_argument("--chroma-port", type=int, default=8000, help="Chroma port (default: 8000)")
    parser.add_argument(
        "--chroma-collection-name",
        type=str,
        default="rag-embeddings",
        help="Chroma collection name (default: rag-embeddings)",
    )

    # FAISS specific
    parser.add_argument(
        "--faiss-index-path", type=str, default="./faiss_index", help="FAISS index path (default: ./faiss_index)"
    )
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default="IndexFlatIP",
        choices=["IndexFlatIP", "IndexIVFFlat", "IndexHNSW"],
        help="FAISS index type (default: IndexFlatIP)",
    )

    # Performance Arguments
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads for processing (default: 1)")
    parser.add_argument(
        "--embed-workers", type=int, default=1, help="Number of worker threads for embedding generation (default: 1)"
    )
    parser.add_argument(
        "--pace", action="store_true", default=True, help="Pace API calls to stay within rate limits (default: True)"
    )

    # Rate Limiting Arguments
    parser.add_argument("--rate-limit", action="store_true", help="Enable rate limiting for API requests")
    parser.add_argument(
        "--rate-limit-strategy",
        type=str,
        choices=["fixed_window", "sliding_window", "token_bucket", "adaptive"],
        default="sliding_window",
        help="Rate limiting strategy (default: sliding_window)",
    )
    parser.add_argument(
        "--rate-limit-preset",
        type=str,
        choices=[
            "openai_gpt4",
            "openai_gpt35_turbo",
            "azure_openai_standard",
            "anthropic_claude",
            "conservative",
            "aggressive",
        ],
        help="Use a preset rate limit configuration",
    )
    parser.add_argument("--rate-limit-requests-per-minute", type=int, help="Maximum requests per minute")
    parser.add_argument("--rate-limit-tokens-per-minute", type=int, help="Maximum tokens per minute")
    parser.add_argument("--rate-limit-max-burst", type=int, help="Maximum burst requests allowed")
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries on rate limit errors (default: 3)",
    )

    # Instance Coordination Arguments
    parser.add_argument(
        "--disable-coordination", action="store_true", help="Disable instance coordination for parallel execution"
    )
    parser.add_argument("--instance-id", type=str, help="Custom instance ID (auto-generated if not provided)")
    parser.add_argument("--list-instances", action="store_true", help="List all active instances and exit")


def add_search_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for search command."""
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold (default: 0.0)")
    parser.add_argument(
        "--vector-db-type",
        type=str,
        default="faiss",
        choices=["faiss", "pinecone", "chroma"],
        help="Vector database type (default: faiss)",
    )
    parser.add_argument("--index-path", type=str, help="Path to vector index (for FAISS)")


def add_list_sources_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for list-sources command."""
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["local", "s3", "sharepoint", "all"],
        default="all",
        help="Filter by source type (default: all)",
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed information about each source")


def add_status_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for status command."""
    parser.add_argument("--check-connections", action="store_true", help="Check connections to external services")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to all subcommands."""
    # Azure Arguments
    parser.add_argument(
        "--use-azure-identity", action="store_true", help="Use Azure Default Credentials for authentication"
    )

    # Utility Arguments
    parser.add_argument("--preview", action="store_true", help="Show processing preview without running")
    parser.add_argument("--validate", action="store_true", help="Validate configuration and inputs only")
    parser.add_argument("--env-file", type=str, help="Path to .env file for configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Document coordination arguments
    parser.add_argument(
        "--disable-coordination",
        action="store_true",
        help="Disable document coordination (not recommended for multi-instance)",
    )
    parser.add_argument("--list-instances", action="store_true", help="List active instances and exit")
    parser.add_argument("--cleanup-locks", action="store_true", help="Clean up stale document locks and exit")
    parser.add_argument("--reset-failed", action="store_true", help="Reset failed documents for reprocessing and exit")


def override_config_from_args(config: EmbeddingConfig, args: argparse.Namespace) -> EmbeddingConfig:
    """Override configuration with command line arguments."""

    # Only process arguments that exist in the args namespace
    arg_dict = vars(args)

    # Input source configuration
    if "source_type" in arg_dict and arg_dict["source_type"] and arg_dict["source_type"] != "local":
        config.source_type = arg_dict["source_type"]
    if "source_uri" in arg_dict and arg_dict["source_uri"]:
        config.source_uri = arg_dict["source_uri"]
    if "source_credentials" in arg_dict and arg_dict["source_credentials"]:
        try:
            config.source_credentials = json.loads(arg_dict["source_credentials"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in source-credentials: {e}")
            sys.exit(1)
    if "source_include_patterns" in arg_dict and arg_dict["source_include_patterns"]:
        try:
            config.source_include_patterns = json.loads(arg_dict["source_include_patterns"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in source-include-patterns: {e}")
            sys.exit(1)
    if "source_exclude_patterns" in arg_dict and arg_dict["source_exclude_patterns"]:
        try:
            config.source_exclude_patterns = json.loads(arg_dict["source_exclude_patterns"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in source-exclude-patterns: {e}")
            sys.exit(1)
    if "source_max_file_size" in arg_dict and arg_dict["source_max_file_size"] != 50 * 1024 * 1024:
        config.source_max_file_size = arg_dict["source_max_file_size"]
    if "source_batch_size" in arg_dict and arg_dict["source_batch_size"] != 100:
        config.source_batch_size = arg_dict["source_batch_size"]

    # Legacy datapath handling - if provided and no source_uri, use it
    if "datapath" in arg_dict and arg_dict["datapath"] and not config.source_uri:
        if config.source_type == "local":
            config.datapath = arg_dict["datapath"]
            config.source_uri = str(arg_dict["datapath"])
        else:
            config.datapath = arg_dict["datapath"]

    # Output configuration
    if "output" in arg_dict and arg_dict["output"] and arg_dict["output"] != "./embeddings_output":
        config.output = arg_dict["output"]

    # Document processing
    if "chunk_size" in arg_dict and arg_dict["chunk_size"] != 512:
        config.chunk_size = arg_dict["chunk_size"]
    if "doctype" in arg_dict and arg_dict["doctype"] != "pdf":
        config.doctype = arg_dict["doctype"]
    if "chunking_strategy" in arg_dict and arg_dict["chunking_strategy"] != "semantic":
        config.chunking_strategy = arg_dict["chunking_strategy"]
    if "chunking_params" in arg_dict and arg_dict["chunking_params"]:
        try:
            config.chunking_params = json.loads(arg_dict["chunking_params"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid chunking params JSON: {e}")
            sys.exit(1)

    # Embedding configuration
    if "openai_key" in arg_dict and arg_dict["openai_key"]:
        config.openai_api_key = arg_dict["openai_key"]
    if "embedding_model" in arg_dict and arg_dict["embedding_model"] != "text-embedding-3-small":
        config.embedding_model = arg_dict["embedding_model"]
    if "embedding_dimensions" in arg_dict and arg_dict["embedding_dimensions"] != 1536:
        config.embedding_dimensions = arg_dict["embedding_dimensions"]
    if "batch_size_embeddings" in arg_dict and arg_dict["batch_size_embeddings"] != 100:
        config.batch_size_embeddings = arg_dict["batch_size_embeddings"]

    # Vector database configuration
    if "vector_db_type" in arg_dict and arg_dict["vector_db_type"] != "faiss":
        config.vector_db_type = arg_dict["vector_db_type"]
    if "vector_db_config" in arg_dict and arg_dict["vector_db_config"]:
        try:
            config.vector_db_config = json.loads(arg_dict["vector_db_config"])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid vector DB config JSON: {e}")
            sys.exit(1)

    # Pinecone configuration
    if "pinecone_api_key" in arg_dict and arg_dict["pinecone_api_key"]:
        config.pinecone_api_key = arg_dict["pinecone_api_key"]
    if "pinecone_environment" in arg_dict and arg_dict["pinecone_environment"]:
        config.pinecone_environment = arg_dict["pinecone_environment"]
    if "pinecone_index_name" in arg_dict and arg_dict["pinecone_index_name"] != "rag-embeddings":
        config.pinecone_index_name = arg_dict["pinecone_index_name"]

    # Chroma configuration
    if "chroma_host" in arg_dict and arg_dict["chroma_host"] != "localhost":
        config.chroma_host = arg_dict["chroma_host"]
    if "chroma_port" in arg_dict and arg_dict["chroma_port"] != 8000:
        config.chroma_port = arg_dict["chroma_port"]
    if "chroma_collection_name" in arg_dict and arg_dict["chroma_collection_name"] != "rag-embeddings":
        config.chroma_collection_name = arg_dict["chroma_collection_name"]

    # FAISS configuration
    if "faiss_index_path" in arg_dict and arg_dict["faiss_index_path"] != "./faiss_index":
        config.faiss_index_path = arg_dict["faiss_index_path"]
    if "faiss_index_type" in arg_dict and arg_dict["faiss_index_type"] != "IndexFlatIP":
        config.faiss_index_type = arg_dict["faiss_index_type"]

    # Azure configuration
    if "use_azure_identity" in arg_dict and arg_dict["use_azure_identity"]:
        config.use_azure_identity = arg_dict["use_azure_identity"]

    # Performance configuration
    if "workers" in arg_dict and arg_dict["workers"] != 1:
        config.workers = arg_dict["workers"]
    if "embed_workers" in arg_dict and arg_dict["embed_workers"] != 1:
        config.embed_workers = arg_dict["embed_workers"]
    if "pace" in arg_dict and not arg_dict["pace"]:  # Only if explicitly disabled
        config.pace = arg_dict["pace"]

    # Rate limiting arguments
    if "rate_limit" in arg_dict and arg_dict["rate_limit"]:
        config.rate_limit_enabled = arg_dict["rate_limit"]
    if "rate_limit_strategy" in arg_dict and arg_dict["rate_limit_strategy"] != "sliding_window":
        config.rate_limit_strategy = arg_dict["rate_limit_strategy"]
    if "rate_limit_preset" in arg_dict and arg_dict["rate_limit_preset"]:
        config.rate_limit_preset = arg_dict["rate_limit_preset"]
    if "rate_limit_requests_per_minute" in arg_dict and arg_dict["rate_limit_requests_per_minute"]:
        config.rate_limit_requests_per_minute = arg_dict["rate_limit_requests_per_minute"]
    if "rate_limit_tokens_per_minute" in arg_dict and arg_dict["rate_limit_tokens_per_minute"]:
        config.rate_limit_tokens_per_minute = arg_dict["rate_limit_tokens_per_minute"]
    if "rate_limit_max_burst" in arg_dict and arg_dict["rate_limit_max_burst"]:
        config.rate_limit_max_burst = arg_dict["rate_limit_max_burst"]
    if "rate_limit_max_retries" in arg_dict and arg_dict["rate_limit_max_retries"] != 3:
        config.rate_limit_max_retries = arg_dict["rate_limit_max_retries"]

    return config


def show_create_embeddings_preview(service: DocumentService, config: EmbeddingConfig) -> None:
    """Show processing preview for create-embeddings command."""
    logger.info(f"Generating preview for {config.source_type} source")

    try:
        print("\n" + "=" * 60)
        print("EMBEDDING CREATION PREVIEW")
        print("=" * 60)

        print(f"Source Type: {config.source_type.title()}")
        if config.source_type == "local":
            print(f"Input Path: {config.datapath}")
        else:
            print(f"Source URI: {config.source_uri}")

        print(f"Document Type: {config.doctype}")
        print(f"Chunking Strategy: {config.chunking_strategy}")
        print(f"Chunk Size: {config.chunk_size}")
        print(f"Embedding Model: {config.embedding_model}")
        print(f"Vector Database: {config.vector_db_type}")
        print(f"Output Path: {config.output}")

        if config.source_type == "local" and config.datapath:
            if config.datapath.is_dir():
                files = list(config.datapath.rglob(f"**/*.{config.doctype}"))
                print(f"Files to Process: {len(files)}")
                if files and len(files) <= 5:
                    for file_path in files:
                        print(f"  - {file_path}")
                elif files:
                    for file_path in files[:3]:
                        print(f"  - {file_path}")
                    print(f"  ... and {len(files) - 3} more files")
            else:
                print(f"Single file: {config.datapath}")

        print("\nUse --validate to check configuration or run without --preview to start processing.")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error generating preview: {e}", exc_info=True)
        sys.exit(1)


def validate_create_embeddings(service: DocumentService, config: EmbeddingConfig) -> None:
    """Validate configuration and inputs for create-embeddings command."""
    logger.info(f"Validating {config.source_type} input source")

    try:
        # Validate configuration
        config.validate()

        # Additional validation for source accessibility
        if config.source_type == "local":
            if not config.datapath.exists():
                raise ValueError(f"Input path does not exist: {config.datapath}")

        print("\n✅ Configuration and inputs are valid!")
        if config.source_type == "local":
            print(f"Ready to process: {config.datapath}")
        else:
            print(f"Ready to process: {config.source_uri}")
        print(f"Source type: {config.source_type}")
        print(f"Output will be saved to: {config.output}")
        print(f"Document type: {config.doctype}")
        print(f"Vector database type: {config.vector_db_type}")

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


def handle_list_instances(args: argparse.Namespace) -> None:
    """Handle list-instances command."""
    from nbedr.core.utils.instance_coordinator import create_instance_coordinator

    try:
        # Create a temporary coordinator to access instance registry
        config = get_config(args.env_file)
        coordinator = create_instance_coordinator(config)

        active_instances = coordinator.get_active_instances()

        print("\n📋 Active nBedR Instances")
        print("=" * 50)

        if not active_instances:
            print("No active instances found.")
        else:
            for instance in active_instances:
                print(f"Instance ID: {instance.instance_id}")
                print(f"  Process ID: {instance.process_id}")
                print(f"  Started: {instance.started_at}")
                print(f"  Output Path: {instance.output_path}")
                print(f"  Vector DB Path: {instance.vector_db_path or 'N/A'}")
                print(f"  Status: {instance.status}")
                print(f"  Config Hash: {instance.config_hash}")
                print()

        print(f"Total Active Instances: {len(active_instances)}")

    except Exception as e:
        logger.error(f"Failed to list instances: {e}")
        sys.exit(1)


def handle_cleanup_locks(args: argparse.Namespace) -> None:
    """Handle cleanup-locks command."""
    from nbedr.core.utils.document_coordinator import DocumentCoordinator
    from nbedr.core.utils.instance_coordinator import create_instance_coordinator

    try:
        config = get_config(args.env_file)
        coordinator = create_instance_coordinator(config)

        if coordinator.register_instance():
            doc_coordinator = DocumentCoordinator(
                coordination_dir=coordinator.coordination_dir, instance_id=coordinator.instance_id
            )

            print("\n🧹 Cleaning up stale document locks...")
            doc_coordinator.cleanup_stale_locks()
            print("✅ Stale locks cleaned up successfully")

            coordinator.unregister_instance()
        else:
            print("❌ Failed to register instance for cleanup")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to cleanup locks: {e}")
        sys.exit(1)


def handle_reset_failed(args: argparse.Namespace) -> None:
    """Handle reset-failed command."""
    from nbedr.core.utils.document_coordinator import DocumentCoordinator
    from nbedr.core.utils.instance_coordinator import create_instance_coordinator

    try:
        config = get_config(args.env_file)
        coordinator = create_instance_coordinator(config)

        if coordinator.register_instance():
            doc_coordinator = DocumentCoordinator(
                coordination_dir=coordinator.coordination_dir, instance_id=coordinator.instance_id
            )

            print("\n🔄 Resetting failed documents for reprocessing...")
            doc_coordinator.reset_failed_files()
            print("✅ Failed documents reset successfully")

            coordinator.unregister_instance()
        else:
            print("❌ Failed to register instance for reset")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to reset failed documents: {e}")
        sys.exit(1)


def handle_create_embeddings(args: argparse.Namespace) -> None:
    """Handle create-embeddings command."""
    logger.info("Loading configuration for embedding creation")
    config = get_config(args.env_file)

    # Override with command line arguments
    config = override_config_from_args(config, args)

    # Handle coordination commands
    if hasattr(args, "list_instances") and args.list_instances:
        handle_list_instances(args)
        return

    if hasattr(args, "cleanup_locks") and args.cleanup_locks:
        handle_cleanup_locks(args)
        return

    if hasattr(args, "reset_failed") and args.reset_failed:
        handle_reset_failed(args)
        return

    # Validate required arguments
    if config.source_type == "local":
        if not config.datapath and not args.datapath:
            print("Error: --datapath is required for local source type")
            sys.exit(1)
    else:
        if not config.source_uri:
            print(f"Error: --source-uri is required for {config.source_type} source type")
            sys.exit(1)

    # Create document service with coordination
    logger.info("Initializing document service")
    enable_coordination = not (hasattr(args, "disable_coordination") and args.disable_coordination)
    service = DocumentService(config, enable_coordination=enable_coordination)

    # Handle special modes
    if args.preview:
        logger.info("Generating processing preview")
        show_create_embeddings_preview(service, config)
        return

    if args.validate:
        logger.info("Validating configuration and inputs")
        validate_create_embeddings(service, config)
        return

    # Normal processing
    logger.info("Starting embedding creation")

    input_path = config.source_uri if config.source_type != "local" else str(config.datapath)

    logger.info(f"Source type: {config.source_type}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {config.output}")
    logger.info(f"Document type: {config.doctype}")
    logger.info(f"Chunking strategy: {config.chunking_strategy}")
    logger.info(f"Embedding model: {config.embedding_model}")
    logger.info(f"Vector database: {config.vector_db_type}")

    start_time = time.time()

    try:
        # Process documents
        logger.info("Processing documents and creating chunks")
        service.update_heartbeat()  # Update heartbeat
        chunks = service.process_documents(
            config.datapath if config.source_type == "local" else Path(config.source_uri or ".")
        )

        # Generate embeddings
        logger.info("Generating embeddings for chunks")
        service.update_heartbeat()  # Update heartbeat
        embedded_chunks = service.generate_embeddings(chunks)

        # Store embeddings
        logger.info("Storing embeddings in vector database")
        service.update_heartbeat()  # Update heartbeat
        success = service.store_embeddings(embedded_chunks)

        # Show results
        total_time = time.time() - start_time
        stats = service.get_stats()

        print("\n" + "=" * 60)
        print("EMBEDDING CREATION COMPLETED")
        print("=" * 60)
        print(f"Total Chunks Processed: {len(chunks)}")
        print(f"Embeddings Generated: {len(embedded_chunks)}")
        print(f"Storage Success: {'✅' if success else '❌'}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Vector Database: {config.vector_db_type}")
        print(f"Embedding Model: {config.embedding_model}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error during embedding creation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup instance coordination
        service.cleanup_instance()


def handle_search(args: argparse.Namespace) -> None:
    """Handle search command."""
    logger.info(f"Searching for: '{args.query}'")

    # TODO: Implement actual search functionality
    print(f"\n🔍 Searching for: '{args.query}'")
    print(f"Top-K: {args.top_k}")
    print(f"Threshold: {args.threshold}")
    print(f"Vector Database: {args.vector_db_type}")

    # Mock search results
    print("\nSearch Results:")
    print("1. Document: example1.pdf, Score: 0.95")
    print("2. Document: example2.txt, Score: 0.87")
    print("3. Document: example3.pdf, Score: 0.82")
    print("\n⚠️  Search functionality not yet implemented")


def handle_list_sources(args: argparse.Namespace) -> None:
    """Handle list-sources command."""
    logger.info(f"Listing sources (type: {args.source_type})")

    print(f"\n📁 Available Sources (filter: {args.source_type})")
    print("=" * 40)

    # Mock source listing
    if args.source_type in ["local", "all"]:
        print("Local Sources:")
        print("  - ./documents/ (PDF files)")
        print("  - ./texts/ (TXT files)")

    if args.source_type in ["s3", "all"]:
        print("S3 Sources:")
        print("  - s3://my-bucket/docs/")

    if args.source_type in ["sharepoint", "all"]:
        print("SharePoint Sources:")
        print("  - https://mycompany.sharepoint.com/sites/docs")

    print("\n⚠️  Source listing functionality not yet implemented")


def handle_status(args: argparse.Namespace) -> None:
    """Handle status command."""
    logger.info("Checking system status")

    print("\n📊 System Status")
    print("=" * 40)

    # Check configuration
    try:
        config = get_config(args.env_file)
        print("✅ Configuration: Valid")

        if args.show_config:
            print("\nCurrent Configuration:")
            print(f"  Vector Database: {config.vector_db_type}")
            print(f"  Embedding Model: {config.embedding_model}")
            print(f"  Chunk Size: {config.chunk_size}")
            print(f"  Source Type: {config.source_type}")
    except Exception as e:
        print(f"❌ Configuration: Invalid ({e})")

    # Check connections if requested
    if args.check_connections:
        print("\nConnection Status:")
        print("  OpenAI API: ⚠️  Not checked (not implemented)")
        print("  Vector Database: ⚠️  Not checked (not implemented)")

    print("\n⚠️  Full status checking not yet implemented")


def main():
    """Main CLI entry point."""
    # Initialize logging system
    setup_logging()

    parser = create_parser()
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # Route to appropriate command handler
        if args.command == "create-embeddings":
            handle_create_embeddings(args)
        elif args.command == "search":
            handle_search(args)
        elif args.command == "list-sources":
            handle_list_sources(args)
        elif args.command == "status":
            handle_status(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
