"""
Configuration management for RAG embedding database application.
All configuration should be loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback for environments where python-dotenv is not available
    def load_dotenv(dotenv_path=None, **kwargs) -> bool:  # type: ignore[misc]
        return False


import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration class for RAG embedding database application."""

    # I/O Configuration
    datapath: Path = field(default_factory=lambda: Path("."))
    output: str = "./"
    output_format: str = "jsonl"

    # Input Source Configuration
    source_type: str = "local"  # local, s3, sharepoint
    source_uri: Optional[str] = None  # If None, uses datapath
    source_credentials: Dict[str, Any] = field(default_factory=dict)
    source_include_patterns: list = field(default_factory=lambda: ["**/*"])
    source_exclude_patterns: list = field(default_factory=list)
    source_max_file_size: int = 50 * 1024 * 1024  # 50MB
    source_batch_size: int = 100

    # Document Processing Configuration
    chunk_size: int = 512
    doctype: str = "pdf"
    chunking_strategy: str = "semantic"
    chunking_params: Dict[str, Any] = field(default_factory=dict)

    # Prompt Template Configuration
    embedding_prompt_template: Optional[str] = None
    custom_prompt_variables: Dict[str, Any] = field(default_factory=dict)

    # Embedding Provider Configuration
    embedding_provider: str = "openai"  # openai, azure_openai, aws_bedrock, google_vertex, lmstudio, ollama, llamacpp
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    batch_size_embeddings: int = 100

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_timeout: int = 60
    openai_max_retries: int = 3

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_deployment_name: Optional[str] = None
    azure_openai_deployment_mapping: Dict[str, str] = field(default_factory=dict)
    azure_openai_timeout: int = 60
    azure_openai_max_retries: int = 3

    # AWS Bedrock Configuration
    aws_bedrock_region: str = "us-east-1"
    aws_bedrock_access_key_id: Optional[str] = None
    aws_bedrock_secret_access_key: Optional[str] = None
    aws_bedrock_session_token: Optional[str] = None
    aws_bedrock_profile_name: Optional[str] = None
    aws_bedrock_role_arn: Optional[str] = None
    aws_bedrock_timeout: int = 60

    # Google Vertex AI Configuration
    google_vertex_project_id: Optional[str] = None
    google_vertex_location: str = "us-central1"
    google_vertex_credentials_path: Optional[str] = None
    google_vertex_timeout: int = 60

    # LMStudio Configuration
    lmstudio_base_url: str = "http://localhost:1234"
    lmstudio_api_key: Optional[str] = None
    lmstudio_timeout: int = 60
    lmstudio_verify_ssl: bool = True

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120
    ollama_verify_ssl: bool = True

    # Llama.cpp Configuration
    llamacpp_base_url: str = "http://localhost:8000"
    llamacpp_api_key: Optional[str] = None
    llamacpp_model_name: str = "unknown"
    llamacpp_timeout: int = 120
    llamacpp_verify_ssl: bool = True
    llamacpp_dimensions: Optional[int] = None

    # Vector Database Configuration
    # TODO: Add vector database configurations
    vector_db_type: str = "faiss"  # faiss, pinecone, chroma
    vector_db_config: Dict[str, Any] = field(default_factory=dict)

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "rag-embeddings"

    # Chroma Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "rag-embeddings"

    # FAISS Configuration
    faiss_index_path: str = "./faiss_index"
    faiss_index_type: str = "IndexFlatIP"  # IndexFlatIP, IndexIVFFlat, IndexHNSW

    # Azure AI Search Configuration
    azure_search_service_name: Optional[str] = None
    azure_search_api_key: Optional[str] = None
    azure_search_index_name: str = "rag-embeddings"
    azure_search_api_version: str = "2023-11-01"

    # AWS Elasticsearch Configuration
    aws_elasticsearch_endpoint: Optional[str] = None
    aws_elasticsearch_region: str = "us-east-1"
    aws_elasticsearch_index_name: str = "rag-embeddings"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # PGVector Configuration
    pgvector_host: str = "localhost"
    pgvector_port: int = 5432
    pgvector_database: str = "vectordb"
    pgvector_user: str = "postgres"
    pgvector_password: Optional[str] = None
    pgvector_table_name: str = "rag_embeddings"

    # Azure Configuration
    use_azure_identity: bool = False
    azure_openai_enabled: bool = False

    # Performance Configuration
    workers: int = 1
    embed_workers: int = 1
    pace: bool = True

    # Rate Limiting Configuration for Embedding Providers
    rate_limit_enabled: bool = False
    rate_limit_strategy: str = "sliding_window"  # fixed_window, sliding_window, token_bucket, adaptive
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_requests_per_hour: Optional[int] = None
    rate_limit_tokens_per_minute: Optional[int] = None
    rate_limit_tokens_per_hour: Optional[int] = None
    rate_limit_max_burst: Optional[int] = None
    rate_limit_burst_window: float = 60.0
    rate_limit_target_response_time: float = 2.0
    rate_limit_max_response_time: float = 10.0
    rate_limit_adaptation_factor: float = 0.1
    rate_limit_max_retries: int = 3
    rate_limit_base_delay: float = 1.0
    rate_limit_max_retry_delay: float = 60.0
    rate_limit_exponential_backoff: bool = True
    rate_limit_jitter: bool = True
    rate_limit_retry_on_rate_limit: bool = True
    rate_limit_retry_on_server_error: bool = True
    rate_limit_fail_fast_on_auth_error: bool = True
    rate_limit_preset: Optional[str] = None

    # Vector Store Rate Limiting Configuration
    vector_store_rate_limit_enabled: bool = False
    vector_store_rate_limit_strategy: str = "sliding_window"
    vector_store_rate_limit_requests_per_minute: Optional[int] = None
    vector_store_rate_limit_requests_per_hour: Optional[int] = None
    vector_store_rate_limit_max_burst: Optional[int] = None
    vector_store_rate_limit_burst_window: float = 60.0
    vector_store_rate_limit_target_response_time: float = 1.0
    vector_store_rate_limit_max_response_time: float = 5.0
    vector_store_rate_limit_max_retries: int = 3
    vector_store_rate_limit_base_delay: float = 0.5

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Load default .env file if it exists
            load_dotenv()

        config = cls()

        # I/O Configuration
        if datapath := os.getenv("EMBEDDING_DATAPATH"):
            config.datapath = Path(datapath)
        config.output = os.getenv("EMBEDDING_OUTPUT", config.output)
        config.output_format = os.getenv("EMBEDDING_OUTPUT_FORMAT", config.output_format)

        # Input Source Configuration
        config.source_type = os.getenv("EMBEDDING_SOURCE_TYPE", config.source_type)
        config.source_uri = os.getenv("EMBEDDING_SOURCE_URI", config.source_uri)
        config.source_max_file_size = int(os.getenv("EMBEDDING_SOURCE_MAX_FILE_SIZE", config.source_max_file_size))
        config.source_batch_size = int(os.getenv("EMBEDDING_SOURCE_BATCH_SIZE", config.source_batch_size))

        # Parse source credentials from JSON string
        source_credentials_str = os.getenv("EMBEDDING_SOURCE_CREDENTIALS")
        if source_credentials_str:
            try:
                config.source_credentials = json.loads(source_credentials_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_CREDENTIALS: {e}")

        # Parse include/exclude patterns from JSON strings
        source_include_str = os.getenv("EMBEDDING_SOURCE_INCLUDE_PATTERNS")
        if source_include_str:
            try:
                config.source_include_patterns = json.loads(source_include_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_INCLUDE_PATTERNS: {e}")

        source_exclude_str = os.getenv("EMBEDDING_SOURCE_EXCLUDE_PATTERNS")
        if source_exclude_str:
            try:
                config.source_exclude_patterns = json.loads(source_exclude_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_EXCLUDE_PATTERNS: {e}")

        # Document Processing Configuration
        config.chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", config.chunk_size))
        config.doctype = os.getenv("EMBEDDING_DOCTYPE", config.doctype)
        config.chunking_strategy = os.getenv("EMBEDDING_CHUNKING_STRATEGY", config.chunking_strategy)

        # Parse chunking params from JSON string
        chunking_params_str = os.getenv("EMBEDDING_CHUNKING_PARAMS")
        if chunking_params_str:
            try:
                config.chunking_params = json.loads(chunking_params_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_CHUNKING_PARAMS: {e}")

        # Prompt Template Configuration
        config.embedding_prompt_template = os.getenv("EMBEDDING_PROMPT_TEMPLATE", config.embedding_prompt_template)

        # Parse custom prompt variables from JSON string
        custom_prompt_vars_str = os.getenv("EMBEDDING_CUSTOM_PROMPT_VARIABLES")
        if custom_prompt_vars_str:
            try:
                config.custom_prompt_variables = json.loads(custom_prompt_vars_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_CUSTOM_PROMPT_VARIABLES: {e}")

        # Embedding Provider Configuration
        config.embedding_provider = os.getenv("EMBEDDING_PROVIDER", config.embedding_provider)
        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        config.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", config.embedding_dimensions))
        config.batch_size_embeddings = int(os.getenv("EMBEDDING_BATCH_SIZE", config.batch_size_embeddings))

        # OpenAI Configuration
        config.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        config.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        config.openai_base_url = os.getenv("OPENAI_BASE_URL")
        config.openai_timeout = int(os.getenv("OPENAI_TIMEOUT", config.openai_timeout))
        config.openai_max_retries = int(os.getenv("OPENAI_MAX_RETRIES", config.openai_max_retries))

        # Azure OpenAI Configuration
        config.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        config.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        config.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", config.azure_openai_api_version)
        config.azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        config.azure_openai_timeout = int(os.getenv("AZURE_OPENAI_TIMEOUT", config.azure_openai_timeout))
        config.azure_openai_max_retries = int(os.getenv("AZURE_OPENAI_MAX_RETRIES", config.azure_openai_max_retries))

        # Parse Azure OpenAI deployment mapping from JSON
        deployment_mapping_str = os.getenv("AZURE_OPENAI_DEPLOYMENT_MAPPING")
        if deployment_mapping_str:
            try:
                config.azure_openai_deployment_mapping = json.loads(deployment_mapping_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AZURE_OPENAI_DEPLOYMENT_MAPPING: {e}")

        # AWS Bedrock Configuration
        config.aws_bedrock_region = os.getenv("AWS_BEDROCK_REGION", config.aws_bedrock_region)
        config.aws_bedrock_access_key_id = os.getenv("AWS_BEDROCK_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        config.aws_bedrock_secret_access_key = os.getenv("AWS_BEDROCK_SECRET_ACCESS_KEY") or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        config.aws_bedrock_session_token = os.getenv("AWS_BEDROCK_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
        config.aws_bedrock_profile_name = os.getenv("AWS_BEDROCK_PROFILE_NAME") or os.getenv("AWS_PROFILE")
        config.aws_bedrock_role_arn = os.getenv("AWS_BEDROCK_ROLE_ARN")
        config.aws_bedrock_timeout = int(os.getenv("AWS_BEDROCK_TIMEOUT", config.aws_bedrock_timeout))

        # Google Vertex AI Configuration
        config.google_vertex_project_id = os.getenv("GOOGLE_VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        config.google_vertex_location = os.getenv("GOOGLE_VERTEX_LOCATION", config.google_vertex_location)
        config.google_vertex_credentials_path = os.getenv("GOOGLE_VERTEX_CREDENTIALS_PATH") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        config.google_vertex_timeout = int(os.getenv("GOOGLE_VERTEX_TIMEOUT", config.google_vertex_timeout))

        # LMStudio Configuration
        config.lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", config.lmstudio_base_url)
        config.lmstudio_api_key = os.getenv("LMSTUDIO_API_KEY")
        config.lmstudio_timeout = int(os.getenv("LMSTUDIO_TIMEOUT", config.lmstudio_timeout))
        config.lmstudio_verify_ssl = os.getenv("LMSTUDIO_VERIFY_SSL", "true").lower() in ("true", "1", "yes")

        # Ollama Configuration
        config.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.ollama_base_url)
        config.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", config.ollama_timeout))
        config.ollama_verify_ssl = os.getenv("OLLAMA_VERIFY_SSL", "true").lower() in ("true", "1", "yes")

        # Llama.cpp Configuration
        config.llamacpp_base_url = os.getenv("LLAMACPP_BASE_URL", config.llamacpp_base_url)
        config.llamacpp_api_key = os.getenv("LLAMACPP_API_KEY")
        config.llamacpp_model_name = os.getenv("LLAMACPP_MODEL_NAME", config.llamacpp_model_name)
        config.llamacpp_timeout = int(os.getenv("LLAMACPP_TIMEOUT", config.llamacpp_timeout))
        config.llamacpp_verify_ssl = os.getenv("LLAMACPP_VERIFY_SSL", "true").lower() in ("true", "1", "yes")
        if llamacpp_dimensions := os.getenv("LLAMACPP_DIMENSIONS"):
            config.llamacpp_dimensions = int(llamacpp_dimensions)

        # Vector Database Configuration
        config.vector_db_type = os.getenv("VECTOR_DB_TYPE", config.vector_db_type)

        # Parse vector DB config from JSON string
        vector_db_config_str = os.getenv("VECTOR_DB_CONFIG")
        if vector_db_config_str:
            try:
                config.vector_db_config = json.loads(vector_db_config_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse VECTOR_DB_CONFIG: {e}")

        # Pinecone Configuration
        config.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        config.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        config.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", config.pinecone_index_name)

        # Chroma Configuration
        config.chroma_host = os.getenv("CHROMA_HOST", config.chroma_host)
        config.chroma_port = int(os.getenv("CHROMA_PORT", config.chroma_port))
        config.chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", config.chroma_collection_name)

        # FAISS Configuration
        config.faiss_index_path = os.getenv("FAISS_INDEX_PATH", config.faiss_index_path)
        config.faiss_index_type = os.getenv("FAISS_INDEX_TYPE", config.faiss_index_type)

        # Azure AI Search Configuration
        config.azure_search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        config.azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        config.azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", config.azure_search_index_name)
        config.azure_search_api_version = os.getenv("AZURE_SEARCH_API_VERSION", config.azure_search_api_version)

        # AWS Elasticsearch Configuration
        config.aws_elasticsearch_endpoint = os.getenv("AWS_ELASTICSEARCH_ENDPOINT")
        config.aws_elasticsearch_region = os.getenv("AWS_ELASTICSEARCH_REGION", config.aws_elasticsearch_region)
        config.aws_elasticsearch_index_name = os.getenv(
            "AWS_ELASTICSEARCH_INDEX_NAME", config.aws_elasticsearch_index_name
        )
        config.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        config.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # PGVector Configuration
        config.pgvector_host = os.getenv("PGVECTOR_HOST", config.pgvector_host)
        config.pgvector_port = int(os.getenv("PGVECTOR_PORT", config.pgvector_port))
        config.pgvector_database = os.getenv("PGVECTOR_DATABASE", config.pgvector_database)
        config.pgvector_user = os.getenv("PGVECTOR_USER", config.pgvector_user)
        config.pgvector_password = os.getenv("PGVECTOR_PASSWORD")
        config.pgvector_table_name = os.getenv("PGVECTOR_TABLE_NAME", config.pgvector_table_name)

        # Azure Configuration
        config.use_azure_identity = os.getenv("EMBEDDING_USE_AZURE_IDENTITY", "false").lower() in ("true", "1", "yes")
        config.azure_openai_enabled = os.getenv("AZURE_OPENAI_ENABLED", "false").lower() in ("true", "1", "yes")

        # Performance Configuration
        config.workers = int(os.getenv("EMBEDDING_WORKERS", config.workers))
        config.embed_workers = int(os.getenv("EMBEDDING_EMBED_WORKERS", config.embed_workers))
        config.pace = os.getenv("EMBEDDING_PACE", "true").lower() in ("true", "1", "yes")

        # Rate Limiting Configuration for Embedding Providers
        config.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "false").lower() in ("true", "1", "yes")
        config.rate_limit_strategy = os.getenv("RATE_LIMIT_STRATEGY", config.rate_limit_strategy)
        config.rate_limit_preset = os.getenv("RATE_LIMIT_PRESET")

        # Parse numeric rate limits for embedding providers
        if requests_per_minute := os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE"):
            config.rate_limit_requests_per_minute = int(requests_per_minute)
        if requests_per_hour := os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR"):
            config.rate_limit_requests_per_hour = int(requests_per_hour)
        if tokens_per_minute := os.getenv("RATE_LIMIT_TOKENS_PER_MINUTE"):
            config.rate_limit_tokens_per_minute = int(tokens_per_minute)
        if tokens_per_hour := os.getenv("RATE_LIMIT_TOKENS_PER_HOUR"):
            config.rate_limit_tokens_per_hour = int(tokens_per_hour)
        if max_burst := os.getenv("RATE_LIMIT_MAX_BURST"):
            config.rate_limit_max_burst = int(max_burst)

        config.rate_limit_burst_window = float(os.getenv("RATE_LIMIT_BURST_WINDOW", config.rate_limit_burst_window))
        config.rate_limit_target_response_time = float(
            os.getenv("RATE_LIMIT_TARGET_RESPONSE_TIME", config.rate_limit_target_response_time)
        )
        config.rate_limit_max_response_time = float(
            os.getenv("RATE_LIMIT_MAX_RESPONSE_TIME", config.rate_limit_max_response_time)
        )
        config.rate_limit_adaptation_factor = float(
            os.getenv("RATE_LIMIT_ADAPTATION_FACTOR", config.rate_limit_adaptation_factor)
        )
        config.rate_limit_max_retries = int(os.getenv("RATE_LIMIT_MAX_RETRIES", config.rate_limit_max_retries))
        config.rate_limit_base_delay = float(os.getenv("RATE_LIMIT_BASE_DELAY", config.rate_limit_base_delay))
        config.rate_limit_max_retry_delay = float(
            os.getenv("RATE_LIMIT_MAX_RETRY_DELAY", config.rate_limit_max_retry_delay)
        )
        config.rate_limit_exponential_backoff = os.getenv("RATE_LIMIT_EXPONENTIAL_BACKOFF", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        config.rate_limit_jitter = os.getenv("RATE_LIMIT_JITTER", "true").lower() in ("true", "1", "yes")
        config.rate_limit_retry_on_rate_limit = os.getenv("RATE_LIMIT_RETRY_ON_RATE_LIMIT", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        config.rate_limit_retry_on_server_error = os.getenv("RATE_LIMIT_RETRY_ON_SERVER_ERROR", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        config.rate_limit_fail_fast_on_auth_error = os.getenv("RATE_LIMIT_FAIL_FAST_ON_AUTH_ERROR", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        # Vector Store Rate Limiting Configuration
        config.vector_store_rate_limit_enabled = os.getenv("VECTOR_STORE_RATE_LIMIT_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        config.vector_store_rate_limit_strategy = os.getenv(
            "VECTOR_STORE_RATE_LIMIT_STRATEGY", config.vector_store_rate_limit_strategy
        )

        if vs_requests_per_minute := os.getenv("VECTOR_STORE_RATE_LIMIT_REQUESTS_PER_MINUTE"):
            config.vector_store_rate_limit_requests_per_minute = int(vs_requests_per_minute)
        if vs_requests_per_hour := os.getenv("VECTOR_STORE_RATE_LIMIT_REQUESTS_PER_HOUR"):
            config.vector_store_rate_limit_requests_per_hour = int(vs_requests_per_hour)
        if vs_max_burst := os.getenv("VECTOR_STORE_RATE_LIMIT_MAX_BURST"):
            config.vector_store_rate_limit_max_burst = int(vs_max_burst)

        config.vector_store_rate_limit_burst_window = float(
            os.getenv("VECTOR_STORE_RATE_LIMIT_BURST_WINDOW", config.vector_store_rate_limit_burst_window)
        )
        config.vector_store_rate_limit_target_response_time = float(
            os.getenv(
                "VECTOR_STORE_RATE_LIMIT_TARGET_RESPONSE_TIME", config.vector_store_rate_limit_target_response_time
            )
        )
        config.vector_store_rate_limit_max_response_time = float(
            os.getenv("VECTOR_STORE_RATE_LIMIT_MAX_RESPONSE_TIME", config.vector_store_rate_limit_max_response_time)
        )
        config.vector_store_rate_limit_max_retries = int(
            os.getenv("VECTOR_STORE_RATE_LIMIT_MAX_RETRIES", config.vector_store_rate_limit_max_retries)
        )
        config.vector_store_rate_limit_base_delay = float(
            os.getenv("VECTOR_STORE_RATE_LIMIT_BASE_DELAY", config.vector_store_rate_limit_base_delay)
        )

        return config

    def validate(self) -> None:
        """Validate configuration."""
        # For local sources, validate datapath
        if self.source_type == "local" and not self.source_uri:
            if not self.datapath.exists() and str(self.datapath) != ".":
                raise ValueError(f"Data path does not exist: {self.datapath}")

        # Validate source type
        if self.source_type not in ["local", "s3", "sharepoint"]:
            raise ValueError(f"Invalid source type: {self.source_type}")

        # For non-local sources, require source_uri
        if self.source_type != "local" and not self.source_uri:
            raise ValueError(f"source_uri is required for source type: {self.source_type}")

        if self.doctype not in ["pdf", "txt", "json", "api", "pptx"]:
            raise ValueError(f"Invalid doctype: {self.doctype}")

        if self.output_format not in ["jsonl", "parquet"]:
            raise ValueError(f"Invalid output format: {self.output_format}")

        if self.chunking_strategy not in ["semantic", "fixed", "sentence"]:
            raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")

        # Validate embedding provider
        valid_providers = ["openai", "azure_openai", "aws_bedrock", "google_vertex", "lmstudio", "ollama", "llamacpp"]
        if self.embedding_provider not in valid_providers:
            raise ValueError(
                f"Invalid embedding provider: {self.embedding_provider}. Must be one of: {valid_providers}"
            )

        # Validate embedding provider specific requirements
        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")

        elif self.embedding_provider == "azure_openai":
            if not self.azure_openai_api_key:
                raise ValueError("Azure OpenAI API key is required for Azure OpenAI provider")
            if not self.azure_openai_endpoint:
                raise ValueError("Azure OpenAI endpoint is required for Azure OpenAI provider")

        elif self.embedding_provider == "aws_bedrock":
            # AWS credentials can come from multiple sources, so we don't enforce them here
            pass

        elif self.embedding_provider == "google_vertex":
            if not self.google_vertex_project_id:
                raise ValueError("Google Cloud project ID is required for Vertex AI provider")

        elif self.embedding_provider == "lmstudio":
            # LMStudio only requires base URL, which has a default
            pass

        elif self.embedding_provider == "ollama":
            # Ollama only requires base URL, which has a default
            pass

        elif self.embedding_provider == "llamacpp":
            # Llama.cpp only requires base URL, which has a default
            pass

        # Validate vector database type
        if self.vector_db_type not in [
            "faiss",
            "pinecone",
            "chroma",
            "azure_ai_search",
            "aws_elasticsearch",
            "pgvector",
        ]:
            raise ValueError(f"Invalid vector database type: {self.vector_db_type}")

        # Validate vector database specific requirements
        if self.vector_db_type == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key is required for Pinecone vector database")
            if not self.pinecone_environment:
                raise ValueError("Pinecone environment is required for Pinecone vector database")
        elif self.vector_db_type == "pgvector":
            if not self.pgvector_password:
                raise ValueError("PostgreSQL password is required for pgvector database")

        # Validate source file size limit
        if self.source_max_file_size <= 0:
            raise ValueError("source_max_file_size must be positive")

        if self.source_batch_size <= 0:
            raise ValueError("source_batch_size must be positive")

        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")

        if self.batch_size_embeddings <= 0:
            raise ValueError("batch_size_embeddings must be positive")

        # Skip API key validation for local providers or demo mode
        demo_key_values = ["demo_key_for_testing", "test", "mock"]
        local_providers = ["lmstudio", "ollama", "llamacpp"]

        if self.embedding_provider not in local_providers:
            # For cloud providers, check if we have appropriate credentials
            has_credentials = False

            if self.embedding_provider == "openai" and self.openai_api_key:
                has_credentials = True
            elif self.embedding_provider == "azure_openai" and self.azure_openai_api_key:
                has_credentials = True
            elif self.embedding_provider == "aws_bedrock":
                # AWS can use IAM roles, so don't enforce explicit keys
                has_credentials = True
            elif self.embedding_provider == "google_vertex":
                # Google can use default credentials, so don't enforce explicit path
                has_credentials = True

            # Check for demo/test mode
            if self.openai_api_key in demo_key_values or self.azure_openai_api_key in demo_key_values:
                has_credentials = True

            if not has_credentials and not self.use_azure_identity:
                raise ValueError(f"Credentials required for {self.embedding_provider} provider")

        # Validate rate limiting configuration
        valid_strategies = ["fixed_window", "sliding_window", "token_bucket", "adaptive"]
        if self.rate_limit_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid rate limiting strategy: {self.rate_limit_strategy}. Must be one of: {valid_strategies}"
            )

        if self.vector_store_rate_limit_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid vector store rate limiting strategy: {self.vector_store_rate_limit_strategy}. "
                f"Must be one of: {valid_strategies}"
            )

        # Validate rate limiting numeric values
        if self.rate_limit_enabled:
            if self.rate_limit_requests_per_minute is not None and self.rate_limit_requests_per_minute <= 0:
                raise ValueError("rate_limit_requests_per_minute must be positive")
            if self.rate_limit_tokens_per_minute is not None and self.rate_limit_tokens_per_minute <= 0:
                raise ValueError("rate_limit_tokens_per_minute must be positive")
            if self.rate_limit_max_burst is not None and self.rate_limit_max_burst <= 0:
                raise ValueError("rate_limit_max_burst must be positive")

        if self.vector_store_rate_limit_enabled:
            if (
                self.vector_store_rate_limit_requests_per_minute is not None
                and self.vector_store_rate_limit_requests_per_minute <= 0
            ):
                raise ValueError("vector_store_rate_limit_requests_per_minute must be positive")
            if self.vector_store_rate_limit_max_burst is not None and self.vector_store_rate_limit_max_burst <= 0:
                raise ValueError("vector_store_rate_limit_max_burst must be positive")

        # Validate adaptive rate limiting parameters
        if self.rate_limit_adaptation_factor < 0.0 or self.rate_limit_adaptation_factor > 1.0:
            raise ValueError("rate_limit_adaptation_factor must be between 0.0 and 1.0")

        if self.rate_limit_target_response_time <= 0:
            raise ValueError("rate_limit_target_response_time must be positive")

        if self.rate_limit_max_response_time <= self.rate_limit_target_response_time:
            raise ValueError("rate_limit_max_response_time must be greater than rate_limit_target_response_time")


def get_config(env_file: Optional[str] = None) -> EmbeddingConfig:
    """Get validated configuration instance."""
    config = EmbeddingConfig.from_env(env_file)
    config.validate()
    return config
