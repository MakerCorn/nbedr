"""
Configuration management for RAG embedding database application.
All configuration should be loaded from environment variables.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback for environments where python-dotenv is not available
    def load_dotenv(*args, **kwargs):
        pass
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
    source_include_patterns: list = field(default_factory=lambda: ['**/*'])
    source_exclude_patterns: list = field(default_factory=list)
    source_max_file_size: int = 50 * 1024 * 1024  # 50MB
    source_batch_size: int = 100
    
    # Document Processing Configuration
    chunk_size: int = 512
    doctype: str = "pdf"
    chunking_strategy: str = "semantic"
    chunking_params: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding Configuration
    openai_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    batch_size_embeddings: int = 100
    
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
    
    # Azure Configuration
    use_azure_identity: bool = False
    azure_openai_enabled: bool = False
    
    # Performance Configuration
    workers: int = 1
    embed_workers: int = 1
    pace: bool = True
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = False
    rate_limit_strategy: str = "sliding_window"
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_requests_per_hour: Optional[int] = None
    rate_limit_tokens_per_minute: Optional[int] = None
    rate_limit_tokens_per_hour: Optional[int] = None
    rate_limit_max_burst: Optional[int] = None
    rate_limit_burst_window: float = 60.0
    rate_limit_max_retries: int = 3
    rate_limit_base_delay: float = 1.0
    rate_limit_preset: Optional[str] = None
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'EmbeddingConfig':
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Load default .env file if it exists
            load_dotenv()
        
        config = cls()
        
        # I/O Configuration
        if os.getenv('EMBEDDING_DATAPATH'):
            config.datapath = Path(os.getenv('EMBEDDING_DATAPATH'))
        config.output = os.getenv('EMBEDDING_OUTPUT', config.output)
        config.output_format = os.getenv('EMBEDDING_OUTPUT_FORMAT', config.output_format)
        
        # Input Source Configuration
        config.source_type = os.getenv('EMBEDDING_SOURCE_TYPE', config.source_type)
        config.source_uri = os.getenv('EMBEDDING_SOURCE_URI', config.source_uri)
        config.source_max_file_size = int(os.getenv('EMBEDDING_SOURCE_MAX_FILE_SIZE', config.source_max_file_size))
        config.source_batch_size = int(os.getenv('EMBEDDING_SOURCE_BATCH_SIZE', config.source_batch_size))
        
        # Parse source credentials from JSON string
        source_credentials_str = os.getenv('EMBEDDING_SOURCE_CREDENTIALS')
        if source_credentials_str:
            try:
                config.source_credentials = json.loads(source_credentials_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_CREDENTIALS: {e}")
        
        # Parse include/exclude patterns from JSON strings
        source_include_str = os.getenv('EMBEDDING_SOURCE_INCLUDE_PATTERNS')
        if source_include_str:
            try:
                config.source_include_patterns = json.loads(source_include_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_INCLUDE_PATTERNS: {e}")
        
        source_exclude_str = os.getenv('EMBEDDING_SOURCE_EXCLUDE_PATTERNS')
        if source_exclude_str:
            try:
                config.source_exclude_patterns = json.loads(source_exclude_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_SOURCE_EXCLUDE_PATTERNS: {e}")
        
        # Document Processing Configuration
        config.chunk_size = int(os.getenv('EMBEDDING_CHUNK_SIZE', config.chunk_size))
        config.doctype = os.getenv('EMBEDDING_DOCTYPE', config.doctype)
        config.chunking_strategy = os.getenv('EMBEDDING_CHUNKING_STRATEGY', config.chunking_strategy)
        
        # Parse chunking params from JSON string
        chunking_params_str = os.getenv('EMBEDDING_CHUNKING_PARAMS')
        if chunking_params_str:
            try:
                config.chunking_params = json.loads(chunking_params_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse EMBEDDING_CHUNKING_PARAMS: {e}")
        
        # Embedding Configuration
        config.openai_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY')
        config.embedding_model = os.getenv('EMBEDDING_MODEL', config.embedding_model)
        config.embedding_dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', config.embedding_dimensions))
        config.batch_size_embeddings = int(os.getenv('EMBEDDING_BATCH_SIZE', config.batch_size_embeddings))
        
        # Vector Database Configuration
        config.vector_db_type = os.getenv('VECTOR_DB_TYPE', config.vector_db_type)
        
        # Parse vector DB config from JSON string
        vector_db_config_str = os.getenv('VECTOR_DB_CONFIG')
        if vector_db_config_str:
            try:
                config.vector_db_config = json.loads(vector_db_config_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse VECTOR_DB_CONFIG: {e}")
        
        # Pinecone Configuration
        config.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        config.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        config.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', config.pinecone_index_name)
        
        # Chroma Configuration
        config.chroma_host = os.getenv('CHROMA_HOST', config.chroma_host)
        config.chroma_port = int(os.getenv('CHROMA_PORT', config.chroma_port))
        config.chroma_collection_name = os.getenv('CHROMA_COLLECTION_NAME', config.chroma_collection_name)
        
        # FAISS Configuration
        config.faiss_index_path = os.getenv('FAISS_INDEX_PATH', config.faiss_index_path)
        config.faiss_index_type = os.getenv('FAISS_INDEX_TYPE', config.faiss_index_type)
        
        # Azure AI Search Configuration
        config.azure_search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        config.azure_search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
        config.azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', config.azure_search_index_name)
        config.azure_search_api_version = os.getenv('AZURE_SEARCH_API_VERSION', config.azure_search_api_version)
        
        # AWS Elasticsearch Configuration
        config.aws_elasticsearch_endpoint = os.getenv('AWS_ELASTICSEARCH_ENDPOINT')
        config.aws_elasticsearch_region = os.getenv('AWS_ELASTICSEARCH_REGION', config.aws_elasticsearch_region)
        config.aws_elasticsearch_index_name = os.getenv('AWS_ELASTICSEARCH_INDEX_NAME', config.aws_elasticsearch_index_name)
        config.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        config.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Azure Configuration
        config.use_azure_identity = os.getenv('EMBEDDING_USE_AZURE_IDENTITY', 'false').lower() in ('true', '1', 'yes')
        config.azure_openai_enabled = os.getenv('AZURE_OPENAI_ENABLED', 'false').lower() in ('true', '1', 'yes')
        
        # Performance Configuration
        config.workers = int(os.getenv('EMBEDDING_WORKERS', config.workers))
        config.embed_workers = int(os.getenv('EMBEDDING_EMBED_WORKERS', config.embed_workers))
        config.pace = os.getenv('EMBEDDING_PACE', 'true').lower() in ('true', '1', 'yes')
        
        # Rate Limiting Configuration
        config.rate_limit_enabled = os.getenv('EMBEDDING_RATE_LIMIT_ENABLED', 'false').lower() in ('true', '1', 'yes')
        config.rate_limit_strategy = os.getenv('EMBEDDING_RATE_LIMIT_STRATEGY', config.rate_limit_strategy)
        config.rate_limit_preset = os.getenv('EMBEDDING_RATE_LIMIT_PRESET')
        
        # Parse numeric rate limits
        if os.getenv('EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE'):
            config.rate_limit_requests_per_minute = int(os.getenv('EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE'))
        if os.getenv('EMBEDDING_RATE_LIMIT_REQUESTS_PER_HOUR'):
            config.rate_limit_requests_per_hour = int(os.getenv('EMBEDDING_RATE_LIMIT_REQUESTS_PER_HOUR'))
        if os.getenv('EMBEDDING_RATE_LIMIT_TOKENS_PER_MINUTE'):
            config.rate_limit_tokens_per_minute = int(os.getenv('EMBEDDING_RATE_LIMIT_TOKENS_PER_MINUTE'))
        if os.getenv('EMBEDDING_RATE_LIMIT_TOKENS_PER_HOUR'):
            config.rate_limit_tokens_per_hour = int(os.getenv('EMBEDDING_RATE_LIMIT_TOKENS_PER_HOUR'))
        if os.getenv('EMBEDDING_RATE_LIMIT_MAX_BURST'):
            config.rate_limit_max_burst = int(os.getenv('EMBEDDING_RATE_LIMIT_MAX_BURST'))
        
        config.rate_limit_burst_window = float(os.getenv('EMBEDDING_RATE_LIMIT_BURST_WINDOW', config.rate_limit_burst_window))
        config.rate_limit_max_retries = int(os.getenv('EMBEDDING_RATE_LIMIT_MAX_RETRIES', config.rate_limit_max_retries))
        config.rate_limit_base_delay = float(os.getenv('EMBEDDING_RATE_LIMIT_BASE_DELAY', config.rate_limit_base_delay))
        
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
        
        # Validate vector database type
        if self.vector_db_type not in ["faiss", "pinecone", "chroma"]:
            raise ValueError(f"Invalid vector database type: {self.vector_db_type}")
        
        # Validate vector database specific requirements
        if self.vector_db_type == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key is required for Pinecone vector database")
            if not self.pinecone_environment:
                raise ValueError("Pinecone environment is required for Pinecone vector database")
        
        # Validate source file size limit
        if self.source_max_file_size <= 0:
            raise ValueError("source_max_file_size must be positive")
        
        if self.source_batch_size <= 0:
            raise ValueError("source_batch_size must be positive")
        
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        
        if self.batch_size_embeddings <= 0:
            raise ValueError("batch_size_embeddings must be positive")
        
        # Allow demo mode with mock API key
        if not self.openai_key and not self.use_azure_identity:
            raise ValueError("OpenAI API key is required unless using Azure identity")
        elif self.openai_key == "demo_key_for_testing":
            pass  # Allow demo mode

def get_config(env_file: Optional[str] = None) -> EmbeddingConfig:
    """Get validated configuration instance."""
    config = EmbeddingConfig.from_env(env_file)
    config.validate()
    return config