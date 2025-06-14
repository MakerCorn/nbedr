# Changelog

All notable changes to the RAG Embeddings Database project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2025-06-14

### Added

#### Advanced Rate Limiting and Throttling System
- **Comprehensive Rate Limiting**: Complete rate limiting system for both embedding providers and vector databases
  - Four intelligent strategies: fixed_window, sliding_window, token_bucket, adaptive
  - Provider-specific rate limiting with optimized presets for all major services
  - Vector store rate limiting for database operations
  - Adaptive rate limiting that automatically adjusts based on response times
  - Burst handling with configurable windows and limits
  - Token-aware rate limiting for cost control

#### Rate Limiting Features
- **Multi-Strategy Support**: 
  - Fixed Window: Classic time-based rate limiting
  - Sliding Window: More accurate rolling window rate limiting
  - Token Bucket: Allows controlled burst traffic
  - Adaptive: Self-tuning based on API response times
- **Provider-Specific Presets**: Pre-configured rate limits for popular services
  - OpenAI (Tier 1, 2, 3): 500-5000 RPM, 350K-5M TPM
  - Azure OpenAI: Standard and Provisioned deployment limits
  - AWS Bedrock: Titan and Cohere embedding limits
  - Google Vertex AI: Gecko model optimized limits
  - Local Providers: Conservative limits for development
- **Advanced Configuration Options**:
  - Request and token-based rate limiting
  - Burst traffic handling with configurable windows
  - Exponential backoff with jitter for retries
  - Response time monitoring and adaptive adjustments
  - Configurable retry logic with multiple error types

#### Vector Store Rate Limiting
- **Separate Rate Limiting**: Independent rate limiting for vector database operations
- **Database Operation Throttling**: Rate limiting for add_documents, search, and delete operations
- **Performance Monitoring**: Response time tracking for vector store operations
- **Error Handling**: Comprehensive error handling with retry logic

#### Rate Limiting Configuration
- **Environment Variable Support**: Complete configuration via environment variables
  - `RATE_LIMIT_ENABLED`: Enable/disable embedding provider rate limiting
  - `RATE_LIMIT_STRATEGY`: Choose rate limiting strategy
  - `RATE_LIMIT_REQUESTS_PER_MINUTE`: Set request rate limits
  - `RATE_LIMIT_TOKENS_PER_MINUTE`: Set token rate limits
  - `VECTOR_STORE_RATE_LIMIT_ENABLED`: Enable vector store rate limiting
- **Configuration Validation**: Comprehensive validation of all rate limiting parameters
- **Preset System**: Easy configuration using predefined service presets

#### Rate Limiting Documentation
- **Comprehensive Guide**: Detailed documentation in README with practical examples
- **Best Practices**: Production-ready configuration recommendations
- **Configuration Examples**: Multiple setup scenarios from conservative to high-volume
- **Cost Optimization**: Guidance for controlling API costs through rate limiting
- **Monitoring**: Information on rate limiting statistics and optimization

### Technical Improvements

#### Rate Limiting Architecture
- **BaseEmbeddingProvider Enhancement**: Integrated rate limiting into base embedding provider class
  - `_create_rate_limiter()`: Rate limiter factory method
  - `_apply_rate_limiting()`: Pre-request rate limiting application
  - `_record_response()`: Response time and token usage tracking
  - `_record_error()`: Error tracking for adaptive adjustments
- **BaseVectorStore Enhancement**: Added rate limiting to vector store operations
  - Vector-specific rate limiting configuration
  - Operation-level rate limiting (add_documents, search, delete)
  - Performance monitoring and error tracking
- **Enhanced Rate Limiter**: Improved rate limiting engine with embedding-specific features
  - Token-aware rate limiting for cost control
  - Provider-specific preset configurations
  - Advanced statistics tracking and reporting

#### Implementation Details
- **Async Integration**: Full async/await support for non-blocking rate limiting
- **Memory Efficient**: Sliding window implementation with automatic cleanup
- **Thread Safe**: Thread-safe rate limiting for concurrent operations
- **Statistics Tracking**: Comprehensive metrics collection for optimization
- **Error Recovery**: Robust error handling with configurable retry strategies

#### Configuration System Updates
- **Extended EmbeddingConfig**: Added 20+ new rate limiting configuration options
- **Environment Variable Support**: Complete environment variable configuration
- **Validation System**: Comprehensive validation for all rate limiting parameters
- **Backward Compatibility**: All existing configurations continue to work

#### Multi-Instance Parallel Processing
- **Instance Coordination System**: Complete coordination system for running multiple instances safely
  - Automatic conflict detection for output paths and vector database files
  - Instance-specific path generation to prevent file conflicts
  - Instance registry with heartbeat monitoring and cleanup
  - Compatible instance grouping based on configuration hash
- **File Locking**: Comprehensive file locking for FAISS index operations
  - Exclusive locks for index saving and loading operations
  - Atomic file operations to prevent corruption
  - Cross-instance coordination for shared vector database access
- **Shared Rate Limiting**: Rate limit coordination across multiple instances
  - Automatic rate limit distribution among active instances
  - Shared state tracking for token usage and response times
  - Fair resource allocation to prevent quota violations
- **CLI Enhancements**: New command-line options for instance management
  - `--disable-coordination`: Disable instance coordination
  - `--instance-id`: Custom instance identification
  - `--list-instances`: View all active instances
- **Production Ready**: Enterprise-grade parallel processing capabilities
  - Process isolation with independent failure handling
  - Scalable to dozens of concurrent instances
  - Container and orchestration platform support

## [1.3.0] - 2025-06-14

### Added

#### Comprehensive Embedding Provider Support
- **7 Embedding Providers**: Complete support for all major embedding platforms
  - **OpenAI**: Industry standard with text-embedding-3-large/small models
  - **Azure OpenAI**: Enterprise-grade OpenAI with deployment mapping support
  - **AWS Bedrock**: Amazon's model marketplace (Titan, Cohere embeddings)
  - **Google Vertex AI**: Google Cloud's embedding models (Gecko, multilingual)
  - **LMStudio**: Local development with GUI model management
  - **Ollama**: Privacy-focused local embeddings with command-line interface
  - **Llama.cpp**: Advanced local deployment with custom model support

#### New Provider Architecture
- **BaseEmbeddingProvider**: Abstract interface for all embedding providers
- **EmbeddingProviderFactory**: Factory pattern for provider instantiation
- **Async Support**: Full async/await support for high-performance operations
- **Unified Configuration**: Consistent configuration across all providers
- **Health Checks**: Built-in health monitoring for all providers
- **Model Discovery**: Automatic model listing and capability detection

#### Enhanced Configuration System
- **Provider Selection**: Environment variable-driven provider configuration
- **Provider-Specific Settings**: Dedicated configuration for each provider
  - OpenAI: API key, organization, base URL, timeout, retries
  - Azure OpenAI: Endpoint, deployment mapping, API version
  - AWS Bedrock: Region, credentials, IAM role support
  - Google Vertex: Project ID, location, service account credentials
  - LMStudio: Base URL, API key, SSL verification
  - Ollama: Base URL, timeout, SSL verification
  - Llama.cpp: Model name, dimensions, API authentication
- **Credential Validation**: Provider-specific credential requirements
- **Demo Mode Support**: Mock providers for testing and development

#### Local Provider Features
- **Privacy-First Options**: Complete local processing capabilities
- **No Internet Required**: Offline embedding generation
- **Cost-Free Operation**: Zero per-token costs for local providers
- **Custom Model Support**: Ability to use fine-tuned and custom models
- **Performance Optimization**: Direct hardware utilization

#### Cloud Provider Features
- **Enterprise Integration**: Full support for enterprise cloud platforms
- **Scalability**: Automatic scaling and load balancing
- **Reliability**: Built-in retry logic and error handling
- **Cost Optimization**: Batch processing and rate limiting
- **Compliance Support**: Enterprise security and compliance features

#### Comprehensive Documentation
- **Provider Comparison Guide**: Detailed comparison of all 7 providers
- **Quick Start Guides**: Provider-specific setup instructions
- **Configuration Examples**: Complete environment variable examples
- **Cost Analysis**: Detailed cost comparison and optimization guidance
- **Privacy Guidelines**: Security and privacy considerations
- **Performance Benchmarks**: Latency, throughput, and reliability metrics
- **RAG Integration Guide**: Complete examples for using embeddings in RAG applications
- **Framework Integration**: LangChain and other popular framework examples

### Technical Improvements

#### Architecture Enhancements
- **Modular Design**: Clean separation between providers
- **Error Handling**: Comprehensive error handling and logging
- **Fallback Support**: Graceful degradation to mock embeddings
- **Type Safety**: Full type hints and validation
- **Resource Management**: Proper connection and resource cleanup

#### Performance Optimizations
- **Batch Processing**: Optimal batch sizes for each provider
- **Concurrent Operations**: Parallel processing where supported
- **Connection Pooling**: Efficient connection reuse
- **Caching**: Model info and capability caching
- **Rate Limiting**: Provider-specific rate limiting

#### Developer Experience
- **Factory Pattern**: Easy provider instantiation
- **Configuration Builder**: Automatic config generation from EmbeddingConfig
- **Error Messages**: Clear, actionable error messages
- **Logging**: Comprehensive logging for debugging
- **Testing Support**: Mock providers for unit testing

### Dependencies
- **aiohttp**: HTTP client for local providers (LMStudio, Ollama, Llama.cpp)
- **google-cloud-aiplatform**: Google Vertex AI integration
- **vertexai**: Google Vertex AI Python SDK
- **boto3/botocore**: AWS Bedrock integration (reused existing dependency)

## [1.2.0] - 2025-06-14

### Added

#### PGVector Database Support
- **PostgreSQL with pgvector Extension**: Complete integration with PostgreSQL vector capabilities
  - ACID compliance with full transactional support
  - Cost-effective solution leveraging existing PostgreSQL infrastructure
  - Rich querying combining vector search with SQL joins and filters
  - IVFFlat indexing for efficient similarity search
  - Support for cosine similarity operations
  - Metadata filtering using JSONB and GIN indexes

#### Application Rebranding
- **Renamed CLI from rag_cli.py to nbedr.py**: Aligned with application branding as NBEDR
- **Updated all documentation**: References to CLI commands now use nbedr.py
- **Maintained backward compatibility**: All functionality preserved during rename

#### Enhanced Configuration
- **PGVector Configuration**: Complete environment variable support for PostgreSQL connections
  - PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DATABASE
  - PGVECTOR_USER, PGVECTOR_PASSWORD, PGVECTOR_TABLE_NAME
- **Vector Database Validation**: Added pgvector to supported database types
- **Configuration Validation**: Password requirement validation for pgvector connections

#### Documentation Enhancements
- **PGVector Integration Guide**: Complete setup and usage documentation
- **Updated Decision Matrix**: Added PGVector comparison across all evaluation criteria
- **Use Case Recommendations**: Specific guidance for when to choose PGVector
- **Architecture Diagrams**: Updated to include PGVector as sixth database option

### Dependencies
- **asyncpg**: PostgreSQL async client for high-performance database operations
- **psycopg2-binary**: PostgreSQL adapter for Python with binary optimizations

## [1.1.0] - 2025-06-14

### Added

#### Enhanced Vector Database Support
- **Azure AI Search Integration**: Complete integration with Microsoft's enterprise search service
  - Support for hybrid search (keyword + semantic + vector)
  - Enterprise-grade security and compliance features
  - Multi-modal data support (text, images, structured data)
  - Advanced filtering, faceting, and aggregation capabilities
  - High availability with 99.9% SLA
- **AWS Elasticsearch Integration**: Full support for Amazon OpenSearch Service
  - Mature Elasticsearch-based vector search capabilities
  - Advanced analytics and visualization with Kibana
  - Real-time processing and indexing
  - Multi-tenancy support for multiple applications
- **Vector Store Architecture**: Unified interface for all vector database implementations
  - Base abstract class for consistent API across all vector stores
  - Async/await support for high-performance operations
  - Comprehensive error handling and logging

#### Configuration Enhancements
- **Azure AI Search Configuration**: Environment variables and settings for Azure AI Search
- **AWS Elasticsearch Configuration**: Complete configuration support for AWS OpenSearch
- **Vector Database Selection**: Enhanced configuration system supporting all 6 vector databases

#### Documentation Improvements
- **Comprehensive Vector Database Guide**: Detailed explanations of all 6 supported vector databases
- **Decision Matrix**: Comparison table helping users choose the right vector database
- **Use Case Recommendations**: Specific guidance for different application scenarios
- **Benefits and Limitations**: Honest assessment of pros and cons for each vector database
- **Enterprise Considerations**: Guidance for enterprise-scale deployments

## [1.0.0] - 2025-06-14

### Added

#### Core Application
- **New RAG Embeddings Database Application**: Complete application for creating and managing embedding databases for RAG operations
- **Multi-Vector Database Support**: Integration with FAISS, Pinecone, and ChromaDB
- **Document Processing Pipeline**: Comprehensive document ingestion and processing system
- **Embedding Generation**: OpenAI and Azure OpenAI embedding generation with batch processing
- **Advanced Chunking Strategies**: Semantic, fixed-size, and sentence-aware chunking options

#### Configuration System
- **EmbeddingConfig Class**: 12-factor app configuration with environment variable support
- **Vector Database Configuration**: Support for FAISS, Pinecone, and ChromaDB settings
- **Rate Limiting Configuration**: Configurable rate limiting for API calls
- **Multi-Source Configuration**: Support for local, S3, and SharePoint document sources

#### Data Models
- **DocumentChunk Model**: Enhanced with embedding vector storage capabilities
- **EmbeddingBatch Model**: Batch processing support for efficient embedding generation
- **VectorSearchResult Model**: Search result representation with similarity scores
- **VectorDatabaseConfig Model**: Configuration for different vector database types
- **EmbeddingStats Model**: Comprehensive statistics tracking for processing operations

#### Services
- **DocumentService**: Core service for document processing and embedding generation
  - Multi-format document support (PDF, TXT, JSON, PPTX)
  - Parallel processing with configurable workers
  - Integration points for vector database storage
  - Progress tracking and error handling
- **EmbeddingClient**: Dedicated client for embedding operations
  - OpenAI and Azure OpenAI support
  - Batch processing capabilities
  - Fallback to mock embeddings for testing

#### Source Adapters
- **LocalSource**: Process documents from local file system
- **S3Source**: Direct integration with Amazon S3 buckets
- **SharePointSource**: Microsoft SharePoint document library support
- **Factory Pattern**: Pluggable source system for extensibility

#### Utilities
- **Rate Limiting**: Multiple strategies (fixed window, sliding window, token bucket, adaptive)
- **File Utilities**: JSONL processing, file sampling, and batch operations
- **Environment Configuration**: Robust environment variable handling
- **Azure Identity**: Support for Azure authentication and identity management

#### CLI Interface
- **create-embeddings**: Process documents and create embeddings
- **search**: Search for similar documents using embeddings
- **list-sources**: List available document sources
- **status**: Show system status and configuration
- **Preview Mode**: Dry-run functionality for testing configurations
- **Comprehensive Help**: Detailed help text and usage examples

#### Testing Infrastructure
- **Unit Tests**: Comprehensive test coverage for all core components
  - Configuration validation tests
  - Data model serialization/deserialization tests
  - Client functionality tests
  - Utility function tests
- **Integration Tests**: End-to-end workflow testing
  - Document processing integration tests
  - Vector database integration tests
  - Multi-source processing tests
- **Test Fixtures**: Reusable test data and mock objects
- **Async Test Support**: Infrastructure for testing async functionality

#### Documentation
- **Comprehensive README**: Complete usage guide with examples and non-technical explanations
  - **Embeddings Overview**: Simple explanation of embeddings and their role in GenAI
  - **RAG Process Visualization**: Mermaid diagrams showing how RAG systems work
  - **Chunking Best Practices**: Detailed guide to text chunking strategies
  - **Configuration Impact Analysis**: Real-world examples of configuration effects
  - **Use-Case Specific Recommendations**: Pre-configured settings for common scenarios
- **Mermaid Architecture Diagrams**: Visual representation of system architecture
- **CLI Documentation**: Detailed command-line interface documentation
- **API Documentation**: Code-level documentation for all classes and methods
- **Configuration Guide**: Environment variable and configuration options with business impact explanations

#### Project Structure
- **Clean Architecture**: Service-oriented design with clear separation of concerns
- **Dependency Injection**: Configuration-driven dependency management
- **Error Handling**: Comprehensive error handling and logging
- **Type Safety**: Full type hints and Pydantic model validation

### Technical Improvements

#### Performance
- **Parallel Processing**: Configurable worker pools for document processing
- **Batch Operations**: Efficient batch processing for embeddings and vector operations
- **Rate Limiting**: Smart API rate limiting to prevent quota exhaustion
- **Caching**: Configurable caching for embedding operations

#### Security
- **Environment Variable Configuration**: Secure handling of API keys and secrets
- **Input Validation**: Comprehensive input validation using Pydantic
- **Error Sanitization**: Safe error handling without exposing sensitive information

#### Scalability
- **Cloud Storage Integration**: Support for S3 and Azure Blob storage
- **Vector Database Scaling**: Support for both local (FAISS) and cloud (Pinecone) vector databases
- **Configurable Resource Limits**: Memory and processing limits for large document collections

### Development Tools

#### Code Quality
- **Black Formatting**: Consistent code formatting with Black
- **isort Import Sorting**: Organized import statements
- **MyPy Type Checking**: Static type checking for improved code quality
- **Flake8 Linting**: Code style and quality enforcement
- **Bandit Security Scanning**: Security vulnerability detection

#### Build System
- **pyproject.toml**: Modern Python packaging configuration
- **requirements.txt**: Comprehensive dependency management
- **Optional Dependencies**: Modular installation with cloud, dev, and test dependencies
- **Entry Points**: Clean CLI entry points for easy installation

### Migration from RAFT Toolkit

#### Adapted Components
- **Configuration System**: Adapted from RaftConfig to EmbeddingConfig
- **Document Processing**: Streamlined from Q&A generation to embedding generation
- **Source Adapters**: Preserved multi-source document ingestion capabilities
- **Rate Limiting**: Maintained robust rate limiting infrastructure
- **Parallel Processing**: Kept efficient parallel processing framework

#### Removed Components
- **Q&A Generation**: Removed RAFT-specific question and answer generation
- **Template System**: Removed prompt templates for completion models
- **Completion Models**: Removed chat and completion model integration
- **RAFT-specific Configuration**: Removed dataset generation specific settings

#### Enhanced Components
- **Vector Database Integration**: Added comprehensive vector database support
- **Embedding Focus**: Specialized all components for embedding operations
- **Search Capabilities**: Added similarity search functionality
- **Batch Processing**: Enhanced batch processing for embedding generation

### Dependencies

#### Core Dependencies
- **openai**: OpenAI API client for embedding generation
- **langchain**: Text processing and chunking utilities
- **pydantic**: Data validation and settings management
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support

#### Vector Database Dependencies
- **faiss-cpu**: High-performance similarity search
- **chromadb**: Open-source embedding database
- **pinecone-client**: Managed vector database service

#### Optional Dependencies
- **boto3**: AWS S3 integration
- **azure-storage-blob**: Azure Blob storage integration
- **msal**: Microsoft authentication library

### Known Issues
- Vector database integration requires additional implementation for production use
- Search functionality needs vector database-specific implementation
- Large document collections may require additional memory optimization

### Next Steps
- Implement actual vector database storage operations
- Add similarity search functionality
- Optimize memory usage for large document collections
- Add incremental document processing capabilities
- Implement vector database management operations (create, delete, update indexes)