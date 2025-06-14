"""
Unit tests for the EmbeddingConfig class and configuration management.
"""
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.config import EmbeddingConfig, get_config


class TestEmbeddingConfig:
    """Test cases for the EmbeddingConfig class."""
    
    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = EmbeddingConfig()
        
        assert config.datapath == Path(".")
        assert config.output == "./"
        assert config.output_format == "jsonl"
        assert config.source_type == "local"
        assert config.chunk_size == 512
        assert config.doctype == "pdf"
        assert config.chunking_strategy == "semantic"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dimensions == 1536
        assert config.batch_size_embeddings == 100
        assert config.vector_db_type == "faiss"
        assert config.workers == 1
        assert config.embed_workers == 1
        assert config.pace is True
        assert config.rate_limit_enabled is False
        assert config.use_azure_identity is False
        assert config.azure_openai_enabled is False
    
    def test_config_with_custom_values(self):
        """Test creating a config with custom values."""
        config = EmbeddingConfig(
            datapath=Path("/custom/path"),
            output="./custom_output",
            chunk_size=1024,
            embedding_model="text-embedding-ada-002",
            vector_db_type="pinecone",
            workers=4,
            rate_limit_enabled=True
        )
        
        assert config.datapath == Path("/custom/path")
        assert config.output == "./custom_output"
        assert config.chunk_size == 1024
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.vector_db_type == "pinecone"
        assert config.workers == 4
        assert config.rate_limit_enabled is True
    
    def test_from_env_basic(self, clean_environment, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("EMBEDDING_DATAPATH", "/env/test/path")
        monkeypatch.setenv("EMBEDDING_OUTPUT", "./env_output")
        monkeypatch.setenv("EMBEDDING_CHUNK_SIZE", "256")
        monkeypatch.setenv("EMBEDDING_DOCTYPE", "txt")
        monkeypatch.setenv("OPENAI_API_KEY", "env_test_key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("VECTOR_DB_TYPE", "chroma")
        monkeypatch.setenv("EMBEDDING_WORKERS", "2")
        monkeypatch.setenv("EMBEDDING_EMBED_WORKERS", "3")
        monkeypatch.setenv("EMBEDDING_PACE", "false")
        
        config = EmbeddingConfig.from_env()
        
        assert config.datapath == Path("/env/test/path")
        assert config.output == "./env_output"
        assert config.chunk_size == 256
        assert config.doctype == "txt"
        assert config.openai_key == "env_test_key"
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.vector_db_type == "chroma"
        assert config.workers == 2
        assert config.embed_workers == 3
        assert config.pace is False
    
    def test_from_env_with_json_fields(self, clean_environment, monkeypatch):
        """Test loading config with JSON fields from environment."""
        credentials = {"username": "test", "password": "secret"}
        include_patterns = ["*.pdf", "*.txt"]
        exclude_patterns = ["temp/*", "*.tmp"]
        chunking_params = {"overlap": 50, "strategy": "custom"}
        vector_db_config = {"host": "localhost", "port": 8000}
        
        monkeypatch.setenv("EMBEDDING_SOURCE_CREDENTIALS", json.dumps(credentials))
        monkeypatch.setenv("EMBEDDING_SOURCE_INCLUDE_PATTERNS", json.dumps(include_patterns))
        monkeypatch.setenv("EMBEDDING_SOURCE_EXCLUDE_PATTERNS", json.dumps(exclude_patterns))
        monkeypatch.setenv("EMBEDDING_CHUNKING_PARAMS", json.dumps(chunking_params))
        monkeypatch.setenv("VECTOR_DB_CONFIG", json.dumps(vector_db_config))
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        config = EmbeddingConfig.from_env()
        
        assert config.source_credentials == credentials
        assert config.source_include_patterns == include_patterns
        assert config.source_exclude_patterns == exclude_patterns
        assert config.chunking_params == chunking_params
        assert config.vector_db_config == vector_db_config
    
    def test_from_env_invalid_json_handling(self, clean_environment, monkeypatch, caplog):
        """Test handling of invalid JSON in environment variables."""
        monkeypatch.setenv("EMBEDDING_SOURCE_CREDENTIALS", "invalid_json{")
        monkeypatch.setenv("EMBEDDING_SOURCE_INCLUDE_PATTERNS", "not_json")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        config = EmbeddingConfig.from_env()
        
        # Should use default values when JSON parsing fails
        assert config.source_credentials == {}
        assert config.source_include_patterns == ['**/*']
        
        # Should log warnings for invalid JSON
        assert "Failed to parse EMBEDDING_SOURCE_CREDENTIALS" in caplog.text
        assert "Failed to parse EMBEDDING_SOURCE_INCLUDE_PATTERNS" in caplog.text
    
    def test_from_env_rate_limiting_config(self, clean_environment, monkeypatch):
        """Test loading rate limiting configuration from environment."""
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_STRATEGY", "token_bucket")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE", "100")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_TOKENS_PER_MINUTE", "5000")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_MAX_BURST", "20")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_MAX_RETRIES", "5")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_BASE_DELAY", "2.0")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        config = EmbeddingConfig.from_env()
        
        assert config.rate_limit_enabled is True
        assert config.rate_limit_strategy == "token_bucket"
        assert config.rate_limit_requests_per_minute == 100
        assert config.rate_limit_tokens_per_minute == 5000
        assert config.rate_limit_max_burst == 20
        assert config.rate_limit_max_retries == 5
        assert config.rate_limit_base_delay == 2.0
    
    def test_from_env_azure_config(self, clean_environment, monkeypatch):
        """Test loading Azure-specific configuration from environment."""
        monkeypatch.setenv("EMBEDDING_USE_AZURE_IDENTITY", "true")
        monkeypatch.setenv("AZURE_OPENAI_ENABLED", "1")
        
        config = EmbeddingConfig.from_env()
        
        assert config.use_azure_identity is True
        assert config.azure_openai_enabled is True
    
    def test_from_env_boolean_parsing(self, clean_environment, monkeypatch):
        """Test parsing of boolean environment variables."""
        test_cases = [
            ("true", True), ("1", True), ("yes", True),
            ("false", False), ("0", False), ("no", False),
            ("", False), ("invalid", False)
        ]
        
        for env_value, expected in test_cases:
            monkeypatch.setenv("EMBEDDING_PACE", env_value)
            monkeypatch.setenv("OPENAI_API_KEY", "test_key")
            config = EmbeddingConfig.from_env()
            assert config.pace == expected, f"Failed for env_value: {env_value}"
    
    def test_from_env_with_dotenv_file(self, clean_environment):
        """Test loading config from a .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("EMBEDDING_CHUNK_SIZE=2048\n")
            f.write("EMBEDDING_MODEL=custom-model\n")
            f.write("OPENAI_API_KEY=file_test_key\n")
            f.write("VECTOR_DB_TYPE=pinecone\n")
            env_file = f.name
        
        try:
            config = EmbeddingConfig.from_env(env_file)
            assert config.chunk_size == 2048
            assert config.embedding_model == "custom-model"
            assert config.openai_key == "file_test_key"
            assert config.vector_db_type == "pinecone"
        finally:
            os.unlink(env_file)
    
    def test_validate_success(self, sample_config):
        """Test successful validation of a valid config."""
        # Should not raise any exceptions
        sample_config.validate()
    
    def test_validate_invalid_source_type(self, sample_config):
        """Test validation failure for invalid source type."""
        sample_config.source_type = "invalid_source"
        
        with pytest.raises(ValueError, match="Invalid source type: invalid_source"):
            sample_config.validate()
    
    def test_validate_missing_source_uri_for_remote_sources(self, sample_config):
        """Test validation failure for missing source_uri with remote sources."""
        sample_config.source_type = "s3"
        sample_config.source_uri = None
        
        with pytest.raises(ValueError, match="source_uri is required for source type: s3"):
            sample_config.validate()
    
    def test_validate_invalid_doctype(self, sample_config):
        """Test validation failure for invalid document type."""
        sample_config.doctype = "invalid_type"
        
        with pytest.raises(ValueError, match="Invalid doctype: invalid_type"):
            sample_config.validate()
    
    def test_validate_invalid_output_format(self, sample_config):
        """Test validation failure for invalid output format."""
        sample_config.output_format = "xml"
        
        with pytest.raises(ValueError, match="Invalid output format: xml"):
            sample_config.validate()
    
    def test_validate_invalid_chunking_strategy(self, sample_config):
        """Test validation failure for invalid chunking strategy."""
        sample_config.chunking_strategy = "invalid_strategy"
        
        with pytest.raises(ValueError, match="Invalid chunking strategy: invalid_strategy"):
            sample_config.validate()
    
    def test_validate_invalid_vector_db_type(self, sample_config):
        """Test validation failure for invalid vector database type."""
        sample_config.vector_db_type = "invalid_db"
        
        with pytest.raises(ValueError, match="Invalid vector database type: invalid_db"):
            sample_config.validate()
    
    def test_validate_pinecone_requirements(self, sample_config):
        """Test validation of Pinecone-specific requirements."""
        sample_config.vector_db_type = "pinecone"
        sample_config.pinecone_api_key = None
        sample_config.pinecone_environment = None
        
        with pytest.raises(ValueError, match="Pinecone API key is required"):
            sample_config.validate()
        
        sample_config.pinecone_api_key = "test_key"
        with pytest.raises(ValueError, match="Pinecone environment is required"):
            sample_config.validate()
        
        sample_config.pinecone_environment = "test-env"
        sample_config.validate()  # Should pass now
    
    def test_validate_numeric_field_requirements(self, sample_config):
        """Test validation of numeric field requirements."""
        # Test negative values
        sample_config.source_max_file_size = -1
        with pytest.raises(ValueError, match="source_max_file_size must be positive"):
            sample_config.validate()
        
        sample_config.source_max_file_size = 1024
        sample_config.source_batch_size = 0
        with pytest.raises(ValueError, match="source_batch_size must be positive"):
            sample_config.validate()
        
        sample_config.source_batch_size = 10
        sample_config.embedding_dimensions = -100
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            sample_config.validate()
        
        sample_config.embedding_dimensions = 1536
        sample_config.batch_size_embeddings = 0
        with pytest.raises(ValueError, match="batch_size_embeddings must be positive"):
            sample_config.validate()
    
    def test_validate_missing_openai_key(self, sample_config):
        """Test validation failure for missing OpenAI API key."""
        sample_config.openai_key = None
        sample_config.use_azure_identity = False
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            sample_config.validate()
    
    def test_validate_demo_mode_allowed(self, sample_config):
        """Test that demo mode with special key is allowed."""
        sample_config.openai_key = "demo_key_for_testing"
        sample_config.validate()  # Should pass
    
    def test_validate_azure_identity_mode(self, sample_config):
        """Test validation with Azure identity mode."""
        sample_config.openai_key = None
        sample_config.use_azure_identity = True
        sample_config.validate()  # Should pass without OpenAI key
    
    def test_validate_datapath_existence_local_source(self):
        """Test validation of datapath existence for local sources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with existing path
            config = EmbeddingConfig(
                datapath=Path(temp_dir),
                source_type="local",
                openai_key="test_key"
            )
            config.validate()  # Should pass
        
        # Test with non-existing path (not current directory)
        config = EmbeddingConfig(
            datapath=Path("/non/existent/path"),
            source_type="local",
            openai_key="test_key"
        )
        with pytest.raises(ValueError, match="Data path does not exist"):
            config.validate()
        
        # Test with current directory (should pass)
        config = EmbeddingConfig(
            datapath=Path("."),
            source_type="local",
            openai_key="test_key"
        )
        config.validate()  # Should pass
    
    @pytest.mark.parametrize("source_type", ["local", "s3", "sharepoint"])
    def test_validate_valid_source_types(self, sample_config, source_type):
        """Test validation with all valid source types."""
        sample_config.source_type = source_type
        if source_type != "local":
            sample_config.source_uri = f"{source_type}://test/path"
        sample_config.validate()
    
    @pytest.mark.parametrize("doctype", ["pdf", "txt", "json", "api", "pptx"])
    def test_validate_valid_doctypes(self, sample_config, doctype):
        """Test validation with all valid document types."""
        sample_config.doctype = doctype
        sample_config.validate()
    
    @pytest.mark.parametrize("output_format", ["jsonl", "parquet"])
    def test_validate_valid_output_formats(self, sample_config, output_format):
        """Test validation with all valid output formats."""
        sample_config.output_format = output_format
        sample_config.validate()
    
    @pytest.mark.parametrize("chunking_strategy", ["semantic", "fixed", "sentence"])
    def test_validate_valid_chunking_strategies(self, sample_config, chunking_strategy):
        """Test validation with all valid chunking strategies."""
        sample_config.chunking_strategy = chunking_strategy
        sample_config.validate()
    
    @pytest.mark.parametrize("vector_db_type", ["faiss", "pinecone", "chroma"])
    def test_validate_valid_vector_db_types(self, sample_config, vector_db_type):
        """Test validation with all valid vector database types."""
        sample_config.vector_db_type = vector_db_type
        if vector_db_type == "pinecone":
            sample_config.pinecone_api_key = "test_key"
            sample_config.pinecone_environment = "test-env"
        sample_config.validate()


class TestGetConfig:
    """Test cases for the get_config helper function."""
    
    def test_get_config_success(self, clean_environment, monkeypatch):
        """Test successful config creation and validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        config = get_config()
        assert isinstance(config, EmbeddingConfig)
        assert config.openai_key == "test_key"
    
    def test_get_config_validation_failure(self, clean_environment, monkeypatch):
        """Test that get_config raises validation errors."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("EMBEDDING_DOCTYPE", "invalid_type")
        
        with pytest.raises(ValueError, match="Invalid doctype"):
            get_config()
    
    def test_get_config_with_env_file(self, clean_environment):
        """Test get_config with a specific env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=file_test_key\n")
            f.write("EMBEDDING_CHUNK_SIZE=1024\n")
            env_file = f.name
        
        try:
            config = get_config(env_file)
            assert config.openai_key == "file_test_key" 
            assert config.chunk_size == 1024
        finally:
            os.unlink(env_file)


class TestConfigIntegration:
    """Integration tests for configuration management."""
    
    def test_full_config_workflow(self, clean_environment, monkeypatch):
        """Test a complete configuration workflow."""
        # Set up comprehensive environment
        monkeypatch.setenv("EMBEDDING_DATAPATH", "/test/data")
        monkeypatch.setenv("EMBEDDING_OUTPUT", "./test_output")
        monkeypatch.setenv("EMBEDDING_OUTPUT_FORMAT", "parquet")
        monkeypatch.setenv("EMBEDDING_SOURCE_TYPE", "s3")
        monkeypatch.setenv("EMBEDDING_SOURCE_URI", "s3://test-bucket/data")
        monkeypatch.setenv("EMBEDDING_CHUNK_SIZE", "1024")
        monkeypatch.setenv("EMBEDDING_DOCTYPE", "pdf")
        monkeypatch.setenv("EMBEDDING_CHUNKING_STRATEGY", "sentence")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
        monkeypatch.setenv("VECTOR_DB_TYPE", "pinecone")
        monkeypatch.setenv("PINECONE_API_KEY", "test_pinecone_key")
        monkeypatch.setenv("PINECONE_ENVIRONMENT", "test-env")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
        
        config = get_config()
        
        # Verify all settings were loaded correctly
        assert config.datapath == Path("/test/data")
        assert config.output == "./test_output"
        assert config.output_format == "parquet"
        assert config.source_type == "s3"
        assert config.source_uri == "s3://test-bucket/data"
        assert config.chunk_size == 1024
        assert config.doctype == "pdf"
        assert config.chunking_strategy == "sentence"
        assert config.openai_key == "test_openai_key"
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.embedding_dimensions == 1536
        assert config.vector_db_type == "pinecone"
        assert config.pinecone_api_key == "test_pinecone_key"
        assert config.pinecone_environment == "test-env"
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests_per_minute == 60
    
    def test_config_precedence_env_over_defaults(self, clean_environment, monkeypatch):
        """Test that environment variables take precedence over defaults."""
        monkeypatch.setenv("EMBEDDING_CHUNK_SIZE", "2048")
        monkeypatch.setenv("OPENAI_API_KEY", "env_key")
        
        config = EmbeddingConfig.from_env()
        
        # Environment values should override defaults
        assert config.chunk_size == 2048  # Not default 512
        assert config.openai_key == "env_key"
        
        # Non-overridden values should remain defaults
        assert config.doctype == "pdf"  # Default value
        assert config.output_format == "jsonl"  # Default value
    
    @patch('core.config.load_dotenv')
    def test_load_dotenv_called_correctly(self, mock_load_dotenv, clean_environment):
        """Test that load_dotenv is called with correct parameters."""
        # Test without env_file
        EmbeddingConfig.from_env()
        mock_load_dotenv.assert_called_with()
        
        # Test with env_file
        mock_load_dotenv.reset_mock()
        EmbeddingConfig.from_env("/custom/.env")
        mock_load_dotenv.assert_called_with("/custom/.env")
    
    def test_missing_required_dependencies_handling(self):
        """Test handling when dotenv is not available."""
        # This test verifies the fallback when python-dotenv is not available
        # The fallback load_dotenv function should be a no-op
        with patch.dict('sys.modules', {'dotenv': None}):
            # Should not raise ImportError
            config = EmbeddingConfig()
            assert config.chunk_size == 512  # Should use defaults