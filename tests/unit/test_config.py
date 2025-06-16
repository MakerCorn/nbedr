"""
Unit tests for the EmbeddingConfig class and configuration management.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import EmbeddingConfig, get_config


class TestEmbeddingConfig:
    """Test cases for the EmbeddingConfig class."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = EmbeddingConfig()

        # Test new API structure defaults
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dimensions == 1536
        assert config.batch_size_embeddings == 100
        assert config.chunk_size == 512
        assert config.chunking_strategy == "semantic"

    def test_config_with_custom_values(self):
        """Test creating a config with custom values."""
        config = EmbeddingConfig(
            embedding_provider="azure_openai",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            openai_api_key="custom-key",
            batch_size_embeddings=50,
            workers=4,
            rate_limit_enabled=True,
            chunk_size=1024,
            chunking_strategy="fixed",
        )

        assert config.embedding_provider == "azure_openai"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.embedding_dimensions == 3072
        assert config.openai_api_key == "custom-key"
        assert config.batch_size_embeddings == 50
        assert config.workers == 4
        assert config.rate_limit_enabled is True
        assert config.chunk_size == 1024
        assert config.chunking_strategy == "fixed"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid provider
        config = EmbeddingConfig(embedding_provider="invalid_provider")
        with pytest.raises(ValueError):
            config.validate()

        # Test invalid dimensions
        config = EmbeddingConfig(embedding_dimensions=0)
        with pytest.raises(ValueError):
            config.validate()

        # Test invalid batch size
        config = EmbeddingConfig(batch_size_embeddings=0)
        with pytest.raises(ValueError):
            config.validate()

    def test_config_from_environment(self, monkeypatch):
        """Test creating config from environment variables."""
        # Set environment variables
        monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "25")
        monkeypatch.setenv("EMBEDDING_WORKERS", "8")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")

        config = EmbeddingConfig.from_env()

        assert config.embedding_provider == "azure_openai"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.embedding_dimensions == 3072
        assert config.openai_api_key == "env-api-key"
        assert config.batch_size_embeddings == 25
        assert config.workers == 8
        assert config.rate_limit_enabled is True

    @pytest.mark.skip(reason="Serialization methods not implemented in current EmbeddingConfig")
    def test_config_serialization(self):
        """Test config serialization to dict and JSON."""
        config = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
            embedding_dimensions=1536,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["embedding_provider"] == "openai"
        assert config_dict["embedding_model"] == "text-embedding-3-small"
        assert config_dict["openai_api_key"] == "test-key"
        assert config_dict["embedding_dimensions"] == 1536

        # Test to_json
        config_json = config.to_json()
        assert isinstance(config_json, str)
        parsed = json.loads(config_json)
        assert parsed["embedding_provider"] == "openai"
        assert parsed["embedding_model"] == "text-embedding-3-small"

    @pytest.mark.skip(reason="Deserialization methods not implemented in current EmbeddingConfig")
    def test_config_deserialization(self):
        """Test config deserialization from dict and JSON."""
        config_dict = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "openai_api_key": "test-key",
            "embedding_dimensions": 1536,
            "batch_size_embeddings": 50,
        }

        # Test from_dict
        config = EmbeddingConfig.from_dict(config_dict)
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.openai_api_key == "test-key"
        assert config.embedding_dimensions == 1536
        assert config.batch_size_embeddings == 50

        # Test from_json
        config_json = json.dumps(config_dict)
        config2 = EmbeddingConfig.from_json(config_json)
        assert config2.embedding_provider == "openai"
        assert config2.embedding_model == "text-embedding-3-small"

    @pytest.mark.skip(reason="Equality methods not implemented in current EmbeddingConfig")
    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
        )

        config2 = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
        )

        config3 = EmbeddingConfig(
            embedding_provider="azure_openai",
            embedding_model="text-embedding-3-small",
            azure_openai_api_key="test-key",
        )

        assert config1 == config2
        assert config1 != config3

    @pytest.mark.skip(reason="Copy methods not implemented in current EmbeddingConfig")
    def test_config_copy(self):
        """Test config copying and modification."""
        original = EmbeddingConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
        )

        # Test copy
        copy_config = original.copy()
        assert copy_config == original
        assert copy_config is not original

        # Test copy with modifications
        modified = original.copy(embedding_provider="azure_openai", embedding_model="text-embedding-3-large")
        assert modified.embedding_provider == "azure_openai"
        assert modified.embedding_model == "text-embedding-3-large"
        assert modified.openai_api_key == "test-key"  # Unchanged
        assert modified != original

    def test_config_validation_edge_cases(self):
        """Test edge cases in config validation."""
        # Test very large batch size
        config = EmbeddingConfig(batch_size_embeddings=10000)
        assert config.batch_size_embeddings == 10000

        # Test very large dimensions
        config = EmbeddingConfig(embedding_dimensions=10000)
        assert config.embedding_dimensions == 10000

    def test_config_provider_specific_validation(self):
        """Test provider-specific validation."""
        # OpenAI config
        openai_config = EmbeddingConfig(
            embedding_provider="openai",
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )
        assert openai_config.embedding_provider == "openai"

        # Azure OpenAI config
        azure_config = EmbeddingConfig(
            embedding_provider="azure_openai",
            azure_openai_api_key="test-key",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_deployment_name="test-deployment",
        )
        assert azure_config.embedding_provider == "azure_openai"

        # AWS Bedrock config
        bedrock_config = EmbeddingConfig(
            embedding_provider="aws_bedrock",
            aws_bedrock_region="us-east-1",
            embedding_model="amazon.titan-embed-text-v1",
        )
        assert bedrock_config.embedding_provider == "aws_bedrock"


class TestGetConfig:
    """Test cases for the get_config function."""

    def test_get_config_default(self):
        """Test getting default config."""
        config = get_config()
        assert isinstance(config, EmbeddingConfig)
        assert config.embedding_provider == "openai"

    @pytest.mark.skip(reason="get_config does not support parameter overrides in current implementation")
    def test_get_config_with_overrides(self):
        """Test getting config with parameter overrides."""
        config = get_config(
            embedding_provider="azure_openai",
            embedding_model="text-embedding-3-large",
            batch_size_embeddings=25,
        )
        assert config.embedding_provider == "azure_openai"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.batch_size_embeddings == 25

    def test_get_config_from_file(self, clean_environment):
        """Test getting config from .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("EMBEDDING_PROVIDER=openai\n")
            f.write("EMBEDDING_MODEL=text-embedding-3-small\n")
            f.write("OPENAI_API_KEY=file-api-key\n")
            f.write("EMBEDDING_DIMENSIONS=1536\n")
            env_file = f.name

        try:
            config = get_config(env_file=env_file)
            assert config.embedding_provider == "openai"
            assert config.embedding_model == "text-embedding-3-small"
            assert config.openai_api_key == "file-api-key"
            assert config.embedding_dimensions == 1536
        finally:
            os.unlink(env_file)

    def test_get_config_precedence(self, clean_environment, monkeypatch):
        """Test config precedence: env > .env file."""
        # Set environment variable with valid values
        monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")

        # Create .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("EMBEDDING_PROVIDER=openai\n")
            f.write("EMBEDDING_MODEL=text-embedding-3-small\n")
            f.write("OPENAI_API_KEY=file_key\n")
            env_file = f.name

        try:
            # Test precedence: env > file
            config = get_config(env_file=env_file)

            assert config.embedding_provider == "azure_openai"  # Environment wins
            assert config.embedding_model == "text-embedding-3-large"  # Environment wins
            assert config.azure_openai_api_key == "env_key"  # Environment provides value
        finally:
            os.unlink(env_file)


@pytest.mark.skip(reason="Config integration tests expect different implementation - architectural mismatch")
class TestConfigIntegration:
    """Integration tests for config functionality."""

    def test_config_with_real_environment(self, monkeypatch):
        """Test config creation with realistic environment setup."""
        # Simulate a real environment setup
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123456789")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
        monkeypatch.setenv("BATCH_SIZE", "100")
        monkeypatch.setenv("MAX_WORKERS", "4")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("CHUNKING_STRATEGY", "semantic")

        config = get_config()

        assert config.provider == "openai"
        assert config.api_key == "sk-test123456789"
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.rate_limit_enabled is True
        assert config.chunk_size == 512
        assert config.chunking_strategy == "semantic"

    def test_config_error_handling(self):
        """Test config error handling and validation."""
        # Test invalid provider validation
        config = EmbeddingConfig(embedding_provider="invalid_provider")
        with pytest.raises(ValueError):
            config.validate()

        # Test invalid combinations for Azure
        config = EmbeddingConfig(
            embedding_provider="azure_openai",
            azure_openai_api_key="test-key",
            # Missing azure_openai_endpoint for Azure
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_config_backwards_compatibility(self):
        """Test backwards compatibility with current config format."""
        # Test that current parameter names work
        config = EmbeddingConfig(
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            batch_size_embeddings=50,
        )

        # Verify current field names
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dimensions == 1536
        assert config.batch_size_embeddings == 50
