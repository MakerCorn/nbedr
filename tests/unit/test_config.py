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
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.max_workers == 1
        assert config.rate_limit_enabled is False
        assert config.chunk_size == 512
        assert config.chunking_strategy == "semantic"

    def test_config_with_custom_values(self):
        """Test creating a config with custom values."""
        config = EmbeddingConfig(
            provider="azure_openai",
            model="text-embedding-3-large",
            dimensions=3072,
            api_key="custom-key",
            batch_size=50,
            max_workers=4,
            rate_limit_enabled=True,
            chunk_size=1024,
            chunking_strategy="fixed",
        )

        assert config.provider == "azure_openai"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 3072
        assert config.api_key == "custom-key"
        assert config.batch_size == 50
        assert config.max_workers == 4
        assert config.rate_limit_enabled is True
        assert config.chunk_size == 1024
        assert config.chunking_strategy == "fixed"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid provider
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="invalid_provider")

        # Test invalid dimensions
        with pytest.raises(ValueError):
            EmbeddingConfig(dimensions=0)

        # Test invalid batch size
        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)

        # Test invalid max workers
        with pytest.raises(ValueError):
            EmbeddingConfig(max_workers=0)

    def test_config_from_environment(self, monkeypatch):
        """Test creating config from environment variables."""
        # Set environment variables
        monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        monkeypatch.setenv("BATCH_SIZE", "25")
        monkeypatch.setenv("MAX_WORKERS", "8")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")

        config = EmbeddingConfig.from_env()

        assert config.provider == "azure_openai"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 3072
        assert config.api_key == "env-api-key"
        assert config.batch_size == 25
        assert config.max_workers == 8
        assert config.rate_limit_enabled is True

    def test_config_serialization(self):
        """Test config serialization to dict and JSON."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=1536,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "text-embedding-3-small"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["dimensions"] == 1536

        # Test to_json
        config_json = config.to_json()
        assert isinstance(config_json, str)
        parsed = json.loads(config_json)
        assert parsed["provider"] == "openai"
        assert parsed["model"] == "text-embedding-3-small"

    def test_config_deserialization(self):
        """Test config deserialization from dict and JSON."""
        config_dict = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test-key",
            "dimensions": 1536,
            "batch_size": 50,
        }

        # Test from_dict
        config = EmbeddingConfig.from_dict(config_dict)
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.api_key == "test-key"
        assert config.dimensions == 1536
        assert config.batch_size == 50

        # Test from_json
        config_json = json.dumps(config_dict)
        config2 = EmbeddingConfig.from_json(config_json)
        assert config2.provider == "openai"
        assert config2.model == "text-embedding-3-small"

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        config2 = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        config3 = EmbeddingConfig(
            provider="azure_openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        assert config1 == config2
        assert config1 != config3

    def test_config_copy(self):
        """Test config copying and modification."""
        original = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        # Test copy
        copy_config = original.copy()
        assert copy_config == original
        assert copy_config is not original

        # Test copy with modifications
        modified = original.copy(provider="azure_openai", model="text-embedding-3-large")
        assert modified.provider == "azure_openai"
        assert modified.model == "text-embedding-3-large"
        assert modified.api_key == "test-key"  # Unchanged
        assert modified != original

    def test_config_validation_edge_cases(self):
        """Test edge cases in config validation."""
        # Test empty API key
        with pytest.raises(ValueError):
            EmbeddingConfig(api_key="")

        # Test very large batch size
        config = EmbeddingConfig(batch_size=10000)
        assert config.batch_size == 10000

        # Test very large dimensions
        config = EmbeddingConfig(dimensions=10000)
        assert config.dimensions == 10000

    def test_config_provider_specific_validation(self):
        """Test provider-specific validation."""
        # OpenAI config
        openai_config = EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
        )
        assert openai_config.provider == "openai"

        # Azure OpenAI config
        azure_config = EmbeddingConfig(
            provider="azure_openai",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
        )
        assert azure_config.provider == "azure_openai"

        # AWS Bedrock config
        bedrock_config = EmbeddingConfig(
            provider="aws_bedrock",
            aws_region="us-east-1",
            model="amazon.titan-embed-text-v1",
        )
        assert bedrock_config.provider == "aws_bedrock"


class TestGetConfig:
    """Test cases for the get_config function."""

    def test_get_config_default(self):
        """Test getting default config."""
        config = get_config()
        assert isinstance(config, EmbeddingConfig)
        assert config.provider == "openai"

    def test_get_config_with_overrides(self):
        """Test getting config with parameter overrides."""
        config = get_config(
            provider="azure_openai",
            model="text-embedding-3-large",
            batch_size=25,
        )
        assert config.provider == "azure_openai"
        assert config.model == "text-embedding-3-large"
        assert config.batch_size == 25

    def test_get_config_from_file(self):
        """Test getting config from file."""
        config_data = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "file-api-key",
            "dimensions": 1536,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = get_config(config_file=config_file)
            assert config.provider == "openai"
            assert config.model == "text-embedding-3-small"
            assert config.api_key == "file-api-key"
        finally:
            os.unlink(config_file)

    def test_get_config_precedence(self, monkeypatch):
        """Test config precedence: file < env < parameters."""
        # Set environment variable
        monkeypatch.setenv("EMBEDDING_PROVIDER", "env_provider")
        monkeypatch.setenv("EMBEDDING_MODEL", "env_model")

        # Create config file
        config_data = {
            "provider": "file_provider",
            "model": "file_model",
            "api_key": "file_key",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Test precedence: parameter > env > file
            config = get_config(
                config_file=config_file,
                provider="param_provider",  # Should win
                # model not specified, should come from env
                # api_key not specified, should come from file
            )

            assert config.provider == "param_provider"  # Parameter wins
            assert config.model == "env_model"  # Environment wins over file
            assert config.api_key == "file_key"  # File provides missing value
        finally:
            os.unlink(config_file)


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
        # Test missing required fields
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="openai")  # Missing API key

        # Test invalid combinations
        with pytest.raises(ValueError):
            EmbeddingConfig(
                provider="azure_openai",
                api_key="test-key",
                # Missing azure_endpoint for Azure
            )

    def test_config_backwards_compatibility(self):
        """Test backwards compatibility with old config format."""
        # Test that old parameter names still work
        config = EmbeddingConfig(
            embedding_model="text-embedding-3-small",  # Old name
            embedding_dimensions=1536,  # Old name
            batch_size_embeddings=50,  # Old name
        )

        # Should map to new names
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 50
