"""
Unit tests for embedding client functionality.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Test both new and legacy APIs
from core.clients import (
    BaseEmbeddingProvider,
    EmbeddingProviderFactory,
    EmbeddingResult,
    OpenAIEmbeddingProvider,
    create_provider_from_config,
)
from core.clients.openai_client import (
    EmbeddingClient,
    build_langchain_embeddings,
    build_openai_client,
    create_embedding_client,
    is_azure,
)
from core.config import EmbeddingConfig, get_config


class TestAzureDetection:
    """Test cases for Azure OpenAI detection (legacy API)."""

    def test_is_azure_true_cases(self, monkeypatch):
        """Test cases where is_azure should return True."""
        test_cases = ["1", "true", "True", "TRUE", "yes", "YES", "Yes"]

        for value in test_cases:
            monkeypatch.setenv("AZURE_OPENAI_ENABLED", value)
            assert is_azure() is True, f"Failed for value: {value}"

    def test_is_azure_false_cases(self, monkeypatch):
        """Test cases where is_azure should return False."""
        test_cases = ["0", "false", "False", "FALSE", "no", "NO", "No", "", "invalid"]

        for value in test_cases:
            monkeypatch.setenv("AZURE_OPENAI_ENABLED", value)
            assert is_azure() is False, f"Failed for value: {value}"

    def test_is_azure_not_set(self, clean_environment):
        """Test is_azure when environment variable is not set."""
        assert is_azure() is False


class TestBuildOpenAIClient:
    """Test cases for building OpenAI client instances (legacy API)."""

    @patch("core.clients.openai_client.OpenAI")
    @patch("core.clients.openai_client.is_azure")
    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_standard(self, mock_set_env, mock_read_env, mock_is_azure, mock_openai):
        """Test building standard OpenAI client."""
        # Setup mocks
        mock_is_azure.return_value = False
        mock_read_env.return_value = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_ORGANIZATION": "test-org",
        }
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Call function
        result = build_openai_client()

        # Assertions
        assert result == mock_client
        mock_openai.assert_called_once_with(
            api_key="test-key",
            organization="test-org",
        )

    @patch("core.clients.openai_client.AzureOpenAI")
    @patch("core.clients.openai_client.is_azure")
    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_azure(self, mock_set_env, mock_read_env, mock_is_azure, mock_azure_openai):
        """Test building Azure OpenAI client."""
        # Setup mocks
        mock_is_azure.return_value = True
        mock_read_env.return_value = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
        }
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        # Call function
        result = build_openai_client()

        # Assertions
        assert result == mock_client
        mock_azure_openai.assert_called_once_with(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
        )


class TestEmbeddingProviderFactory:
    """Test cases for the new embedding provider factory."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = {
            "provider": "openai",
            "api_key": "test-key",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        }

        provider = EmbeddingProviderFactory.create_provider("openai", **config)

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.provider_name == "openai"
        assert provider.model_name == "text-embedding-3-small"

    def test_create_provider_from_config(self):
        """Test creating provider from EmbeddingConfig."""
        config = EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        provider = create_provider_from_config(config)

        assert isinstance(provider, BaseEmbeddingProvider)
        assert provider.provider_name == "openai"

    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            EmbeddingProviderFactory.create_provider("unsupported_provider")


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAI embedding provider."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )

    def test_provider_initialization(self, openai_provider):
        """Test provider initialization."""
        assert openai_provider.provider_name == "openai"
        assert openai_provider.model_name == "text-embedding-3-small"
        assert openai_provider.dimensions == 1536
        assert openai_provider.max_batch_size == 2048

    @patch("core.clients.openai_embedding_provider.OpenAI")
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, mock_openai_class, openai_provider):
        """Test successful embedding generation."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_response.usage = Mock(prompt_tokens=20, total_tokens=20)
        mock_client.embeddings.create.return_value = mock_response

        # Test
        texts = ["test text 1", "test text 2"]
        result = await openai_provider.generate_embeddings(texts)

        # Assertions
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 1536
        assert result.model == "text-embedding-3-small"
        assert result.usage["prompt_tokens"] == 20

    @patch("core.clients.openai_embedding_provider.OpenAI")
    @pytest.mark.asyncio
    async def test_generate_embeddings_fallback(self, mock_openai_class, openai_provider):
        """Test fallback to mock embeddings when API fails."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")

        # Test
        texts = ["test text 1", "test text 2"]
        result = await openai_provider.generate_embeddings(texts)

        # Assertions - should get mock embeddings
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 1536
        assert result.model == "text-embedding-3-small"

    def test_get_model_info(self, openai_provider):
        """Test getting model information."""
        info = openai_provider.get_model_info()

        assert info.name == "text-embedding-3-small"
        assert info.dimensions == 1536
        assert info.max_tokens == 8191
        assert info.provider == "openai"


class TestEmbeddingClient:
    """Test cases for legacy embedding client."""

    @pytest.fixture
    def embedding_client(self, mock_openai_client):
        """Create embedding client for testing."""
        with patch("core.clients.openai_client.build_openai_client", return_value=mock_openai_client):
            return EmbeddingClient()

    def test_client_initialization(self, embedding_client):
        """Test client initialization."""
        assert embedding_client is not None
        assert hasattr(embedding_client, "client")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, embedding_client, mock_openai_client):
        """Test embedding generation with legacy client."""
        # Setup
        texts = ["test text 1", "test text 2"]

        # Test
        embeddings = await embedding_client.generate_embeddings(texts)

        # Assertions
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        mock_openai_client.embeddings.create.assert_called_once()

    def test_get_model_info(self, embedding_client):
        """Test getting model info from legacy client."""
        info = embedding_client.get_model_info()

        assert isinstance(info, dict)
        assert "model" in info
        assert "dimensions" in info


class TestBuildLangChainEmbeddings:
    """Test cases for LangChain embeddings builder."""

    @patch("core.clients.openai_client.OpenAIEmbeddings")
    @patch("core.clients.openai_client.is_azure")
    def test_build_openai_embeddings(self, mock_is_azure, mock_openai_embeddings):
        """Test building OpenAI embeddings for LangChain."""
        mock_is_azure.return_value = False
        mock_embeddings = Mock()
        mock_openai_embeddings.return_value = mock_embeddings

        result = build_langchain_embeddings()

        assert result == mock_embeddings
        mock_openai_embeddings.assert_called_once()

    @patch("core.clients.openai_client.AzureOpenAIEmbeddings")
    @patch("core.clients.openai_client.is_azure")
    def test_build_azure_embeddings(self, mock_is_azure, mock_azure_embeddings):
        """Test building Azure OpenAI embeddings for LangChain."""
        mock_is_azure.return_value = True
        mock_embeddings = Mock()
        mock_azure_embeddings.return_value = mock_embeddings

        result = build_langchain_embeddings()

        assert result == mock_embeddings
        mock_azure_embeddings.assert_called_once()


class TestCreateEmbeddingClient:
    """Test cases for embedding client factory function."""

    @patch("core.clients.openai_client.EmbeddingClient")
    def test_create_embedding_client(self, mock_embedding_client):
        """Test creating embedding client."""
        mock_client = Mock()
        mock_embedding_client.return_value = mock_client

        result = create_embedding_client()

        assert result == mock_client
        mock_embedding_client.assert_called_once()


class TestProviderIntegration:
    """Integration tests for provider system."""

    @pytest.mark.asyncio
    async def test_provider_workflow(self):
        """Test complete provider workflow."""
        # Create config
        config = EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        # Create provider
        provider = create_provider_from_config(config)

        # Test basic properties
        assert provider.provider_name == "openai"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536

        # Test model info
        info = provider.get_model_info()
        assert info.name == "text-embedding-3-small"
        assert info.provider == "openai"

    def test_provider_error_handling(self):
        """Test provider error handling."""
        # Test invalid provider
        with pytest.raises(ValueError):
            EmbeddingProviderFactory.create_provider("invalid_provider")

        # Test missing required parameters
        with pytest.raises((ValueError, TypeError)):
            OpenAIEmbeddingProvider()  # Missing required api_key


class TestMockProviders:
    """Test cases for mock providers used in testing."""

    @pytest.mark.asyncio
    async def test_mock_provider_functionality(self, mock_embedding_provider):
        """Test mock provider works correctly."""
        texts = ["test text 1", "test text 2"]
        result = await mock_embedding_provider.generate_embeddings(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 1536
        assert result.model == "mock-model"

    def test_mock_provider_properties(self, mock_embedding_provider):
        """Test mock provider properties."""
        assert mock_embedding_provider.provider_name == "mock"
        assert mock_embedding_provider.model_name == "mock-model"
        assert mock_embedding_provider.dimensions == 1536
