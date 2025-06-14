"""
Unit tests for embedding client functionality.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from core.clients.openai_client import (
    EmbeddingClient,
    build_langchain_embeddings,
    build_openai_client,
    create_embedding_client,
    is_azure,
)


class TestAzureDetection:
    """Test cases for Azure OpenAI detection."""

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
    """Test cases for building OpenAI client instances."""

    @patch("core.clients.openai_client.OpenAI")
    @patch("core.clients.openai_client.is_azure")
    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_standard(self, mock_set_env, mock_read_env, mock_is_azure, mock_openai):
        """Test building standard OpenAI client."""
        mock_is_azure.return_value = False
        mock_read_env.return_value = {"OPENAI_API_KEY": "test_key"}
        mock_set_env.return_value.__enter__ = Mock()
        mock_set_env.return_value.__exit__ = Mock()

        client = build_openai_client()

        mock_read_env.assert_called_once_with("EMBEDDING")
        mock_openai.assert_called_once()
        assert client == mock_openai.return_value

    @patch("core.clients.openai_client.AzureOpenAI")
    @patch("core.clients.openai_client.is_azure")
    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_azure(self, mock_set_env, mock_read_env, mock_is_azure, mock_azure_openai):
        """Test building Azure OpenAI client."""
        mock_is_azure.return_value = True
        mock_read_env.return_value = {"AZURE_OPENAI_API_KEY": "test_key"}
        mock_set_env.return_value.__enter__ = Mock()
        mock_set_env.return_value.__exit__ = Mock()

        client = build_openai_client()

        mock_read_env.assert_called_once_with("EMBEDDING")
        mock_azure_openai.assert_called_once()
        assert client == mock_azure_openai.return_value

    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_with_custom_prefix(self, mock_set_env, mock_read_env):
        """Test building client with custom environment prefix."""
        mock_read_env.return_value = {}
        mock_set_env.return_value.__enter__ = Mock()
        mock_set_env.return_value.__exit__ = Mock()

        with (
            patch("core.clients.openai_client.is_azure", return_value=False),
            patch("core.clients.openai_client.OpenAI"),
        ):
            build_openai_client(env_prefix="CUSTOM")

        mock_read_env.assert_called_once_with("CUSTOM")

    @patch("core.clients.openai_client.read_env_config")
    @patch("core.clients.openai_client.set_env")
    def test_build_openai_client_with_kwargs(self, mock_set_env, mock_read_env):
        """Test building client with additional keyword arguments."""
        mock_read_env.return_value = {}
        mock_set_env.return_value.__enter__ = Mock()
        mock_set_env.return_value.__exit__ = Mock()

        with (
            patch("core.clients.openai_client.is_azure", return_value=False),
            patch("core.clients.openai_client.OpenAI") as mock_openai,
        ):
            build_openai_client(timeout=30, max_retries=5)

        mock_openai.assert_called_once_with(timeout=30, max_retries=5)


class TestBuildLangchainEmbeddings:
    """Test cases for building LangChain embeddings."""

    @patch("core.clients.openai_client.OpenAIEmbeddings")
    @patch("core.clients.openai_client.is_azure")
    def test_build_langchain_embeddings_standard(self, mock_is_azure, mock_openai_embeddings):
        """Test building standard OpenAI embeddings."""
        mock_is_azure.return_value = False

        embeddings = build_langchain_embeddings(model="text-embedding-ada-002")

        mock_openai_embeddings.assert_called_once_with(model="text-embedding-ada-002")
        assert embeddings == mock_openai_embeddings.return_value

    @patch("core.clients.openai_client.AzureOpenAIEmbeddings")
    @patch("core.clients.openai_client.is_azure")
    def test_build_langchain_embeddings_azure(self, mock_is_azure, mock_azure_embeddings):
        """Test building Azure OpenAI embeddings."""
        mock_is_azure.return_value = True

        embeddings = build_langchain_embeddings(model="text-embedding-ada-002")

        mock_azure_embeddings.assert_called_once_with(model="text-embedding-ada-002")
        assert embeddings == mock_azure_embeddings.return_value

    def test_build_langchain_embeddings_import_error(self):
        """Test fallback when LangChain imports fail."""
        with patch.dict("sys.modules", {"langchain_openai": None}):
            embeddings = build_langchain_embeddings()

            # Should return mock embeddings
            assert hasattr(embeddings, "embed_documents")
            assert hasattr(embeddings, "embed_query")

            # Test mock functionality
            docs = ["test doc 1", "test doc 2"]
            doc_embeddings = embeddings.embed_documents(docs)
            assert len(doc_embeddings) == 2
            assert all(len(emb) == 3 for emb in doc_embeddings)

            query_embedding = embeddings.embed_query("test query")
            assert len(query_embedding) == 3


class TestEmbeddingClient:
    """Test cases for the EmbeddingClient class."""

    def test_init_success(self, mock_openai_client):
        """Test successful EmbeddingClient initialization."""
        with patch("core.clients.openai_client.OpenAI", return_value=mock_openai_client):
            client = EmbeddingClient(api_key="test_key", model="text-embedding-3-small", azure_enabled=False)

            assert client.model == "text-embedding-3-small"
            assert client.azure_enabled is False
            assert client.client == mock_openai_client

    def test_init_azure_enabled(self, mock_openai_client):
        """Test EmbeddingClient initialization with Azure enabled."""
        with patch("core.clients.openai_client.AzureOpenAI", return_value=mock_openai_client):
            client = EmbeddingClient(api_key="test_key", model="text-embedding-ada-002", azure_enabled=True)

            assert client.model == "text-embedding-ada-002"
            assert client.azure_enabled is True
            assert client.client == mock_openai_client

    def test_init_failure_handling(self):
        """Test EmbeddingClient initialization failure handling."""
        with patch("core.clients.openai_client.OpenAI", side_effect=Exception("API error")):
            client = EmbeddingClient(api_key="test_key")

            assert client.client is None

    def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        mock_client = MagicMock()

        # Mock embeddings response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]), MagicMock(embedding=[0.4, 0.5, 0.6])]
        mock_client.embeddings.create.return_value = mock_response

        with patch("core.clients.openai_client.OpenAI", return_value=mock_client):
            client = EmbeddingClient(api_key="test_key")

            texts = ["text 1", "text 2"]
            embeddings = client.generate_embeddings(texts)

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

            mock_client.embeddings.create.assert_called_once_with(input=texts, model="text-embedding-3-small")

    def test_generate_embeddings_batching(self):
        """Test embedding generation with batching."""
        mock_client = MagicMock()

        # Mock responses for two batches
        mock_response_1 = MagicMock()
        mock_response_1.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_response_2 = MagicMock()
        mock_response_2.data = [MagicMock(embedding=[0.4, 0.5, 0.6])]

        mock_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]

        with patch("core.clients.openai_client.OpenAI", return_value=mock_client):
            client = EmbeddingClient(api_key="test_key")

            texts = ["text 1", "text 2"]
            embeddings = client.generate_embeddings(texts, batch_size=1)

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

            # Should make two separate API calls
            assert mock_client.embeddings.create.call_count == 2
            mock_client.embeddings.create.assert_has_calls(
                [
                    call(input=["text 1"], model="text-embedding-3-small"),
                    call(input=["text 2"], model="text-embedding-3-small"),
                ]
            )

    def test_generate_embeddings_api_failure(self):
        """Test embedding generation with API failures."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API error")

        with patch("core.clients.openai_client.OpenAI", return_value=mock_client):
            client = EmbeddingClient(api_key="test_key")

            texts = ["text 1", "text 2"]
            embeddings = client.generate_embeddings(texts)

            # Should return mock embeddings when API fails
            assert len(embeddings) == 2
            assert all(len(emb) == 1536 for emb in embeddings)  # Default dimension

    def test_generate_embeddings_no_client(self):
        """Test embedding generation when no client is available."""
        client = EmbeddingClient(api_key="test_key")
        client.client = None

        texts = ["text 1", "text 2"]
        embeddings = client.generate_embeddings(texts)

        # Should return mock embeddings
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)

    def test_generate_single_embedding(self):
        """Test single embedding generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        with patch("core.clients.openai_client.OpenAI", return_value=mock_client):
            client = EmbeddingClient(api_key="test_key")

            embedding = client.generate_single_embedding("test text")

            assert embedding == [0.1, 0.2, 0.3]
            mock_client.embeddings.create.assert_called_once_with(input=["test text"], model="text-embedding-3-small")

    def test_generate_mock_embeddings(self):
        """Test mock embedding generation."""
        client = EmbeddingClient(api_key="test_key")

        texts = ["hello world", "test text", "another example"]
        embeddings = client._generate_mock_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)

        # Test deterministic behavior
        embeddings2 = client._generate_mock_embeddings(texts)
        assert embeddings == embeddings2

    def test_get_model_info(self):
        """Test model information retrieval."""
        client = EmbeddingClient(api_key="test_key", model="text-embedding-ada-002", azure_enabled=True)

        info = client.get_model_info()

        assert info["model"] == "text-embedding-ada-002"
        assert info["dimensions"] == 1536
        assert info["max_input_tokens"] == 8191
        assert info["azure_enabled"] is True

    def test_different_model_configurations(self):
        """Test client with different model configurations."""
        test_cases = [
            {"model": "text-embedding-3-small", "azure_enabled": False, "expected_model": "text-embedding-3-small"},
            {"model": "text-embedding-ada-002", "azure_enabled": True, "expected_model": "text-embedding-ada-002"},
        ]

        for case in test_cases:
            with patch("core.clients.openai_client.OpenAI"), patch("core.clients.openai_client.AzureOpenAI"):
                client = EmbeddingClient(api_key="test_key", model=case["model"], azure_enabled=case["azure_enabled"])

                assert client.model == case["expected_model"]
                assert client.azure_enabled == case["azure_enabled"]


class TestCreateEmbeddingClient:
    """Test cases for the create_embedding_client factory function."""

    @patch("core.clients.openai_client.EmbeddingClient")
    @patch("core.clients.openai_client.is_azure")
    def test_create_embedding_client_defaults(self, mock_is_azure, mock_embedding_client):
        """Test creating client with default settings."""
        mock_is_azure.return_value = False

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"}):
            client = create_embedding_client()

            mock_embedding_client.assert_called_once_with(
                api_key="env_key", model="text-embedding-3-small", azure_enabled=False
            )

    @patch("core.clients.openai_client.EmbeddingClient")
    @patch("core.clients.openai_client.is_azure")
    def test_create_embedding_client_explicit_params(self, mock_is_azure, mock_embedding_client):
        """Test creating client with explicit parameters."""
        mock_is_azure.return_value = True

        client = create_embedding_client(
            api_key="explicit_key",
            model="text-embedding-ada-002",
            azure_enabled=False,  # Override auto-detection
            timeout=30,
        )

        mock_embedding_client.assert_called_once_with(
            api_key="explicit_key", model="text-embedding-ada-002", azure_enabled=False, timeout=30
        )

    @patch("core.clients.openai_client.EmbeddingClient")
    @patch("core.clients.openai_client.is_azure")
    def test_create_embedding_client_azure_auto_detection(self, mock_is_azure, mock_embedding_client):
        """Test Azure auto-detection in client creation."""
        mock_is_azure.return_value = True

        with patch.dict("os.environ", {"OPENAI_KEY": "env_key_alt"}):
            client = create_embedding_client()

            mock_embedding_client.assert_called_once_with(
                api_key="env_key_alt", model="text-embedding-3-small", azure_enabled=True
            )

    @patch("core.clients.openai_client.EmbeddingClient")
    def test_create_embedding_client_no_api_key(self, mock_embedding_client):
        """Test creating client when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            client = create_embedding_client()

            mock_embedding_client.assert_called_once_with(
                api_key=None, model="text-embedding-3-small", azure_enabled=False
            )

    @patch("core.clients.openai_client.EmbeddingClient")
    def test_create_embedding_client_prefer_openai_api_key(self, mock_embedding_client):
        """Test that OPENAI_API_KEY is preferred over OPENAI_KEY."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "primary_key", "OPENAI_KEY": "secondary_key"}):
            client = create_embedding_client()

            mock_embedding_client.assert_called_once_with(
                api_key="primary_key", model="text-embedding-3-small", azure_enabled=False
            )


class TestEmbeddingClientIntegration:
    """Integration tests for embedding client functionality."""

    def test_end_to_end_embedding_generation(self):
        """Test end-to-end embedding generation with mock API."""
        # Create mock OpenAI client
        mock_openai_client = MagicMock()

        # Setup mock response for multiple texts
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512),  # 1536 dimensions
            MagicMock(embedding=[0.4, 0.5, 0.6] * 512),
            MagicMock(embedding=[0.7, 0.8, 0.9] * 512),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response

        with patch("core.clients.openai_client.OpenAI", return_value=mock_openai_client):
            client = EmbeddingClient(api_key="test_key", model="text-embedding-3-small")

            texts = [
                "This is the first document to embed.",
                "Here is another document with different content.",
                "A third document for comprehensive testing.",
            ]

            embeddings = client.generate_embeddings(texts, batch_size=100)

            # Verify results
            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)

            # Verify API was called correctly
            mock_openai_client.embeddings.create.assert_called_once_with(input=texts, model="text-embedding-3-small")

    def test_large_batch_processing(self):
        """Test processing of large batches with multiple API calls."""
        mock_openai_client = MagicMock()

        # Setup mock responses for multiple batches
        def create_mock_response(batch_texts):
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[hash(text) % 100 / 100.0] * 10) for text in batch_texts  # Small embedding for test
            ]
            return mock_response

        mock_openai_client.embeddings.create.side_effect = lambda input, model: create_mock_response(input)

        with patch("core.clients.openai_client.OpenAI", return_value=mock_openai_client):
            client = EmbeddingClient(api_key="test_key")

            # Create 5 texts with batch size of 2
            texts = [f"Document number {i}" for i in range(5)]
            embeddings = client.generate_embeddings(texts, batch_size=2)

            # Should make 3 API calls (2+2+1)
            assert mock_openai_client.embeddings.create.call_count == 3
            assert len(embeddings) == 5

    def test_error_recovery_and_fallback(self):
        """Test error recovery with fallback to mock embeddings."""
        mock_openai_client = MagicMock()

        # First call succeeds, second fails, third succeeds
        mock_success_response = MagicMock()
        mock_success_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_openai_client.embeddings.create.side_effect = [
            mock_success_response,
            Exception("Rate limit exceeded"),
            mock_success_response,
        ]

        with patch("core.clients.openai_client.OpenAI", return_value=mock_openai_client):
            client = EmbeddingClient(api_key="test_key")

            texts = ["text1", "text2", "text3"]
            embeddings = client.generate_embeddings(texts, batch_size=1)

            # Should have embeddings for all texts
            assert len(embeddings) == 3

            # First and third should be real, second should be mock
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert len(embeddings[1]) == 1536  # Mock embedding dimension
            assert embeddings[2] == [0.1, 0.2, 0.3]

    def test_client_factory_integration(self):
        """Test integration of client factory with environment configuration."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "integration_test_key", "AZURE_OPENAI_ENABLED": "false"}):
            with patch("core.clients.openai_client.OpenAI") as mock_openai:
                mock_openai_instance = MagicMock()
                mock_openai.return_value = mock_openai_instance

                client = create_embedding_client(model="text-embedding-ada-002", timeout=60)

                # Verify client was created with correct parameters
                mock_openai.assert_called_once_with(api_key="integration_test_key", timeout=60)

                assert client.model == "text-embedding-ada-002"
                assert client.azure_enabled is False

    def test_model_info_consistency(self):
        """Test that model info is consistent across different configurations."""
        configurations = [
            {"model": "text-embedding-3-small", "azure_enabled": False},
            {"model": "text-embedding-ada-002", "azure_enabled": True},
            {"model": "text-embedding-3-large", "azure_enabled": False},
        ]

        for config in configurations:
            with patch("core.clients.openai_client.OpenAI"), patch("core.clients.openai_client.AzureOpenAI"):
                client = EmbeddingClient(
                    api_key="test_key", model=config["model"], azure_enabled=config["azure_enabled"]
                )

                info = client.get_model_info()

                assert info["model"] == config["model"]
                assert info["azure_enabled"] == config["azure_enabled"]
                assert "dimensions" in info
                assert "max_input_tokens" in info
                assert isinstance(info["dimensions"], int)
                assert isinstance(info["max_input_tokens"], int)
