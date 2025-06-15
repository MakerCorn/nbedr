"""
Google Vertex AI embedding provider implementation.
"""
# mypy: disable-error-code="assignment,misc,arg-type,no-any-return"

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import vertexai
    from google.api_core.exceptions import GoogleAPIError
    from google.auth.exceptions import DefaultCredentialsError
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
except ImportError:
    vertexai = None
    TextEmbeddingModel = None
    TextEmbeddingInput = None
    DefaultCredentialsError = None
    GoogleAPIError = None


class GoogleVertexEmbeddingProvider(BaseEmbeddingProvider):
    """Google Vertex AI embedding provider."""

    # Google Vertex AI embedding models
    MODELS = {
        "textembedding-gecko@003": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Latest Gecko text embedding model",
        },
        "textembedding-gecko@002": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Gecko text embedding model V2",
        },
        "textembedding-gecko@001": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Gecko text embedding model V1",
        },
        "textembedding-gecko-multilingual@001": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Multilingual Gecko text embedding model",
        },
        "text-embedding-004": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Latest text embedding model",
        },
        "text-multilingual-embedding-002": {
            "dimensions": 768,
            "max_input_tokens": 3072,
            "description": "Multilingual text embedding model",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize Google Vertex AI embedding provider.

        Args:
            config: Configuration containing:
                - project_id: Google Cloud project ID (required)
                - location: Google Cloud location/region (required)
                - credentials_path: Path to service account JSON file (optional)
                - default_model: Default model to use
                - timeout: Request timeout in seconds
        """
        super().__init__(config)

        self.project_id = config.get("project_id")
        self.location = config.get("location", "us-central1")
        self.credentials_path = config.get("credentials_path")
        self.timeout = config.get("timeout", 60)

        if not self.project_id:
            raise ValueError("Google Cloud project_id is required for Vertex AI")

        # Initialize Vertex AI
        if vertexai is None:
            logger.warning("Vertex AI library not available, using mock implementation")
            self.initialized = False
        else:
            self.initialized = self._initialize_vertex_ai()

    def _initialize_vertex_ai(self) -> bool:
        """Initialize Vertex AI with project and location.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set up credentials if path provided
            if self.credentials_path:
                import os

                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"Initialized Vertex AI for project {self.project_id} in {self.location}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return False

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingResult:
        """Generate embeddings using Google Vertex AI.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default)
            batch_size: Batch size for processing
            task_type: Task type for embeddings (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            title: Optional title for the texts
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._validate_inputs(texts)

        if model is None:
            model = self.get_default_model() or "textembedding-gecko@003"

        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 250)

        if not self.initialized:
            logger.warning("Vertex AI not available, returning mock embeddings")
            dimensions = int(self.MODELS.get(model, {}).get("dimensions", 768))
            mock_embeddings = self._generate_mock_embeddings(texts, dimensions)  # type: ignore[arg-type]
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=model,
                dimensions=len(mock_embeddings[0]) if mock_embeddings else 768,
                token_count=sum(len(text.split()) for text in texts),
            )

        try:
            embedding_model = TextEmbeddingModel.from_pretrained(model)
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            # Fall back to mock embeddings
            dimensions = int(self.MODELS.get(model, {}).get("dimensions", 768))
            mock_embeddings = self._generate_mock_embeddings(texts, dimensions)  # type: ignore[arg-type]
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=model,
                dimensions=len(mock_embeddings[0]) if mock_embeddings else 768,
                token_count=sum(len(text.split()) for text in texts),
            )

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Prepare embedding inputs
                embedding_inputs = []
                for text in batch_texts:
                    if TextEmbeddingInput:
                        # Use structured input for newer API
                        embedding_input = TextEmbeddingInput(text=text, task_type=task_type, title=title)
                        embedding_inputs.append(embedding_input)
                    else:
                        # Fall back to simple text input
                        embedding_inputs.append(text)

                # Generate embeddings
                if embedding_inputs and hasattr(embedding_model, "get_embeddings"):
                    embeddings = embedding_model.get_embeddings(embedding_inputs)
                    batch_embeddings = [emb.values for emb in embeddings]
                else:
                    # Older API or fallback
                    batch_embeddings = []
                    for text in batch_texts:
                        emb = embedding_model.get_embeddings([text])
                        batch_embeddings.append(emb[0].values)

                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                # Add mock embeddings for failed batch
                mock_batch = self._generate_mock_embeddings(
                    batch_texts, self.MODELS.get(model, {}).get("dimensions", 768)
                )
                all_embeddings.extend(mock_batch)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.MODELS.get(model, {}).get("dimensions", 768),
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={
                "provider": "google_vertex",
                "project_id": self.project_id,
                "location": self.location,
                "task_type": task_type,
                "batches_processed": (len(texts) - 1) // batch_size + 1,
            },
        )

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about a Vertex AI embedding model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        if model not in self.MODELS:
            # If not in known models, create basic info
            return EmbeddingModelInfo(
                model_name=model,
                dimensions=768,  # Default dimension for Vertex AI
                max_input_tokens=3072,  # Default max tokens
                supports_batch=True,
                provider="google_vertex",
                description=f"Google Vertex AI model: {model}",
            )

        model_spec = self.MODELS[model]

        return EmbeddingModelInfo(
            model_name=model,
            dimensions=model_spec["dimensions"],
            max_input_tokens=model_spec["max_input_tokens"],
            supports_batch=True,
            provider="google_vertex",
            description=model_spec["description"],
        )

    def list_models(self) -> List[str]:
        """List available Vertex AI embedding models.

        Returns:
            List of model names
        """
        return list(self.MODELS.keys())

    async def health_check(self) -> bool:
        """Check if Google Vertex AI is accessible.

        Returns:
            True if Vertex AI is accessible, False otherwise
        """
        if not self.initialized:
            return False

        try:
            # Try to load a model and generate a simple embedding
            model = self.get_default_model() or "textembedding-gecko@003"
            embedding_model = TextEmbeddingModel.from_pretrained(model)

            # Simple test embedding
            embeddings = embedding_model.get_embeddings(["test"])
            return len(embeddings) > 0 and len(embeddings[0].values) > 0

        except Exception as e:
            logger.error(f"Vertex AI health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default Vertex AI embedding model.

        Returns:
            Default model name
        """
        return self.config.get("default_model", "textembedding-gecko@003")

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Vertex AI.

        Returns:
            Maximum batch size
        """
        return self.config.get("max_batch_size", 250)

    def get_project_info(self) -> Dict[str, Any]:
        """Get Google Cloud project information.

        Returns:
            Dictionary with project and config info
        """
        return {
            "project_id": self.project_id,
            "location": self.location,
            "credentials_path": self.credentials_path,
            "initialized": self.initialized,
        }

    def get_supported_task_types(self) -> List[str]:
        """Get supported task types for embeddings.

        Returns:
            List of supported task types
        """
        return [
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        ]
