"""
AWS Bedrock embedding provider implementation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from .base_embedding_provider import BaseEmbeddingProvider, EmbeddingModelInfo, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = None
    NoCredentialsError = None


class AWSBedrockEmbeddingProvider(BaseEmbeddingProvider):
    """AWS Bedrock embedding provider."""

    # AWS Bedrock embedding models
    MODELS = {
        "amazon.titan-embed-text-v1": {
            "dimensions": 1536,
            "max_input_tokens": 8192,
            "description": "Amazon Titan Text Embeddings V1",
        },
        "amazon.titan-embed-text-v2:0": {
            "dimensions": 1024,
            "max_input_tokens": 8192,
            "description": "Amazon Titan Text Embeddings V2",
        },
        "cohere.embed-english-v3": {
            "dimensions": 1024,
            "max_input_tokens": 512,
            "description": "Cohere Embed English V3",
        },
        "cohere.embed-multilingual-v3": {
            "dimensions": 1024,
            "max_input_tokens": 512,
            "description": "Cohere Embed Multilingual V3",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS Bedrock embedding provider.

        Args:
            config: Configuration containing:
                - region_name: AWS region (required)
                - aws_access_key_id: AWS access key (optional, uses default chain)
                - aws_secret_access_key: AWS secret key (optional, uses default chain)
                - aws_session_token: AWS session token (optional)
                - profile_name: AWS profile name (optional)
                - role_arn: IAM role ARN to assume (optional)
                - default_model: Default model to use
                - timeout: Request timeout in seconds
        """
        super().__init__(config)

        self.region_name = config.get("region_name")
        if not self.region_name:
            raise ValueError("AWS region_name is required for Bedrock")

        self.aws_access_key_id = config.get("aws_access_key_id")
        self.aws_secret_access_key = config.get("aws_secret_access_key")
        self.aws_session_token = config.get("aws_session_token")
        self.profile_name = config.get("profile_name")
        self.role_arn = config.get("role_arn")
        self.timeout = config.get("timeout", 60)

        # Initialize Bedrock client
        if boto3 is None:
            logger.warning("boto3 not available, using mock implementation")
            self.client = None
        else:
            self.client = self._create_bedrock_client()

    def _create_bedrock_client(self):
        """Create AWS Bedrock runtime client."""
        try:
            session_kwargs = {"region_name": self.region_name}

            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)

            client_kwargs = {"service_name": "bedrock-runtime", "region_name": self.region_name}

            # Use explicit credentials if provided
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs.update(
                    {"aws_access_key_id": self.aws_access_key_id, "aws_secret_access_key": self.aws_secret_access_key}
                )

                if self.aws_session_token:
                    client_kwargs["aws_session_token"] = self.aws_session_token

            client = session.client(**client_kwargs)

            # Assume role if specified
            if self.role_arn:
                sts_client = session.client("sts", region_name=self.region_name)
                assumed_role = sts_client.assume_role(RoleArn=self.role_arn, RoleSessionName="nbedr-embedding-session")

                credentials = assumed_role["Credentials"]
                client = session.client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"],
                )

            return client

        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            return None

    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None, **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings using AWS Bedrock.

        Args:
            texts: List of texts to embed
            model: Model to use (defaults to config default)
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._validate_inputs(texts)

        if model is None:
            model = self.get_default_model() or "amazon.titan-embed-text-v1"

        if batch_size is None:
            batch_size = min(self.get_max_batch_size(), 100)

        if not self.client:
            logger.warning("Bedrock client not available, returning mock embeddings")
            mock_embeddings = self._generate_mock_embeddings(texts, self.MODELS.get(model, {}).get("dimensions", 1536))
            return EmbeddingResult(
                embeddings=mock_embeddings,
                model=model,
                dimensions=len(mock_embeddings[0]) if mock_embeddings else 1536,
                token_count=sum(len(text.split()) for text in texts),
            )

        all_embeddings = []

        # Process texts individually or in small batches depending on model
        for i, text in enumerate(texts):
            try:
                embedding = await self._generate_single_embedding(text, model)
                all_embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated embeddings for {i + 1}/{len(texts)} texts")

            except Exception as e:
                logger.error(f"Failed to generate embedding for text {i + 1}: {e}")
                # Add mock embedding for failed text
                mock_embedding = self._generate_mock_embeddings(
                    [text], self.MODELS.get(model, {}).get("dimensions", 1536)
                )[0]
                all_embeddings.append(mock_embedding)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model,
            dimensions=len(all_embeddings[0]) if all_embeddings else self.MODELS.get(model, {}).get("dimensions", 1536),
            token_count=sum(len(text.split()) for text in texts),
            usage_stats={"provider": "aws_bedrock", "region": self.region_name},
        )

    async def _generate_single_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text using Bedrock.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector
        """
        body = self._prepare_request_body(text, model)

        try:
            response = self.client.invoke_model(
                modelId=model, body=json.dumps(body), contentType="application/json", accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            return self._extract_embedding(response_body, model)

        except Exception as e:
            logger.error(f"Bedrock API call failed: {e}")
            raise

    def _prepare_request_body(self, text: str, model: str) -> Dict[str, Any]:
        """Prepare request body for different Bedrock models.

        Args:
            text: Text to embed
            model: Model identifier

        Returns:
            Request body dictionary
        """
        if model.startswith("amazon.titan-embed"):
            return {"inputText": text}
        elif model.startswith("cohere.embed"):
            return {"texts": [text], "input_type": "search_document"}
        else:
            # Generic format
            return {"inputText": text}

    def _extract_embedding(self, response_body: Dict[str, Any], model: str) -> List[float]:
        """Extract embedding from response body.

        Args:
            response_body: Response from Bedrock API
            model: Model identifier

        Returns:
            Embedding vector
        """
        if model.startswith("amazon.titan-embed"):
            return response_body.get("embedding", [])
        elif model.startswith("cohere.embed"):
            embeddings = response_body.get("embeddings", [])
            return embeddings[0] if embeddings else []
        else:
            # Try common response formats
            if "embedding" in response_body:
                return response_body["embedding"]
            elif "embeddings" in response_body:
                embeddings = response_body["embeddings"]
                return embeddings[0] if isinstance(embeddings, list) and embeddings else []
            else:
                logger.warning(f"Unknown response format for model {model}")
                return []

    async def get_model_info(self, model: str) -> EmbeddingModelInfo:
        """Get information about a Bedrock embedding model.

        Args:
            model: Model name

        Returns:
            EmbeddingModelInfo with model details
        """
        if model not in self.MODELS:
            # If not in known models, create basic info
            return EmbeddingModelInfo(
                model_name=model,
                dimensions=1536,  # Default dimension
                max_input_tokens=8192,  # Default max tokens
                supports_batch=False,  # Most Bedrock models process individually
                provider="aws_bedrock",
                description=f"AWS Bedrock model: {model}",
            )

        model_spec = self.MODELS[model]

        return EmbeddingModelInfo(
            model_name=model,
            dimensions=model_spec["dimensions"],
            max_input_tokens=model_spec["max_input_tokens"],
            supports_batch=False,  # Bedrock typically processes individually
            provider="aws_bedrock",
            description=model_spec["description"],
        )

    def list_models(self) -> List[str]:
        """List available Bedrock embedding models.

        Returns:
            List of model names
        """
        return list(self.MODELS.keys())

    async def health_check(self) -> bool:
        """Check if AWS Bedrock is accessible.

        Returns:
            True if Bedrock is accessible, False otherwise
        """
        if not self.client:
            return False

        try:
            # Try a simple embedding request
            await self._generate_single_embedding("test", self.get_default_model() or "amazon.titan-embed-text-v1")
            return True
        except Exception as e:
            logger.error(f"Bedrock health check failed: {e}")
            return False

    def get_default_model(self) -> str:
        """Get the default Bedrock embedding model.

        Returns:
            Default model name
        """
        return self.config.get("default_model", "amazon.titan-embed-text-v1")

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for Bedrock.

        Returns:
            Maximum batch size (1 for individual processing)
        """
        return 1  # Bedrock typically processes embeddings individually

    def get_region_info(self) -> Dict[str, Any]:
        """Get AWS region and configuration information.

        Returns:
            Dictionary with region and config info
        """
        return {
            "region_name": self.region_name,
            "profile_name": self.profile_name,
            "role_arn": self.role_arn,
            "has_explicit_credentials": bool(self.aws_access_key_id and self.aws_secret_access_key),
        }
