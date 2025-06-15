"""
Tests for embedding utility functions.
"""

import math

import pytest

from core.utils.embedding_utils import cosine_similarity, euclidean_distance, normalize_embedding


class TestNormalizeEmbedding:
    """Test cases for normalize_embedding function."""

    def test_normalize_simple_vector(self):
        """Test normalizing a simple vector."""
        embedding = [3.0, 4.0]  # Magnitude = 5.0
        normalized = normalize_embedding(embedding)

        expected = [0.6, 0.8]  # 3/5, 4/5
        assert len(normalized) == len(expected)
        for actual, exp in zip(normalized, expected):
            assert abs(actual - exp) < 1e-10

    def test_normalize_unit_vector(self):
        """Test normalizing a vector that's already unit length."""
        embedding = [1.0, 0.0, 0.0]
        normalized = normalize_embedding(embedding)

        expected = [1.0, 0.0, 0.0]
        assert len(normalized) == len(expected)
        for actual, exp in zip(normalized, expected):
            assert abs(actual - exp) < 1e-10

    def test_normalize_negative_values(self):
        """Test normalizing a vector with negative values."""
        embedding = [-1.0, 1.0]  # Magnitude = sqrt(2)
        normalized = normalize_embedding(embedding)

        expected = [-1 / math.sqrt(2), 1 / math.sqrt(2)]
        assert len(normalized) == len(expected)
        for actual, exp in zip(normalized, expected):
            assert abs(actual - exp) < 1e-10

    def test_normalize_empty_embedding(self):
        """Test that empty embedding raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize empty embedding"):
            normalize_embedding([])

    def test_normalize_zero_magnitude(self):
        """Test that zero-magnitude embedding raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize zero-magnitude embedding"):
            normalize_embedding([0.0, 0.0, 0.0])

    def test_normalized_vector_has_unit_length(self):
        """Test that normalized vector has unit length."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        normalized = normalize_embedding(embedding)

        # Calculate magnitude of normalized vector
        magnitude = math.sqrt(sum(x * x for x in normalized))
        assert abs(magnitude - 1.0) < 1e-10


class TestCosineSimilarity:
    """Test cases for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0, 3.0]

        similarity = cosine_similarity(embedding1, embedding2)
        assert abs(similarity - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        embedding1 = [1.0, 0.0]
        embedding2 = [0.0, 1.0]

        similarity = cosine_similarity(embedding1, embedding2)
        assert abs(similarity - 0.0) < 1e-10

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        embedding1 = [1.0, 2.0]
        embedding2 = [-1.0, -2.0]

        similarity = cosine_similarity(embedding1, embedding2)
        assert abs(similarity - (-1.0)) < 1e-10

    def test_different_dimensions(self):
        """Test that different dimensions raise ValueError."""
        embedding1 = [1.0, 2.0]
        embedding2 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            cosine_similarity(embedding1, embedding2)

    def test_empty_embeddings(self):
        """Test that empty embeddings raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate similarity for empty embeddings"):
            cosine_similarity([], [1.0, 2.0])

    def test_zero_magnitude_vectors(self):
        """Test cosine similarity with zero-magnitude vectors."""
        embedding1 = [0.0, 0.0]
        embedding2 = [1.0, 2.0]

        similarity = cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0


class TestEuclideanDistance:
    """Test cases for euclidean_distance function."""

    def test_identical_vectors(self):
        """Test Euclidean distance of identical vectors."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0, 3.0]

        distance = euclidean_distance(embedding1, embedding2)
        assert abs(distance - 0.0) < 1e-10

    def test_simple_distance(self):
        """Test Euclidean distance calculation."""
        embedding1 = [0.0, 0.0]
        embedding2 = [3.0, 4.0]

        distance = euclidean_distance(embedding1, embedding2)
        assert abs(distance - 5.0) < 1e-10  # 3-4-5 triangle

    def test_different_dimensions(self):
        """Test that different dimensions raise ValueError."""
        embedding1 = [1.0, 2.0]
        embedding2 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            euclidean_distance(embedding1, embedding2)

    def test_empty_embeddings(self):
        """Test that empty embeddings raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate distance for empty embeddings"):
            euclidean_distance([], [1.0, 2.0])

    def test_negative_values(self):
        """Test Euclidean distance with negative values."""
        embedding1 = [-1.0, -1.0]
        embedding2 = [1.0, 1.0]

        distance = euclidean_distance(embedding1, embedding2)
        expected = math.sqrt(8)  # sqrt((2)^2 + (2)^2)
        assert abs(distance - expected) < 1e-10
