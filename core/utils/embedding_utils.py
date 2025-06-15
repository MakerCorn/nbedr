"""
Utility functions for embedding processing and manipulation.
"""

import math
from typing import List


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize an embedding vector to unit length using L2 normalization.

    Args:
        embedding: List of float values representing an embedding vector

    Returns:
        List of float values representing the normalized embedding vector

    Raises:
        ValueError: If the embedding is empty or has zero magnitude
    """
    if not embedding:
        raise ValueError("Cannot normalize empty embedding")

    # Calculate the L2 norm (magnitude) of the vector
    magnitude = math.sqrt(sum(x * x for x in embedding))

    if magnitude == 0.0:
        raise ValueError("Cannot normalize zero-magnitude embedding")

    # Normalize each component by dividing by the magnitude
    return [x / magnitude for x in embedding]


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If embeddings have different dimensions or are empty
    """
    if not embedding1 or not embedding2:
        raise ValueError("Cannot calculate similarity for empty embeddings")

    if len(embedding1) != len(embedding2):
        raise ValueError(f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}")

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(x * x for x in embedding1))
    magnitude2 = math.sqrt(sum(x * x for x in embedding2))

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate Euclidean distance between two embedding vectors.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Euclidean distance between the vectors

    Raises:
        ValueError: If embeddings have different dimensions or are empty
    """
    if not embedding1 or not embedding2:
        raise ValueError("Cannot calculate distance for empty embeddings")

    if len(embedding1) != len(embedding2):
        raise ValueError(f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}")

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
