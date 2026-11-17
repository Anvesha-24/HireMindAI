
import numpy as np
import pytest

from src.embeddings import (
    cosine_similarity,
    generate_embeddings,
)


def test_cosine_similarity():
    query = np.array([1.0, 0.0])

    documents = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    scores = cosine_similarity(query, documents)

    assert np.allclose(scores, [1.0, 0.0, -1.0])


def test_cosine_similarity_dimension_mismatch():
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="dimensions must match"):
        cosine_similarity(query, documents)


def test_empty_text_list():
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_embeddings([])


def test_empty_string():
    with pytest.raises(ValueError, match="empty strings"):
        generate_embeddings(["Python", "  "])
