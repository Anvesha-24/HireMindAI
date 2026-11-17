
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.

    The model is downloaded the first time it is loaded.
    """
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Convert a list of texts into normalized 384-dimensional vectors.
    """

    if not texts:
        raise ValueError("Text list cannot be empty.")

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Texts cannot contain empty strings.")

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query and multiple documents.

    Inputs can be normalized or unnormalized vectors.
    Returns one similarity score per document.
    """

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    documents = np.asarray(document_embeddings, dtype=np.float32)

    if documents.ndim != 2:
        raise ValueError("Document embeddings must be a 2D array.")

    if query.shape[0] != documents.shape[1]:
        raise ValueError("Query and document dimensions must match.")

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Embeddings cannot be zero vectors.")

    scores = np.dot(documents, query) / (
        document_norms * query_norm
    )

    return scores
