
def precision_at_k(
    recommended_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate the fraction of top-k recommendations that are relevant."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not recommended_ids:
        return 0.0

    top_k = recommended_ids[:k]

    relevant_count = sum(
        1 for job_id in top_k if job_id in relevant_ids
    )

    return relevant_count / len(top_k)


def recall_at_k(
    recommended_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate the fraction of relevant jobs found in the top k."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]

    relevant_count = sum(
        1 for job_id in top_k if job_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def mean_reciprocal_rank(
    recommended_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    if not relevant_ids:
        return 0.0

    for rank, job_id in enumerate(recommended_ids, start=1):
        if job_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_recommendations(
    recommended_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> dict:
    """Return a collection of retrieval evaluation metrics."""

    return {
        f"precision@{k}": precision_at_k(
            recommended_ids,
            relevant_ids,
            k,
        ),
        f"recall@{k}": recall_at_k(
            recommended_ids,
            relevant_ids,
            k,
        ),
        "mrr": mean_reciprocal_rank(
            recommended_ids,
            relevant_ids,
        ),
    }
