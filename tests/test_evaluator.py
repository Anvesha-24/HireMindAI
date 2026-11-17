
import pytest

from src.evaluator import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    evaluate_recommendations,
)


def test_precision_at_k():
    recommended = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    assert precision_at_k(recommended, relevant, 5) == pytest.approx(0.4)


def test_recall_at_k():
    recommended = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    assert recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)


def test_mean_reciprocal_rank():
    recommended = ["A", "B", "C", "D"]
    relevant = {"C", "D"}

    assert mean_reciprocal_rank(recommended, relevant) == pytest.approx(1 / 3)


def test_no_relevant_results():
    assert mean_reciprocal_rank(["A", "B"], {"X"}) == 0.0


def test_evaluate_recommendations():
    result = evaluate_recommendations(
        ["A", "B", "C"],
        {"A", "C"},
        k=3,
    )

    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["recall@3"] == 1.0
    assert result["mrr"] == 1.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)
