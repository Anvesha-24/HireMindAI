
import pytest

from src.job_ranker import (
    calculate_experience_match,
    calculate_skill_match,
    rank_jobs,
)


def test_skill_match():
    score, matched, missing = calculate_skill_match(
        ["Python", "FastAPI", "SQL"],
        ["Python", "SQL", "Docker"],
    )

    assert score == pytest.approx(2 / 3)
    assert matched == ["python", "sql"]
    assert missing == ["docker"]


def test_no_required_skills():
    score, matched, missing = calculate_skill_match(
        ["Python"],
        [],
    )

    assert score == 0
    assert matched == []
    assert missing == []


def test_experience_match():
    assert calculate_experience_match(1, "0-2 years") == 1.0
    assert calculate_experience_match(3, "0-2 years") == 0.8
    assert calculate_experience_match(0, "Internship") == 1.0
    assert calculate_experience_match(None, "0-2 years") == 0.5


def test_rank_jobs():
    profile = {
        "skills": ["Python", "RAG", "Embeddings"],
    }

    jobs = [
        {
            "job_id": "JOB001",
            "title": "Python Developer",
            "skills": ["Python", "SQL"],
            "experience_required": "0-2 years",
            "semantic_score": 0.8,
        },
        {
            "job_id": "JOB002",
            "title": "AI Developer",
            "skills": ["Python", "RAG", "Embeddings"],
            "experience_required": "Internship",
            "semantic_score": 0.7,
        },
    ]

    results = rank_jobs(
        profile,
        jobs,
        candidate_experience_years=0,
    )

    assert len(results) == 2
    assert results[0]["job_id"] == "JOB002"
    assert results[0]["match_percentage"] > results[1]["match_percentage"]
    assert "matched_skills" in results[0]
    assert "missing_skills" in results[0]


def test_rank_jobs_empty():
    assert rank_jobs({"skills": ["Python"]}, []) == []
