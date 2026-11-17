
import pytest

from src.vector_store import (
    JobVectorStore,
    build_job_text,
    load_jobs,
)


def test_load_jobs():
    jobs = load_jobs()

    assert len(jobs) >= 10
    assert all("job_id" in job for job in jobs)


def test_build_job_text():
    job = {
        "title": "Python Developer",
        "skills": ["Python", "FastAPI"],
        "description": "Build backend APIs.",
    }

    text = build_job_text(job)

    assert "Python Developer" in text
    assert "FastAPI" in text
    assert "backend APIs" in text


def test_empty_job_collection():
    with pytest.raises(ValueError, match="cannot be empty"):
        JobVectorStore([])


def test_empty_search_query():
    store = JobVectorStore([{
        "job_id": "TEST001",
        "title": "Python Developer",
        "skills": ["Python"],
        "description": "Build Python applications.",
    }])

    with pytest.raises(ValueError, match="cannot be empty"):
        store.search("  ")


def test_invalid_top_k():
    store = JobVectorStore([{
        "job_id": "TEST001",
        "title": "Python Developer",
        "skills": ["Python"],
        "description": "Build Python applications.",
    }])

    with pytest.raises(ValueError, match="top_k"):
        store.search("Python developer", top_k=0)
