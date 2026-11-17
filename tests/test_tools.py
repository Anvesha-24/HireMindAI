
from src.tools import (
    compare_jobs,
    get_job_details,
    search_jobs,
)


def test_get_existing_job():
    result = get_job_details("JOB004")

    assert result["job_id"] == "JOB004"
    assert result["title"] == "AI Application Developer"


def test_get_missing_job():
    result = get_job_details("UNKNOWN")

    assert "error" in result


def test_compare_jobs():
    result = compare_jobs("JOB001", "JOB004")

    assert result["job_1"]["job_id"] == "JOB001"
    assert result["job_2"]["job_id"] == "JOB004"
    assert "python" in result["common_skills"]


def test_search_jobs():
    result = search_jobs(
        candidate_skills=["Python", "RAG", "Embeddings"],
        candidate_summary="Building AI applications using LLMs",
        experience_years=0,
        top_k=3,
    )

    assert len(result["jobs"]) == 3
    assert result["jobs"][0]["final_score"] >= (
        result["jobs"][-1]["final_score"]
    )


def test_invalid_top_k():
    result = search_jobs(
        candidate_skills=["Python"],
        top_k=20,
    )

    assert "error" in result


def test_search_without_candidate_information():
    result = search_jobs([], "")

    assert "error" in result
