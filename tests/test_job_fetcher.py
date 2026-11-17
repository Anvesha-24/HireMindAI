
from unittest.mock import patch, Mock

import pytest

from src.job_fetcher import (
    normalize_job,
    search_adzuna_jobs,
    AdzunaAPIError,
)


def sample_adzuna_job():
    return {
        "id": "12345",
        "title": "Python Developer",
        "description": "Python and FastAPI developer",
        "redirect_url": "https://example.com/original-job",
        "created": "2026-11-17T10:00:00Z",
        "company": {
            "display_name": "Example Company"
        },
        "location": {
            "display_name": "Noida, Uttar Pradesh"
        },
        "salary_min": 400000,
        "salary_max": 800000,
        "category": {
            "label": "IT Jobs"
        },
    }


def test_normalize_job():
    job = normalize_job(sample_adzuna_job())

    assert job["job_id"] == "ADZUNA_12345"
    assert job["title"] == "Python Developer"
    assert job["company"] == "Example Company"
    assert job["location"] == "Noida, Uttar Pradesh"


def test_original_job_url_is_preserved():
    job = normalize_job(sample_adzuna_job())

    assert job["job_url"] == "https://example.com/original-job"


def test_normalize_job_handles_missing_fields():
    job = normalize_job({})

    assert job["title"] == "Title not available"
    assert job["company"] == "Company not specified"
    assert job["job_url"] == ""


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        search_adzuna_jobs("")


def test_invalid_results_per_page():
    with pytest.raises(ValueError):
        search_adzuna_jobs(
            "Python Developer",
            results_per_page=100,
        )


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)

    with pytest.raises(AdzunaAPIError):
        search_adzuna_jobs("Python Developer")


@patch("src.job_fetcher.requests.get")
def test_successful_api_response(mock_get, monkeypatch):
    monkeypatch.setenv("ADZUNA_APP_ID", "test_id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test_key")

    mock_response = Mock()

    mock_response.json.return_value = {
        "count": 1,
        "results": [sample_adzuna_job()],
    }

    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = search_adzuna_jobs(
        "Python Developer",
        location="Noida",
    )

    assert result["total_matches"] == 1
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["title"] == "Python Developer"
    assert result["jobs"][0]["source"] == "Adzuna"

    mock_get.assert_called_once()


@patch("src.job_fetcher.requests.get")
def test_empty_api_response(mock_get, monkeypatch):
    monkeypatch.setenv("ADZUNA_APP_ID", "test_id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test_key")

    mock_response = Mock()

    mock_response.json.return_value = {
        "count": 0,
        "results": [],
    }

    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = search_adzuna_jobs("Python Developer")

    assert result["total_matches"] == 0
    assert result["jobs"] == []


@patch("src.job_fetcher.requests.get")
def test_api_http_error(mock_get, monkeypatch):
    monkeypatch.setenv("ADZUNA_APP_ID", "test_id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test_key")

    import requests

    mock_response = Mock()
    mock_response.status_code = 401

    mock_response.raise_for_status.side_effect = (
        requests.exceptions.HTTPError(
            response=mock_response
        )
    )

    mock_get.return_value = mock_response

    with pytest.raises(AdzunaAPIError):
        search_adzuna_jobs("Python Developer")
