
import os
import requests

from dotenv import load_dotenv


# Load credentials from the project's .env file
load_dotenv()

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")
REQUEST_TIMEOUT = 15


class AdzunaAPIError(Exception):
    """Raised when the Adzuna API request fails."""


def normalize_job(raw_job):
    """
    Convert an Adzuna job into HireMind AI's common job format.

    Original application URLs are preserved in job_url.
    """

    company_data = raw_job.get("company") or {}
    location_data = raw_job.get("location") or {}

    category_data = raw_job.get("category") or {}

    salary_min = raw_job.get("salary_min")
    salary_max = raw_job.get("salary_max")

    return {
        "job_id": f"ADZUNA_{raw_job.get('id', '')}",
        "external_id": str(raw_job.get("id", "")),
        "title": raw_job.get("title", "Title not available"),
        "company": company_data.get(
            "display_name",
            "Company not specified",
        ),
        "location": location_data.get(
            "display_name",
            "Location not specified",
        ),
        "description": raw_job.get("description", ""),
        "required_skills": [],
        "experience_required": None,
        "job_url": raw_job.get("redirect_url", ""),
        "source": "Adzuna",
        "created": raw_job.get("created"),
        "category": category_data.get("label", ""),
        "contract_type": raw_job.get("contract_type"),
        "contract_time": raw_job.get("contract_time"),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": raw_job.get("salary_currency"),
    }


def search_adzuna_jobs(
    query,
    location="India",
    results_per_page=20,
    page=1,
):
    """
    Fetch real job advertisements from Adzuna.

    Args:
        query: Job title or keywords, e.g. Python Developer.
        location: Search location, e.g. India, Noida, Bangalore.
        results_per_page: Number of results (maximum 50).
        page: Adzuna result page, starting at 1.

    Returns:
        Dictionary containing normalized jobs and result metadata.

    Raises:
        ValueError: Invalid input.
        AdzunaAPIError: Missing credentials or API failure.
    """

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise AdzunaAPIError(
            "Adzuna credentials are missing. "
            "Add ADZUNA_APP_ID and ADZUNA_APP_KEY to your .env file."
        )

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(results_per_page, int) or not 1 <= results_per_page <= 50:
        raise ValueError("results_per_page must be between 1 and 50.")

    if not isinstance(page, int) or page < 1:
        raise ValueError("Page must be a positive integer.")

    endpoint = (
        f"{ADZUNA_BASE_URL}/{ADZUNA_COUNTRY}/search/{page}"
    )

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": query.strip(),
        "content-type": "application/json",
    }

    if location and location.strip():
        params["where"] = location.strip()

    try:
        response = requests.get(
            endpoint,
            params=params,
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.Timeout as exc:
        raise AdzunaAPIError(
            "Adzuna API request timed out. Please try again."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        raise AdzunaAPIError(
            f"Adzuna returned HTTP {status_code}. "
            "Check your credentials and API access."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise AdzunaAPIError(
            f"Could not connect to Adzuna: {exc}"
        ) from exc

    except ValueError as exc:
        raise AdzunaAPIError(
            "Adzuna returned an invalid JSON response."
        ) from exc

    if not isinstance(data, dict):
        raise AdzunaAPIError(
            "Unexpected response format from Adzuna."
        )

    raw_jobs = data.get("results", [])

    if not isinstance(raw_jobs, list):
        raise AdzunaAPIError(
            "Adzuna returned an unexpected jobs format."
        )

    jobs = [
        normalize_job(job)
        for job in raw_jobs
        if isinstance(job, dict)
    ]

    return {
        "total_matches": data.get("count", len(jobs)),
        "jobs": jobs,
        "page": page,
        "query": query,
        "location": location,
        "source": "Adzuna",
    }
