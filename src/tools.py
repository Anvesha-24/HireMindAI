
from functools import lru_cache

from src.vector_store import (
    JobVectorStore,
    create_job_vector_store,
    load_jobs,
)
from src.job_ranker import rank_jobs
from src.job_fetcher import search_adzuna_jobs, AdzunaAPIError


@lru_cache(maxsize=1)
def get_vector_store():
    """Load and cache the curated sample job vector store."""
    return create_job_vector_store()


def search_jobs(
    candidate_skills: list[str],
    candidate_summary: str = "",
    experience_years: float | None = None,
    top_k: int = 5,
) -> dict:
    """Search and rank jobs from the curated sample dataset."""

    if not candidate_skills and not candidate_summary.strip():
        return {"error": "Provide candidate skills or a summary."}

    if top_k < 1 or top_k > 10:
        return {"error": "top_k must be between 1 and 10."}

    query = (
        candidate_summary
        + " Candidate skills: "
        + ", ".join(candidate_skills)
    ).strip()

    store = get_vector_store()

    semantic_results = store.search(
        query,
        top_k=len(store.jobs),
    )

    ranked_results = rank_jobs(
        {"skills": candidate_skills},
        semantic_results,
        candidate_experience_years=experience_years,
    )

    return {
        "total_matches": len(ranked_results[:top_k]),
        "jobs": ranked_results[:top_k],
        "source": "Sample dataset",
    }


def search_live_jobs(
    candidate_skills: list[str],
    candidate_summary: str = "",
    experience_years: float | None = None,
    top_k: int = 5,
    location: str = "India",
) -> dict:
    """
    Fetch live jobs from Adzuna and rank them against
    the candidate's skills and resume summary.
    """

    # Validate candidate information.
    if not candidate_skills and not candidate_summary.strip():
        return {"error": "Provide candidate skills or a summary."}

    if top_k < 1 or top_k > 10:
        return {"error": "top_k must be between 1 and 10."}

    if not location or not location.strip():
        return {"error": "Provide a valid job location."}

    # Normalize candidate skills.
    clean_skills = [
        str(skill).strip()
        for skill in (candidate_skills or [])
        if skill and str(skill).strip()
    ]

    # Use a concise search keyword for Adzuna.
    # Avoid sending the entire resume as an API query.
    if clean_skills:
        search_query = clean_skills[0]
    else:
        search_query = " ".join(candidate_summary.split()[:3])

    # Keep the complete candidate information for semantic ranking.
    ranking_query = (
        candidate_summary
        + " Candidate skills: "
        + ", ".join(clean_skills)
    ).strip()

    # Fetch live jobs.
    try:
        response = search_adzuna_jobs(
            query=search_query,
            location=location.strip(),
            results_per_page=50,
        )

    except AdzunaAPIError as error:
        return {"error": str(error)}

    raw_jobs = response.get("jobs", [])

    if not raw_jobs:
        return {
            "total_matches": 0,
            "available_results": response.get("total_matches", 0),
            "jobs": [],
            "source": "Adzuna",
            "message": "No live jobs were found for this search.",
        }

    # Prepare live jobs for the existing ranking pipeline.
    jobs = []

    for raw_job in raw_jobs:
        job = raw_job.copy()

        description = str(
            job.get("description") or ""
        ).casefold()

        # Adzuna often does not provide structured skill fields.
        # Identify candidate skills explicitly mentioned in
        # the job description.
        mentioned_skills = [
            skill
            for skill in clean_skills
            if skill.casefold() in description
        ]

        job["skills"] = mentioned_skills

        job["experience_required"] = (
            job.get("experience_required") or ""
        )

        jobs.append(job)

    # Generate embeddings for the retrieved live jobs.
    live_vector_store = JobVectorStore(jobs)

    semantic_results = live_vector_store.search(
        ranking_query,
        top_k=len(jobs),
    )

    # Rank jobs using the existing heuristic ranker.
    ranked_results = rank_jobs(
        {"skills": clean_skills},
        semantic_results,
        candidate_experience_years=experience_years,
    )

    return {
        "total_matches": len(ranked_results[:top_k]),
        "available_results": response.get(
            "total_matches",
            len(raw_jobs),
        ),
        "jobs": ranked_results[:top_k],
        "source": "Adzuna",
        "location": location.strip(),
        "search_query": search_query,
    }


def get_job_details(job_id: str) -> dict:
    """Retrieve a job from the curated sample dataset."""

    jobs = load_jobs()

    for job in jobs:
        if job["job_id"].lower() == job_id.lower():
            return job

    return {"error": f"No job found with ID {job_id}."}


def compare_jobs(job_id_1: str, job_id_2: str) -> dict:
    """Compare two jobs using their available information."""

    job1 = get_job_details(job_id_1)
    job2 = get_job_details(job_id_2)

    if "error" in job1:
        return job1

    if "error" in job2:
        return job2

    skills1 = {
        skill.lower()
        for skill in job1.get("skills", [])
    }

    skills2 = {
        skill.lower()
        for skill in job2.get("skills", [])
    }

    return {
        "job_1": {
            "job_id": job1["job_id"],
            "title": job1["title"],
            "company": job1["company"],
            "location": job1.get("location", "Not specified"),
            "experience_required": job1.get(
                "experience_required",
                "Not specified",
            ),
            "skills": job1.get("skills", []),
        },
        "job_2": {
            "job_id": job2["job_id"],
            "title": job2["title"],
            "company": job2["company"],
            "location": job2.get("location", "Not specified"),
            "experience_required": job2.get(
                "experience_required",
                "Not specified",
            ),
            "skills": job2.get("skills", []),
        },
        "common_skills": sorted(skills1 & skills2),
        "unique_to_job_1": sorted(skills1 - skills2),
        "unique_to_job_2": sorted(skills2 - skills1),
    }
