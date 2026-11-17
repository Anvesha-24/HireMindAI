
import json
from pathlib import Path

import numpy as np

from src.embeddings import generate_embeddings, cosine_similarity


ROOT_DIR = Path(__file__).resolve().parent.parent
JOBS_FILE = ROOT_DIR / "data" / "jobs.json"


def load_jobs() -> list[dict]:
    """Load the curated job dataset from JSON."""

    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"Job dataset not found: {JOBS_FILE}")

    with JOBS_FILE.open("r", encoding="utf-8") as file:
        jobs = json.load(file)

    if not isinstance(jobs, list):
        raise ValueError("Job dataset must contain a JSON list.")

    required_fields = {
        "job_id",
        "title",
        "company",
        "skills",
        "description",
    }

    for job in jobs:
        missing = required_fields - job.keys()

        if missing:
            raise ValueError(
                f"Job {job.get('job_id', 'unknown')} is missing: {missing}"
            )

    return jobs


def build_job_text(job: dict) -> str:
    """Combine relevant job fields into one searchable document."""

    skills = ", ".join(job.get("skills", []))

    return (
        f"Job title: {job['title']}. "
        f"Skills: {skills}. "
        f"Description: {job['description']}"
    )


class JobVectorStore:
    """Simple in-memory semantic search over job descriptions."""

    def __init__(self, jobs: list[dict]):
        if not jobs:
            raise ValueError("Job collection cannot be empty.")

        self.jobs = jobs

        job_texts = [build_job_text(job) for job in jobs]

        self.embeddings = generate_embeddings(job_texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top matching jobs for a text query."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = generate_embeddings([query])[0]

        scores = cosine_similarity(
            query_embedding,
            self.embeddings,
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            job = self.jobs[int(index)].copy()
            job["semantic_score"] = float(scores[index])
            results.append(job)

        return results


def create_job_vector_store() -> JobVectorStore:
    """Load jobs and construct the in-memory vector store."""

    return JobVectorStore(load_jobs())
