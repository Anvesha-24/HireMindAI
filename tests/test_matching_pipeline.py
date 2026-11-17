
from src.vector_store import create_job_vector_store
from src.job_ranker import rank_jobs


def test_resume_to_job_matching_pipeline():
    profile = {
        "skills": [
            "Python",
            "RAG",
            "LLMs",
            "Embeddings",
            "FastAPI",
        ],
        "summary": (
            "Built AI applications using Python, "
            "large language models and semantic retrieval."
        ),
    }

    store = create_job_vector_store()

    query = (
        profile["summary"]
        + " Skills: "
        + ", ".join(profile["skills"])
    )

    semantic_results = store.search(
        query,
        top_k=len(store.jobs),
    )

    ranked_results = rank_jobs(
        profile,
        semantic_results,
        candidate_experience_years=0,
    )

    assert len(ranked_results) == len(store.jobs)

    assert ranked_results[0]["final_score"] >= (
        ranked_results[-1]["final_score"]
    )

    assert ranked_results[0]["match_percentage"] > 0

    print("\nTop 5 job matches:")

    for job in ranked_results[:5]:
        print(
            job["title"],
            job["match_percentage"],
            job["matched_skills"],
        )
