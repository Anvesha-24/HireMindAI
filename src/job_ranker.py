
import re


# Initial heuristic weights — these are configurable design choices,
# not weights learned from training data.
SEMANTIC_WEIGHT = 0.55
SKILL_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.15


def normalize_skill(skill: str) -> str:
    """Normalize skill names for case-insensitive comparison."""
    return skill.strip().lower()


def calculate_skill_match(
    candidate_skills: list[str],
    required_skills: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Calculate the fraction of job skills covered by the candidate.

    Returns:
        score, matched_skills, missing_skills
    """

    candidate_set = {
        normalize_skill(skill)
        for skill in candidate_skills
        if skill.strip()
    }

    required_set = {
        normalize_skill(skill)
        for skill in required_skills
        if skill.strip()
    }

    if not required_set:
        return 0.0, [], []

    matched = candidate_set.intersection(required_set)
    missing = required_set - candidate_set

    score = len(matched) / len(required_set)

    return (
        score,
        sorted(matched),
        sorted(missing),
    )


def calculate_experience_match(
    candidate_years: float | None,
    required_experience: str,
) -> float:
    """
    Estimate experience compatibility using the job's stated range.

    Missing candidate experience is treated as unknown, not as zero.
    """

    if candidate_years is None:
        return 0.5

    requirement = (required_experience or "").lower()

    if "intern" in requirement:
        return 1.0 if candidate_years <= 1 else 0.8

    numbers = re.findall(r"\d+(?:\.\d+)?", requirement)

    if not numbers:
        return 0.5

    if len(numbers) >= 2:
        minimum = float(numbers[0])
        maximum = float(numbers[1])
    else:
        minimum = 0.0
        maximum = float(numbers[0])

    if minimum <= candidate_years <= maximum:
        return 1.0

    if candidate_years < minimum:
        return 0.3

    return 0.8


def rank_jobs(
    candidate_profile: dict,
    semantic_results: list[dict],
    candidate_experience_years: float | None = None,
) -> list[dict]:
    """
    Combine semantic relevance, skill overlap and experience compatibility.

    Returns jobs sorted by their final score in descending order.
    """

    candidate_skills = candidate_profile.get("skills", [])

    ranked_jobs = []

    for job in semantic_results:
        semantic_score = max(
            0.0,
            min(1.0, float(job.get("semantic_score", 0.0))),
        )

        skill_score, matched, missing = calculate_skill_match(
            candidate_skills,
            job.get("skills", []),
        )

        experience_score = calculate_experience_match(
            candidate_experience_years,
            job.get("experience_required", ""),
        )

        final_score = (
            SEMANTIC_WEIGHT * semantic_score
            + SKILL_WEIGHT * skill_score
            + EXPERIENCE_WEIGHT * experience_score
        )

        result = job.copy()

        result.update({
            "semantic_score": semantic_score,
            "skill_match_score": skill_score,
            "experience_match_score": experience_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "final_score": round(final_score, 4),
            "match_percentage": round(final_score * 100, 2),
        })

        ranked_jobs.append(result)

    return sorted(
        ranked_jobs,
        key=lambda item: item["final_score"],
        reverse=True,
    )
