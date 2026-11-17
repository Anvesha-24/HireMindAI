
import pytest

from src.llm_extractor import (
    ResumeProfile,
    extract_resume_profile,
)


def test_resume_profile_defaults():
    profile = ResumeProfile()

    assert profile.name == "Not specified"
    assert profile.skills == []
    assert profile.experience == []
    assert profile.education == []


def test_resume_profile_validation():
    profile = ResumeProfile(
        name="Anvesha",
        skills=["Python", "C++"],
        projects=["HireMind AI"],
    )

    assert profile.name == "Anvesha"
    assert "Python" in profile.skills
    assert len(profile.projects) == 1


def test_empty_resume_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_resume_profile("   ")
