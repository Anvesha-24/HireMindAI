
import json
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError


load_dotenv()


class Experience(BaseModel):
    company: str = "Not specified"
    role: str = "Not specified"
    duration: str = "Not specified"
    description: str = ""


class Education(BaseModel):
    institution: str = "Not specified"
    degree: str = "Not specified"
    graduation_year: str = "Not specified"


class ResumeProfile(BaseModel):
    name: str = "Not specified"
    email: str = "Not specified"
    phone: str = "Not specified"

    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)

    summary: str = ""


def get_groq_client() -> Groq:
    """Create a Groq client using the API key from environment variables."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    return Groq(api_key=api_key)


def extract_resume_profile(resume_text: str) -> ResumeProfile:
    """
    Extract structured information from resume text using an LLM.

    Returns a Pydantic-validated ResumeProfile.
    """

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    system_prompt = """
    You are an expert resume information extraction system.

    Extract information only when it is supported by the resume.
    Do not invent skills, companies, dates, degrees, or achievements.

    Return valid JSON with these exact keys:
    {
        "name": "string",
        "email": "string",
        "phone": "string",
        "skills": ["string"],
        "experience": [
            {
                "company": "string",
                "role": "string",
                "duration": "string",
                "description": "string"
            }
        ],
        "education": [
            {
                "institution": "string",
                "degree": "string",
                "graduation_year": "string"
            }
        ],
        "projects": ["string"],
        "summary": "string"
    }

    Use "Not specified" for unavailable scalar information.
    Use empty arrays when no information is available.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Extract the structured profile from this resume:\n\n"
                    + resume_text[:20000]
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The LLM returned an empty response.")

    try:
        data = json.loads(content)
        return ResumeProfile.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Could not validate the extracted resume data: {exc}"
        ) from exc
