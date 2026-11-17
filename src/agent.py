
import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.tools import (
    search_live_jobs,
    get_job_details,
    compare_jobs,
)


load_dotenv()


# Stores live jobs returned during the current application session.
# This allows the assistant to retrieve and compare jobs it has found.
LIVE_JOB_CACHE = {}


def search_jobs_tool(
    candidate_skills: list[str],
    candidate_summary: str,
    experience_years: float | None,
    top_k: int,
    location: str = "India",
) -> dict:
    """Search and rank real job listings using Adzuna."""

    result = search_live_jobs(
        candidate_skills=candidate_skills,
        candidate_summary=candidate_summary,
        experience_years=experience_years,
        top_k=top_k,
        location=location,
    )

    if isinstance(result, dict) and not result.get("error"):
        for job in result.get("jobs", []):
            job_id = job.get("job_id")
            if job_id:
                LIVE_JOB_CACHE[job_id] = job

    return result


def get_job_details_tool(job_id: str) -> dict:
    """Retrieve details for a previously discovered live or sample job."""

    if job_id in LIVE_JOB_CACHE:
        return LIVE_JOB_CACHE[job_id]

    return get_job_details(job_id)


def compare_jobs_tool(job_id_1: str, job_id_2: str) -> dict:
    """Compare two live jobs or two jobs in the sample dataset."""

    job1 = LIVE_JOB_CACHE.get(job_id_1)
    job2 = LIVE_JOB_CACHE.get(job_id_2)

    if job1 and job2:
        return {
            "job_1": {
                "job_id": job1.get("job_id"),
                "title": job1.get("title"),
                "company": job1.get("company"),
                "location": job1.get("location"),
                "skills": job1.get("skills", []),
                "experience_required": job1.get(
                    "experience_required"
                ),
                "job_url": job1.get("job_url"),
            },
            "job_2": {
                "job_id": job2.get("job_id"),
                "title": job2.get("title"),
                "company": job2.get("company"),
                "location": job2.get("location"),
                "skills": job2.get("skills", []),
                "experience_required": job2.get(
                    "experience_required"
                ),
                "job_url": job2.get("job_url"),
            },
        }

    # Preserve support for the original curated sample jobs.
    if not job1 and not job2:
        return compare_jobs(job_id_1, job_id_2)

    return {
        "error": (
            "Both jobs must be available in the current live-job "
            "results or in the sample dataset."
        )
    }


TOOL_FUNCTIONS = {
    "search_jobs": search_jobs_tool,
    "get_job_details": get_job_details_tool,
    "compare_jobs": compare_jobs_tool,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search and rank real job listings from Adzuna "
                "using a candidate's skills, summary, experience "
                "and preferred location. Use this when a candidate "
                "asks for job recommendations or suitable vacancies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate's known skills",
                    },
                    "candidate_summary": {
                        "type": "string",
                        "description": "Candidate's professional summary",
                    },
                    "experience_years": {
                        "type": ["number", "null"],
                        "description": "Years of experience, if known",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of jobs to return (1-10)",
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Preferred job location, such as India, "
                            "Noida, or Bangalore"
                        ),
                    },
                },
                "required": [
                    "candidate_skills",
                    "candidate_summary",
                    "experience_years",
                    "top_k",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": (
                "Retrieve details for a job ID returned by a previous "
                "job search or belonging to the sample dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Unique job ID",
                    }
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_jobs",
            "description": (
                "Compare two jobs returned by a previous search, "
                "or two jobs from the sample dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id_1": {"type": "string"},
                    "job_id_2": {"type": "string"},
                },
                "required": ["job_id_1", "job_id_2"],
            },
        },
    },
]


class RecruitmentAgent:
    """LLM agent with controlled access to recruitment tools."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env.")

        self.client = Groq(api_key=api_key)

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    def run(
        self,
        user_message: str,
        candidate_profile: dict | None = None,
        conversation_history: list | None = None,
    ) -> dict:
        """Run the agent and execute its requested tools."""

        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        profile = candidate_profile or {}

        profile_location = (
            profile.get("location")
            or profile.get("preferred_location")
            or "India"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are HireMind AI, an AI recruitment assistant. "
                    "Use the available tools whenever job data is needed. "
                    "The search_jobs tool retrieves real job listings "
                    "from Adzuna. Never invent job listings, companies, "
                    "application links, skills or job details. "
                    "Clearly distinguish unavailable information from "
                    "verified information. "
                    "Use the candidate profile when relevant. "
                    "Do not claim that a candidate has a skill unless "
                    "it appears in their provided profile. "
                    "Job match scores are heuristic ranking scores, "
                    "not probabilities of getting hired. "
                    "When presenting a job, include its application "
                    "link if one is available."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Candidate profile:\n"
                    + json.dumps(profile, ensure_ascii=False)
                ),
            },
        ]

        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({
            "role": "user",
            "content": user_message,
        })

        tool_calls_log = []

        # Limit tool-calling rounds to avoid an endless loop.
        for _ in range(5):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                return {
                    "answer": assistant_message.content or "",
                    "tool_calls": tool_calls_log,
                    "messages": messages[2:],
                }

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                arguments = {}

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    function = TOOL_FUNCTIONS.get(function_name)

                    if function is None:
                        raise ValueError("Unknown tool requested.")

                    # Use the candidate's location when the model
                    # does not provide one explicitly.
                    if function_name == "search_jobs":
                        arguments.setdefault(
                            "location",
                            profile_location,
                        )

                    result = function(**arguments)

                except Exception as exc:
                    result = {
                        "error": f"Tool execution failed: {str(exc)}"
                    }

                tool_calls_log.append({
                    "tool": function_name,
                    "arguments": arguments,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                })

        return {
            "answer": (
                "I couldn't complete the request within the "
                "allowed tool-call limit. Please try a simpler query."
            ),
            "tool_calls": tool_calls_log,
            "messages": messages[2:],
        }
