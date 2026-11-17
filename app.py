
import html
import traceback
import uuid

import streamlit as st

from src.resume_parser import extract_text_from_pdf
from src.llm_extractor import extract_resume_profile
from src.tools import search_live_jobs
from src.agent import RecruitmentAgent
from src.memory import (
    initialize_memory,
    save_message,
    get_conversation_history,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HireMind AI | Smart Recruitment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --background: #0b1020;
        --surface: #121a2b;
        --surface-light: #192338;
        --border: #27334a;
        --text: #edf1fa;
        --muted: #9aa8bf;
        --accent: #a78bfa;
        --accent-light: #c4b5fd;
        --pink: #f0abfc;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, #25204a 0%, transparent 35%),
            var(--background);
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    p, label, li {
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0e1525;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.55rem;
    }

    [data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 30px 34px;
        border: 1px solid #37335b;
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            rgba(83, 61, 140, 0.35),
            rgba(27, 36, 60, 0.88)
        );
        margin-bottom: 24px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.55rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin: 0;
    }

    .hero-subtitle {
        color: #c4cde0;
        font-size: 1rem;
        margin-top: 10px;
        margin-bottom: 0;
    }

    .eyebrow {
        color: var(--accent-light);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 105px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
    }

    .profile-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .profile-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: white;
    }

    .muted-text {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .skill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .skill-chip {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid #51427c;
        border-radius: 20px;
        background: rgba(167, 139, 250, 0.12);
        color: #d9ccff;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .job-card-title {
        font-family: 'Space Grotesk', sans-serif;
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .job-meta {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 12px;
    }

    .job-score {
        background: rgba(52, 211, 153, 0.10);
        border: 1px solid rgba(52, 211, 153, 0.30);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    .job-score-label {
        color: #9caeae;
        font-size: 0.72rem;
    }

    .job-score-value {
        color: #6ee7b7;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .divider-line {
        height: 1px;
        background: var(--border);
        margin: 20px 0;
    }

    .stButton button,
    .stLinkButton a {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(100deg, #7c5ce0, #a78bfa);
        border: none;
        color: white;
    }

    .stButton button[kind="primary"]:hover {
        background: linear-gradient(100deg, #9275ed, #b8a4ff);
        border: none;
    }

    .stLinkButton a {
        background: linear-gradient(100deg, #7c5ce0, #a78bfa);
        border: none;
        color: white !important;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(18, 26, 43, 0.75);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stChatInput"] textarea {
        border-radius: 10px;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

initialize_memory()

DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "profile": None,
    "resume_text": "",
    "agent": None,
    "ranked_jobs": [],
    "total_matches": 0,
    "search_error": None,
    "job_source": None,
    "search_attempted": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = get_conversation_history(
        st.session_state.session_id
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_profile_dict(profile):
    if profile is None:
        return {}

    if hasattr(profile, "model_dump"):
        return profile.model_dump()

    if isinstance(profile, dict):
        return profile

    return {}


def get_profile_value(profile_data, key, default=None):
    value = profile_data.get(key, default)
    return default if value is None else value


def format_list(value):
    if isinstance(value, list):
        if not value:
            return "Not provided"

        items = []

        for item in value:
            if isinstance(item, dict):
                formatted = ", ".join(
                    f"{key}: {val}"
                    for key, val in item.items()
                    if val not in (None, "", [])
                )
                items.append(formatted)
            else:
                items.append(str(item))

        return "\n".join(f"• {item}" for item in items)

    if isinstance(value, dict):
        return "\n".join(
            f"**{key}:** {val}"
            for key, val in value.items()
            if val not in (None, "", [])
        )

    return str(value) if value else "Not provided"


def extract_candidate_skills(profile_data):
    skills = profile_data.get("skills", [])

    if isinstance(skills, str):
        skills = [
            skill.strip()
            for skill in skills.replace(",", "\n").splitlines()
            if skill.strip()
        ]

    if not isinstance(skills, list):
        return []

    cleaned_skills = []

    for skill in skills:
        if isinstance(skill, dict):
            skill_name = skill.get("name") or skill.get("skill")
            if skill_name:
                cleaned_skills.append(str(skill_name).strip())
        elif skill:
            cleaned_skills.append(str(skill).strip())

    return list(dict.fromkeys(cleaned_skills))


def get_candidate_summary(profile_data):
    summary = profile_data.get("summary", "")

    if summary:
        return str(summary)

    parts = []

    for key in ("name", "education", "experience", "projects", "skills"):
        value = profile_data.get(key)
        if value:
            parts.append(f"{key}: {format_list(value)}")

    return "\n".join(parts)


def normalize_job_results(search_result):
    if isinstance(search_result, dict):
        jobs = search_result.get("jobs", [])
        if not isinstance(jobs, list):
            jobs = []

        return jobs, search_result.get("total_matches", len(jobs))

    if isinstance(search_result, list):
        return search_result, len(search_result)

    return [], 0


def get_job_score(job):
    for key in (
        "match_percentage",
        "match_score",
        "final_score",
        "score",
    ):
        if job.get(key) is not None:
            return job[key]

    return None


def render_skill_chips(skills):
    if not skills:
        st.caption("No skills available.")
        return

    chips = "".join(
        f'<span class="skill-chip">{html.escape(str(skill))}</span>'
        for skill in skills
    )

    st.markdown(
        f'<div class="skill-container">{chips}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_agent():
    if st.session_state.agent is None:
        st.session_state.agent = RecruitmentAgent()

    return st.session_state.agent


def display_job(job, index):
    if not isinstance(job, dict):
        st.write(job)
        return

    title = (
        job.get("title")
        or job.get("job_title")
        or job.get("role")
        or "Job"
    )

    company = job.get("company", "Company not specified")
    location = job.get("location", "Location not specified")
    job_id = job.get("job_id") or job.get("id", "")
    description = job.get("description", "")
    source = job.get("source", "Adzuna")

    score = get_job_score(job)
    score_text = "N/A"

    if score is not None:
        try:
            score_value = float(score)
            if 0 <= score_value <= 1:
                score_value *= 100
            score_text = f"{score_value:.1f}%"
        except (ValueError, TypeError):
            score_text = str(score)

    with st.container(border=True):
        left, right = st.columns([5, 1.25])

        with left:
            st.markdown(
                f'<div class="job-card-title">{html.escape(str(title))}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="job-meta">'
                f'{html.escape(str(company))} &nbsp; · &nbsp; '
                f'{html.escape(str(location))}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f"""
                <div class="job-score">
                    <div class="job-score-label">MATCH SCORE</div>
                    <div class="job-score-value">
                        {html.escape(score_text)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if job_id:
            st.caption(f"Job ID: {job_id}  ·  Source: {source}")

        skills = (
            job.get("required_skills")
            or job.get("skills")
            or []
        )

        if skills:
            st.markdown("**Relevant skills**")
            if isinstance(skills, list):
                render_skill_chips(skills)
            else:
                st.write(skills)

        experience = (
            job.get("experience_required")
            or job.get("experience")
        )

        if experience:
            st.caption(f"Experience: {experience}")

        if description:
            with st.expander("Read job description"):
                st.write(description)

        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        currency = job.get("salary_currency") or ""

        if salary_min is not None or salary_max is not None:
            try:
                if salary_min is not None and salary_max is not None:
                    st.caption(
                        f"Salary: {currency} "
                        f"{float(salary_min):,.0f} – "
                        f"{float(salary_max):,.0f}"
                    )
                elif salary_min is not None:
                    st.caption(
                        f"Minimum salary: {currency} {salary_min}"
                    )
                else:
                    st.caption(
                        f"Maximum salary: {currency} {salary_max}"
                    )
            except (ValueError, TypeError):
                pass

        job_url = job.get("job_url")

        if (
            isinstance(job_url, str)
            and job_url.startswith(("https://", "http://"))
        ):
            st.link_button(
                "View Job / Apply ↗",
                job_url,
                use_container_width=True,
            )
        else:
            st.caption("Application link not available.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 HireMind AI")
    st.caption("Your AI-powered career companion")

    st.divider()

    st.markdown("### Build your profile")
    st.caption("Upload your resume to get started.")

    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
    )

    if st.button(
        "✨ Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner("Reading your resume..."):
                resume_bytes = uploaded_file.getvalue()
                extracted_text = extract_text_from_pdf(resume_bytes)

            if not extracted_text or not extracted_text.strip():
                st.error(
                    "No readable text found. "
                    "Please upload a text-based PDF."
                )
            else:
                with st.spinner("AI is analyzing your profile..."):
                    profile = extract_resume_profile(extracted_text)

                st.session_state.resume_text = extracted_text
                st.session_state.profile = profile

                st.session_state.ranked_jobs = []
                st.session_state.total_matches = 0
                st.session_state.search_error = None
                st.session_state.search_attempted = False
                st.session_state.job_source = None

                st.success("Resume analyzed successfully!")
                st.rerun()

        except Exception as exc:
            st.error(f"Resume analysis failed: {exc}")

    if st.session_state.profile is not None:
        st.success("✓ Your profile is ready")

        if st.button(
            "Clear uploaded profile",
            use_container_width=True,
        ):
            st.session_state.profile = None
            st.session_state.resume_text = ""
            st.session_state.ranked_jobs = []
            st.session_state.total_matches = 0
            st.session_state.search_error = None
            st.session_state.search_attempted = False
            st.session_state.job_source = None
            st.rerun()

    st.divider()

    st.markdown("**HireMind AI**")
    st.caption("Smart recruitment · MVP")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered recruitment platform</div>
        <div class="hero-title">🧠 HireMind AI</div>
        <p class="hero-subtitle">
            Discover opportunities. Understand your skills.
            Make your next career move with AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

profile_tab, recommendations_tab, assistant_tab = st.tabs(
    [
        "👤  Candidate Profile",
        "💼  Job Recommendations",
        "🤖  AI Assistant",
    ]
)


# ============================================================
# TAB 1: CANDIDATE PROFILE
# ============================================================

with profile_tab:

    st.header("Your Candidate Profile")

    if st.session_state.profile is None:
        st.info(
            "Upload your PDF resume using the sidebar "
            "and select Analyze Resume to create your profile."
        )

    else:
        profile_data = get_profile_dict(st.session_state.profile)

        name = get_profile_value(
            profile_data, "name", "Candidate"
        )

        email = get_profile_value(
            profile_data, "email", "Not provided"
        )

        phone = get_profile_value(
            profile_data, "phone", "Not provided"
        )

        skills = extract_candidate_skills(profile_data)

        summary = get_profile_value(
            profile_data,
            "summary",
            "No professional summary extracted.",
        )

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="section-label">Candidate overview</div>
                <div class="profile-name">
                    {html.escape(str(name))}
                </div>
                <div class="muted-text">
                    {html.escape(str(email))}
                    &nbsp; · &nbsp;
                    {html.escape(str(phone))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            render_metric_card("Skills detected", len(skills))

        with metric2:
            render_metric_card(
                "Education records",
                len(profile_data.get("education") or [])
                if isinstance(profile_data.get("education"), list)
                else ("Available" if profile_data.get("education") else 0),
            )

        with metric3:
            render_metric_card(
                "Experience records",
                len(profile_data.get("experience") or [])
                if isinstance(profile_data.get("experience"), list)
                else ("Available" if profile_data.get("experience") else 0),
            )

        st.markdown("---")

        st.subheader("Professional Summary")
        st.write(summary)

        st.subheader("Technical Skills")
        render_skill_chips(skills)

        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.subheader("🎓 Education")
                st.markdown(
                    format_list(
                        get_profile_value(profile_data, "education")
                    )
                )

        with right:
            with st.container(border=True):
                st.subheader("💼 Experience")
                st.markdown(
                    format_list(
                        get_profile_value(profile_data, "experience")
                    )
                )

        with st.container(border=True):
            st.subheader("🚀 Projects")
            st.markdown(
                format_list(
                    get_profile_value(profile_data, "projects")
                )
            )

        with st.expander("View complete extracted profile"):
            st.json(profile_data)


# ============================================================
# TAB 2: LIVE JOB RECOMMENDATIONS
# ============================================================

with recommendations_tab:

    st.header("Discover Relevant Opportunities")

    st.caption(
        "Live job listings are retrieved from Adzuna and ranked "
        "against your candidate profile."
    )

    if st.session_state.profile is None:
        st.info("Analyze your resume first to receive personalized jobs.")

    else:
        profile_data = get_profile_dict(st.session_state.profile)

        candidate_skills = extract_candidate_skills(profile_data)
        candidate_summary = get_candidate_summary(profile_data)

        if candidate_skills:
            st.markdown("**Your detected skills**")
            render_skill_chips(candidate_skills)

        if not candidate_skills:
            st.warning(
                "No skills were extracted. "
                "Job matching may be less accurate."
            )

        col1, col2 = st.columns([1, 2])

        with col1:
            top_k = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=10,
                value=5,
            )

        with col2:
            location = st.text_input(
                "Preferred job location",
                value="India",
                help="Examples: India, Noida, Bangalore, Remote",
            )

        if st.button(
            "🔎 Find Matching Jobs",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.search_attempted = True
            st.session_state.search_error = None
            st.session_state.ranked_jobs = []
            st.session_state.total_matches = 0
            st.session_state.job_source = None

            if not location.strip():
                st.session_state.search_error = (
                    "Please enter a preferred job location."
                )

            elif not candidate_skills and not candidate_summary.strip():
                st.session_state.search_error = (
                    "No usable candidate information was found."
                )

            else:
                try:
                    with st.spinner(
                        "Searching live opportunities and ranking matches..."
                    ):
                        search_result = search_live_jobs(
                            candidate_skills=candidate_skills,
                            candidate_summary=candidate_summary,
                            experience_years=None,
                            top_k=top_k,
                            location=location.strip(),
                        )

                    if not isinstance(search_result, dict):
                        st.session_state.search_error = (
                            "Unexpected response from the job search service."
                        )

                    elif search_result.get("error"):
                        st.session_state.search_error = (
                            search_result["error"]
                        )

                    else:
                        jobs, total_matches = normalize_job_results(
                            search_result
                        )

                        st.session_state.ranked_jobs = jobs
                        st.session_state.total_matches = total_matches
                        st.session_state.job_source = (
                            search_result.get("source", "Adzuna")
                        )

                except Exception as exc:
                    st.session_state.search_error = str(exc)

        if st.session_state.search_error:
            st.error(
                f"Live job search failed: {st.session_state.search_error}"
            )
            st.info(
                "Check your Adzuna credentials and internet connection."
            )

        elif st.session_state.ranked_jobs:
            results = st.session_state.ranked_jobs

            st.markdown("---")

            st.success(
                f"{st.session_state.total_matches} matching "
                f"opportunities found."
            )

            st.caption(
                f"Source: {st.session_state.job_source} · "
                "Match scores are heuristic ranking scores, "
                "not probabilities of getting hired."
            )

            for index, job in enumerate(results, start=1):
                display_job(job, index)

        elif st.session_state.search_attempted:
            st.info(
                "No matching live jobs were found. "
                "Try another location or search again later."
            )

        else:
            st.info(
                "Your profile is ready. Select Find Matching Jobs "
                "to discover opportunities."
            )


# ============================================================
# TAB 3: AI RECRUITMENT ASSISTANT
# ============================================================

with assistant_tab:

    st.header("Your AI Recruitment Assistant")

    st.caption(
        "Ask about job opportunities, skill requirements, "
        "career preparation and job comparisons."
    )

    if st.session_state.profile is None:
        st.info(
            "You can chat without uploading a resume. "
            "Upload one to enable personalized assistance."
        )

    # Display conversation history.
    for message in st.session_state.chat_messages:
        if message.get("role") not in ("user", "assistant"):
            continue

        with st.chat_message(message["role"]):
            st.markdown(message.get("content", ""))

    user_message = st.chat_input(
        "Ask HireMind AI anything about your career..."
    )

    if user_message:

        with st.chat_message("user"):
            st.markdown(user_message)

        save_message(
            st.session_state.session_id,
            "user",
            user_message,
        )

        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message,
        })

        try:
            with st.chat_message("assistant"):

                with st.spinner("HireMind AI is thinking..."):

                    agent = get_agent()

                    profile_data = get_profile_dict(
                        st.session_state.profile
                    )

                    history = get_conversation_history(
                        st.session_state.session_id
                    )

                    # Current message is already passed separately.
                    if (
                        history
                        and history[-1].get("role") == "user"
                        and history[-1].get("content") == user_message
                    ):
                        history = history[:-1]

                    response = agent.run(
                        user_message=user_message,
                        candidate_profile=profile_data or None,
                        conversation_history=history,
                    )

                    if isinstance(response, dict):
                        assistant_response = (
                            response.get("response")
                            or response.get("answer")
                            or response.get("content")
                            or str(response)
                        )
                    else:
                        assistant_response = str(response)

                    st.markdown(assistant_response)

            save_message(
                st.session_state.session_id,
                "assistant",
                assistant_response,
            )

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": assistant_response,
            })

        except Exception:
            error_message = (
                "I couldn't complete that request. "
                "Please try again. If the issue continues, "
                "check your Groq API configuration."
            )

            st.error(error_message)

            save_message(
                st.session_state.session_id,
                "assistant",
                error_message,
            )

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_message,
            })

            with st.expander("Developer error details"):
                st.code(traceback.format_exc())


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:10px 0;">
        <div style="color:#c4b5fd; font-weight:700;">
            🧠 HireMind AI
        </div>
        <div style="color:#8793aa; font-size:0.8rem;">
            AI-powered recruitment assistant · MVP
        </div>
        <div style="color:#65718a; font-size:0.72rem; margin-top:6px;">
            Live job listings are retrieved from Adzuna.
            Availability and application status may change.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
