
from unittest.mock import patch

import pytest

from src.agent import RecruitmentAgent


def test_empty_message():
    agent = RecruitmentAgent.__new__(RecruitmentAgent)

    with pytest.raises(ValueError, match="cannot be empty"):
        agent.run("   ")


@patch("src.agent.Groq")
def test_agent_initialization(mock_groq):
    with patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
        },
    ):
        agent = RecruitmentAgent()

    assert agent.model == "openai/gpt-oss-20b"
    mock_groq.assert_called_once()
