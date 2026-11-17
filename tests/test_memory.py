
import pytest

import src.memory as memory


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    database_file = tmp_path / "test_memory.db"

    monkeypatch.setattr(
        memory,
        "DATABASE_PATH",
        database_file,
    )

    return database_file


def test_save_and_retrieve_messages(temporary_database):
    memory.save_message("session-1", "user", "Find Python jobs")

    memory.save_message(
        "session-1",
        "assistant",
        "I found three relevant jobs.",
    )

    history = memory.get_conversation_history("session-1")

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_sessions_are_separate(temporary_database):
    memory.save_message("session-1", "user", "Hello")
    memory.save_message("session-2", "user", "Find AI jobs")

    history = memory.get_conversation_history("session-1")

    assert len(history) == 1
    assert history[0]["content"] == "Hello"


def test_history_limit(temporary_database):
    for i in range(5):
        memory.save_message("session-1", "user", f"Message {i}")

    history = memory.get_conversation_history("session-1", limit=2)

    assert len(history) == 2
    assert history[0]["content"] == "Message 3"
    assert history[1]["content"] == "Message 4"


def test_clear_conversation(temporary_database):
    memory.save_message("session-1", "user", "Hello")

    memory.clear_conversation("session-1")

    assert memory.get_conversation_history("session-1") == []


def test_invalid_role(temporary_database):
    with pytest.raises(ValueError, match="Role must be"):
        memory.save_message("session-1", "system", "Invalid role")


def test_invalid_limit(temporary_database):
    with pytest.raises(ValueError, match="Limit"):
        memory.get_conversation_history("session-1", limit=0)
