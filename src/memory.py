
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT_DIR / "data" / "hiremind.db"


def get_connection():
    """Create a SQLite database connection."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_memory():
    """Create the conversation memory table if it does not exist."""

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_memory
            ON conversation_memory(session_id, id)
        """)


def save_message(session_id: str, role: str, content: str):
    """Save one message to persistent conversation memory."""

    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")

    if role not in {"user", "assistant"}:
        raise ValueError("Role must be user or assistant.")

    if not content or not content.strip():
        raise ValueError("Message content cannot be empty.")

    initialize_memory()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversation_memory
            (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content.strip()),
        )


def get_conversation_history(
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """Retrieve the most recent messages for a session."""

    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")

    if limit < 1:
        raise ValueError("Limit must be greater than zero.")

    initialize_memory()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM conversation_memory
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    # Reverse so messages are returned in chronological order.
    return [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


def clear_conversation(session_id: str):
    """Delete conversation messages for a particular session."""

    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")

    initialize_memory()

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM conversation_memory
            WHERE session_id = ?
            """,
            (session_id,),
        )
