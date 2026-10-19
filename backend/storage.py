"""Model layer: saves comments in a small SQLite database (one file, no server needed)."""
import sqlite3
import threading
from datetime import datetime, timezone

from .config import DATA_DIR

DB_PATH = DATA_DIR / "comments.db"
_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row            # rows behave like dicts
    return conn


def init_db():
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                comment_id  TEXT PRIMARY KEY,
                platform    TEXT,
                post_id     TEXT,
                author      TEXT,
                content     TEXT,
                status      TEXT DEFAULT 'pending',   -- pending / replied / error
                ai_reply    TEXT,
                received_at TEXT,
                replied_at  TEXT
            )""")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_comment(c: dict) -> dict:
    """Insert a new comment (or overwrite one with the same comment_id)."""
    with _lock, _connect() as conn:
        conn.execute("""INSERT OR REPLACE INTO comments
            (comment_id, platform, post_id, author, content, status, ai_reply, received_at)
            VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?)""",
            (c["comment_id"], c["platform"], c["post_id"], c["author"], c["content"], _now()))
    return get_comment(c["comment_id"])


def set_reply(comment_id: str, reply: str, status: str = "replied") -> dict:
    with _lock, _connect() as conn:
        conn.execute("UPDATE comments SET ai_reply=?, status=?, replied_at=? WHERE comment_id=?",
                     (reply, status, _now(), comment_id))
    return get_comment(comment_id)


def get_comment(comment_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM comments WHERE comment_id=?", (comment_id,)).fetchone()
    return dict(row) if row else None


def list_comments() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM comments ORDER BY received_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_all() -> int:
    with _lock, _connect() as conn:
        return conn.execute("DELETE FROM comments").rowcount
