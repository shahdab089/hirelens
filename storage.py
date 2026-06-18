import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from core.schema import ApplicationRecord

# DB path is env-configurable so the container deploy can point it at a writable
# dir (e.g. /tmp on Hugging Face Spaces).
DB_PATH = os.environ.get("APP_DB_PATH", "applications.db")

VALID_OUTCOMES = {"rejected", "interview", "ghosted", "offer"}


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database and applies lightweight migrations."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL,
                outcome TEXT,
                client_id TEXT
            )
            """
        )
        # Add client_id to pre-existing tables that were created before it existed.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(applications)")}
        if "client_id" not in cols:
            conn.execute("ALTER TABLE applications ADD COLUMN client_id TEXT")
        conn.commit()
    finally:
        conn.close()


def save(record: ApplicationRecord, client_id: Optional[str] = None):
    """Saves an ApplicationRecord, optionally scoped to an anonymous client."""
    init_db()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO applications (id, created_at, data, outcome, client_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.created_at.isoformat(),
                record.model_dump_json(),
                record.outcome,
                client_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_all() -> List[ApplicationRecord]:
    """Loads every ApplicationRecord (used by the eval/CLI and tests)."""
    init_db()
    records = []
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT data FROM applications ORDER BY created_at DESC")
        for row in cursor:
            records.append(ApplicationRecord.model_validate_json(row["data"]))
    finally:
        conn.close()
    return records


def load_by_client(client_id: str) -> List[ApplicationRecord]:
    """Loads only the records belonging to one anonymous client (web app)."""
    init_db()
    records = []
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT data FROM applications WHERE client_id = ? ORDER BY created_at DESC",
            (client_id,),
        )
        for row in cursor:
            records.append(ApplicationRecord.model_validate_json(row["data"]))
    finally:
        conn.close()
    return records


def set_outcome(id: str, outcome: str):
    """Updates the outcome of an application (in both the column and JSON blob)."""
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"Invalid outcome '{outcome}'. Expected one of {sorted(VALID_OUTCOMES)}."
        )
    init_db()
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT data FROM applications WHERE id = ?", (id,))
        row = cursor.fetchone()
        if row:
            record = ApplicationRecord.model_validate_json(row["data"])
            record.outcome = outcome
            conn.execute(
                "UPDATE applications SET outcome = ?, data = ? WHERE id = ?",
                (outcome, record.model_dump_json(), id),
            )
            conn.commit()
    finally:
        conn.close()
