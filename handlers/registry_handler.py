# handlers/registry_handler.py
import hashlib
import logging
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("registry.db")


def init_registry():
    """Initialize the SQLite DB if not present."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artifact_id TEXT UNIQUE,
        name TEXT,
        score REAL,
        tags TEXT,
        code_url TEXT,
        dataset_url TEXT,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    conn.commit()
    conn.close()


def list_models(limit=50, offset=0):
    """Return a list of all models (paginated) with URLs."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name, score, tags, code_url, dataset_url, created_at
        FROM models
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """,
        (limit, offset),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "name": r[0],
            "score": r[1],
            "tags": r[2],
            "code_url": r[3],
            "dataset_url": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def search_models(query):
    """Regex or substring search on model name, tags, or URLs."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name, score, tags, code_url, dataset_url, created_at
        FROM models
    """
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    pattern = re.compile(query, re.IGNORECASE)
    for name, score, tags, code_url, dataset_url, created_at in rows:
        text_blob = " ".join(str(x) for x in [name, tags, code_url, dataset_url])
        if pattern.search(text_blob):
            results.append(
                {
                    "name": name,
                    "score": score,
                    "tags": tags,
                    "code_url": code_url,
                    "dataset_url": dataset_url,
                    "created_at": created_at,
                }
            )
    return results


def add_model(
    name, score, tags="", code_url="unknown", dataset_url="unknown", metadata_json="{}"
):
    artifact_id = str(
        abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10)
    )
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO models (artifact_id, name, score, tags, code_url, dataset_url, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (artifact_id, name, score, tags, code_url, dataset_url, metadata_json),
    )
    conn.commit()
    conn.close()


def reset_registry():
    """
    Completely clears the local registry database and any stored model files.
    Called by CLI or web API 'reset' command.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Drop all rows (not the table schema)
    cur.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    # Optional: also clear downloaded model folders
    import shutil
    from pathlib import Path

    downloads_dir = Path("downloaded_models")
    if downloads_dir.exists():
        shutil.rmtree(downloads_dir, ignore_errors=True)
        downloads_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Registry has been reset successfully.")
