import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path("registry.db")

# Configure logging
logger = logging.getLogger(__name__)


def generate_artifact_id(name: str) -> str:
    """
    Generate deterministic 10-digit artifact ID from name.
    """
    return str(abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10))


def init_registry():
    """
    Initialize the SQLite database with support for multiple artifact types.
    Creates tables if they don't exist and performs schema upgrades if needed.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create main models table (original schema)
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

    # Check if artifact_type column exists, add if not
    cur.execute("PRAGMA table_info(models)")
    columns = [col[1] for col in cur.fetchall()]

    if "artifact_type" not in columns:
        logger.info("Upgrading schema: Adding artifact_type column")
        cur.execute("ALTER TABLE models ADD COLUMN artifact_type TEXT DEFAULT 'model'")

    if "url" not in columns:
        logger.info("Upgrading schema: Adding url column")
        cur.execute("ALTER TABLE models ADD COLUMN url TEXT DEFAULT 'unknown'")

    if "updated_at" not in columns:
        logger.info("Upgrading schema: Adding updated_at column")
        cur.execute(
            "ALTER TABLE models ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )

    # Create relationships table for lineage tracking
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_artifact_id TEXT NOT NULL,
            to_artifact_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_artifact_id) REFERENCES models(artifact_id),
            FOREIGN KEY (to_artifact_id) REFERENCES models(artifact_id)
        )
    """
    )

    # Create index for faster lookups
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifact_id ON models(artifact_id)
    """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifact_type ON models(artifact_type)
    """
    )

    conn.commit()
    conn.close()
    logger.info("Registry initialized successfully")


def add_artifact(
    name: str,
    artifact_type: str,
    score: float = 0.0,
    url: str = "unknown",
    tags: str = "",
    code_url: str = "unknown",
    dataset_url: str = "unknown",
    metadata: Optional[Dict] = None,
) -> str:
    """
    Add an artifact to the registry.
    """
    artifact_id = generate_artifact_id(name)
    metadata_json = json.dumps(metadata) if metadata else "{}"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT OR REPLACE INTO models
            (artifact_id, artifact_type, name, score, tags, url, code_url,
             dataset_url, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                artifact_id,
                artifact_type,
                name,
                score,
                tags,
                url,
                code_url,
                dataset_url,
                metadata_json,
            ),
        )

        conn.commit()
        logger.info(f"Added {artifact_type} artifact: {name} (ID: {artifact_id})")
        return artifact_id

    except Exception as e:
        logger.error(f"Error adding artifact: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_artifact_by_id(
    artifact_id: str, artifact_type: Optional[str] = None
) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        if artifact_type:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at, updated_at
                FROM models
                WHERE artifact_id = ? AND artifact_type = ?
            """,
                (str(artifact_id), artifact_type),
            )
        else:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at, updated_at
                FROM models
                WHERE artifact_id = ?
            """,
                (str(artifact_id),),
            )

        row = cur.fetchone()

        if not row:
            return None

        return {
            "artifact_id": row[0],
            "artifact_type": row[1] if row[1] else "model",
            "name": row[2],
            "score": row[3],
            "tags": row[4],
            "url": row[5] if row[5] else row[6],
            "code_url": row[6],
            "dataset_url": row[7],
            "metadata_json": row[8],
            "created_at": row[9],
            "updated_at": row[10] if len(row) > 10 else row[9],
        }

    except Exception as e:
        logger.error(f"Error getting artifact {artifact_id}: {e}")
        return None
    finally:
        conn.close()


def artifact_exists(artifact_id: str) -> bool:
    """
    Check if an artifact exists in the registry.
    """
    return get_artifact_by_id(artifact_id) is not None


def update_artifact(artifact_id: str, **updates) -> bool:
    """
    Update an artifact's fields.
    """
    if not artifact_exists(artifact_id):
        logger.warning(f"Cannot update non-existent artifact {artifact_id}")
        return False

    allowed_fields = {
        "score",
        "tags",
        "url",
        "code_url",
        "dataset_url",
        "metadata_json",
        "artifact_type",
    }

    # Filter out invalid fields
    valid_updates = {k: v for k, v in updates.items() if k in allowed_fields}

    if not valid_updates:
        logger.warning(f"No valid fields to update for artifact {artifact_id}")
        return False

    # Build UPDATE query
    set_clause = ", ".join([f"{k} = ?" for k in valid_updates.keys()])
    set_clause += ", updated_at = CURRENT_TIMESTAMP"
    values = list(valid_updates.values()) + [str(artifact_id)]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(
            f"""
            UPDATE models
            SET {set_clause}
            WHERE artifact_id = ?
        """,
            values,
        )

        conn.commit()
        logger.info(f"Updated artifact {artifact_id}: {valid_updates}")
        return True

    except Exception as e:
        logger.error(f"Error updating artifact {artifact_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def list_artifacts(
    artifact_type: Optional[str] = None, limit: int = 50, offset: int = 0
) -> List[Dict]:
    """
    List artifacts with optional type filter and pagination.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        if artifact_type:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at
                FROM models
                WHERE artifact_type = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
                (artifact_type, limit, offset),
            )
        else:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at
                FROM models
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

        rows = cur.fetchall()

        return [
            {
                "artifact_id": r[0],
                "artifact_type": r[1] if r[1] else "model",
                "name": r[2],
                "score": r[3],
                "tags": r[4],
                "url": r[5] if r[5] else r[6],
                "code_url": r[6],
                "dataset_url": r[7],
                "metadata_json": r[8],
                "created_at": r[9],
            }
            for r in rows
        ]

    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        return []
    finally:
        conn.close()


def search_artifacts(query: str, artifact_type: Optional[str] = None) -> List[Dict]:
    """
    Search artifacts using regex pattern.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        if artifact_type:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at
                FROM models
                WHERE artifact_type = ?
            """,
                (artifact_type,),
            )
        else:
            cur.execute(
                """
                SELECT artifact_id, artifact_type, name, score, tags, url,
                       code_url, dataset_url, metadata_json, created_at
                FROM models
            """
            )

        rows = cur.fetchall()

        results = []
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {query} - {e}")
            return results

        for row in rows:
            text_blob = f"{row[2]} {row[4]} {row[5]} {row[6]} {row[7]}"
            if pattern.search(text_blob):
                results.append(
                    {
                        "artifact_id": row[0],
                        "artifact_type": row[1] if row[1] else "model",
                        "name": row[2],
                        "score": row[3],
                        "tags": row[4],
                        "url": row[5] if row[5] else row[6],
                        "code_url": row[6],
                        "dataset_url": row[7],
                        "metadata_json": row[8],
                        "created_at": row[9],
                    }
                )

        return results

    except Exception as e:
        logger.error(f"Error searching artifacts: {e}")
        return []
    finally:
        conn.close()


def get_artifact_metadata(artifact_id: str) -> Optional[Dict]:
    """
    Get full metadata for an artifact.
    """
    artifact = get_artifact_by_id(artifact_id)
    if not artifact:
        return None

    try:
        metadata = json.loads(artifact.get("metadata_json", "{}"))
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata for {artifact_id}: {e}")
        return {}


def add_relationship(
    from_artifact_id: str, to_artifact_id: str, relationship_type: str
) -> bool:
    """
    Add a relationship between two artifacts.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO artifact_relationships
            (from_artifact_id, to_artifact_id, relationship_type)
            VALUES (?, ?, ?)
        """,
            (str(from_artifact_id), str(to_artifact_id), relationship_type),
        )

        conn.commit()
        logger.info(
            f"Added relationship: {from_artifact_id} --[{relationship_type}]--> "
            f"{to_artifact_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Error adding relationship: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_artifact_lineage(artifact_id: str) -> Dict:
    """
    Get the lineage graph for an artifact.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # Get the artifact itself
        artifact = get_artifact_by_id(artifact_id)
        if not artifact:
            return {"nodes": [], "edges": []}

        nodes = [
            {
                "artifact_id": int(artifact_id),
                "name": artifact["name"],
                "source": "config_json",
            }
        ]
        edges = []

        # Get relationships where this artifact is the target
        cur.execute(
            """
            SELECT from_artifact_id, relationship_type
            FROM artifact_relationships
            WHERE to_artifact_id = ?
        """,
            (str(artifact_id),),
        )

        relationships = cur.fetchall()

        for from_id, rel_type in relationships:
            from_artifact = get_artifact_by_id(from_id)
            if from_artifact:
                nodes.append(
                    {
                        "artifact_id": int(from_id),
                        "name": from_artifact["name"],
                        "source": (
                            "upstream_dataset"
                            if from_artifact["artifact_type"] == "dataset"
                            else "upstream_code"
                        ),
                    }
                )
                edges.append(
                    {
                        "from_node_artifact_id": int(from_id),
                        "to_node_artifact_id": int(artifact_id),
                        "relationship": rel_type,
                    }
                )

        # If no relationships found, try to infer from dataset_url
        if (
            not edges
            and artifact.get("dataset_url")
            and artifact["dataset_url"] != "unknown"
        ):
            dataset_name = artifact["dataset_url"].rstrip("/").split("/")[-1]
            dataset_id = generate_artifact_id(dataset_name)

            nodes.append(
                {
                    "artifact_id": int(dataset_id),
                    "name": dataset_name,
                    "source": "upstream_dataset",
                }
            )
            edges.append(
                {
                    "from_node_artifact_id": int(dataset_id),
                    "to_node_artifact_id": int(artifact_id),
                    "relationship": "fine_tuning_dataset",
                }
            )

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error(f"Error getting lineage for {artifact_id}: {e}")
        return {"nodes": [], "edges": []}
    finally:
        conn.close()


def list_models(limit=50, offset=0):
    """
    Returns only model artifacts in old format.
    """
    artifacts = list_artifacts(artifact_type="model", limit=limit, offset=offset)

    # Convert to old format
    return [
        {
            "name": a["name"],
            "score": a["score"],
            "tags": a["tags"],
            "code_url": a["code_url"],
            "dataset_url": a["dataset_url"],
            "created_at": a["created_at"],
            "metadata_json": a.get("metadata_json", "{}"),
        }
        for a in artifacts
    ]


def search_models(query):
    """
    Searches only model artifacts in old format.
    """
    artifacts = search_artifacts(query, artifact_type="model")

    # Convert to old format
    return [
        {
            "name": a["name"],
            "score": a["score"],
            "tags": a["tags"],
            "code_url": a["code_url"],
            "dataset_url": a["dataset_url"],
            "created_at": a["created_at"],
            "metadata_json": a.get("metadata_json", "{}"),
        }
        for a in artifacts
    ]


def add_model(
    name, score, tags="", code_url="unknown", dataset_url="unknown", metadata_json="{}"
):
    """
    Adds a model artifact.
    """
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid metadata JSON for {name}, using empty dict")
        metadata = {}

    return add_artifact(
        name=name,
        artifact_type="model",
        score=score,
        tags=tags,
        url=code_url,  # Use code_url as primary URL for models
        code_url=code_url,
        dataset_url=dataset_url,
        metadata=metadata,
    )


def reset_registry():
    """
    Completely clears the local registry database and any stored model files.
    Called by CLI or web API 'reset' command.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop all rows from both tables
    cur.execute("DELETE FROM artifact_relationships")
    cur.execute("DELETE FROM models")

    conn.commit()
    conn.close()

    import shutil

    downloads_dir = Path("downloaded_models")
    if downloads_dir.exists():
        shutil.rmtree(downloads_dir, ignore_errors=True)
        downloads_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Registry has been reset successfully.")


def get_registry_stats() -> Dict:
    """
    Get statistics about the registry.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT artifact_type, COUNT(*)
            FROM models
            GROUP BY artifact_type
        """
        )

        stats = {"total": 0}
        for artifact_type, count in cur.fetchall():
            type_name = artifact_type if artifact_type else "model"
            stats[type_name] = count
            stats["total"] += count

        return stats

    except Exception as e:
        logger.error(f"Error getting registry stats: {e}")
        return {"total": 0}
    finally:
        conn.close()
