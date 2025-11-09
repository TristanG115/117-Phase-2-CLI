import hashlib
import json
import logging
from typing import Dict, List, Optional

from API.dynamo import DynamoDB

# Configure logging
logger = logging.getLogger(__name__)

# Initialize DynamoDB instance
_db: Optional[DynamoDB] = None


def _get_db() -> DynamoDB:
    """Get or create DynamoDB instance (singleton pattern)"""
    global _db
    if _db is None:
        _db = DynamoDB()
    return _db


def generate_artifact_id(name: str) -> str:
    """
    Generate deterministic 10-digit artifact ID from name.
    """
    return str(abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10))


def init_registry():
    """
    Initialize the DynamoDB registry.
    Creates table if it doesn't exist.
    """
    try:
        db = _get_db()
        db.init_table()
        logger.info("Registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize registry: {e}")
        raise


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
    Returns the artifact_id.
    """
    db = _get_db()
    return db.add_artifact(
        name=name,
        artifact_type=artifact_type,
        score=score,
        url=url,
        tags=tags,
        code_url=code_url,
        dataset_url=dataset_url,
        metadata=metadata,
    )


def get_artifact_by_id(
    artifact_id: str, artifact_type: Optional[str] = None
) -> Optional[Dict]:
    """
    Get an artifact by ID.
    If artifact_type provided, validates the type matches.
    """
    db = _get_db()
    return db.get_artifact_by_id(artifact_id, artifact_type)


def artifact_exists(artifact_id: str) -> bool:
    """
    Check if an artifact exists in the registry.
    """
    db = _get_db()
    return db.artifact_exists(artifact_id)


def update_artifact(artifact_id: str, **updates) -> bool:
    """
    Update an artifact's fields.
    Returns True if successful.
    """
    db = _get_db()
    return db.update_artifact(artifact_id, **updates)


def delete_artifact(artifact_id: str) -> bool:
    """
    Delete an artifact from the registry.
    Returns True if successful.
    """
    db = _get_db()
    return db.delete_artifact(artifact_id)


def list_artifacts(
    artifact_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict]:
    """
    List artifacts, optionally filtered by type.
    Supports pagination.
    """
    db = _get_db()
    return db.list_artifacts(artifact_type, limit, offset)


def search_artifacts(query: str, artifact_type: Optional[str] = None) -> List[Dict]:
    """
    Search artifacts by name or tags.
    """
    db = _get_db()
    return db.search_artifacts(query, artifact_type)


def get_registry_stats() -> Dict:
    """
    Get statistics about the registry.
    Returns counts by artifact type.
    """
    db = _get_db()
    return db.get_registry_stats()


def reset_registry():
    """
    Completely clears the registry.
    WARNING: This deletes ALL data!
    """
    db = _get_db()
    db.reset_registry()
    logger.info("Registry has been reset successfully.")


def list_models(limit: int = 100, offset: int = 0) -> List[Dict]:
    """
    List only model artifacts.
    Returns models in old format for backward compatibility.
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


def search_models(query: str) -> List[Dict]:
    """
    Legacy function: Search only model artifacts.
    Returns models in old format.
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
    name: str,
    score: float,
    tags: str = "",
    code_url: str = "unknown",
    dataset_url: str = "unknown",
    metadata_json: str = "{}",
) -> str:
    """
    Add a model artifact.
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


def get_artifact_by_name(
    name: str, artifact_type: Optional[str] = None
) -> Optional[Dict]:
    """
    Get an artifact by name instead of ID.
    Useful for looking up artifacts when you only have the name.
    """
    artifact_id = generate_artifact_id(name)
    return get_artifact_by_id(artifact_id, artifact_type)


def batch_add_artifacts(artifacts: List[Dict]) -> List[str]:
    """
    Add multiple artifacts at once.
    Returns list of artifact_ids.
    """
    artifact_ids = []
    for artifact in artifacts:
        try:
            artifact_id = add_artifact(**artifact)
            artifact_ids.append(artifact_id)
        except Exception as e:
            logger.error(f"Failed to add artifact {artifact.get('name')}: {e}")

    return artifact_ids


def get_artifacts_by_type_count() -> Dict[str, int]:
    """
    Get count of artifacts by type.
    Useful for dashboard statistics.
    """
    stats = get_registry_stats()
    return {k: v for k, v in stats.items() if k != "total"}


def health_check() -> Dict[str, str]:
    """
    Check if DynamoDB connection is healthy.
    """
    try:
        db = _get_db()
        stats = db.get_registry_stats()
        return {
            "status": "healthy",
            "backend": "DynamoDB",
            "total_artifacts": str(stats.get("total", 0)),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "backend": "DynamoDB",
            "error": str(e),
        }
