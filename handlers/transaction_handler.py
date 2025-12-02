import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from API.dynamo import DynamoDB  # type: ignore
    DYNAMO_AVAILABLE = True
except Exception:
    DYNAMO_AVAILABLE = False
    DynamoDB = None  # type: ignore

try:
    from API.storage import S3Storage  # type: ignore
    S3_AVAILABLE = True
except Exception:
    S3_AVAILABLE = False
    S3Storage = None  # type: ignore

logger = logging.getLogger(__name__)


class _InMemoryTxnStore:
    def __init__(self):
        self.txns: Dict[str, Dict[str, Any]] = {}

    def init_transaction(self, owner: Optional[str] = None, ttl_seconds: int = 3600) -> str:
        import uuid, time

        txn_id = uuid.uuid4().hex
        now = "now"
        self.txns[txn_id] = {
            "txn_id": txn_id,
            "status": "collecting",
            "actions": [],
            "owner": owner or "unknown",
            "created_at": now,
            "updated_at": now,
            "ttl": int(time.time()) + int(ttl_seconds),
        }
        return txn_id

    def append_transaction_action(self, txn_id: str, action: Dict[str, Any]) -> bool:
        t = self.txns.get(txn_id)
        if not t:
            return False
        t["actions"].append(action)
        return True

    def transact_update_artifacts(self, updates: List[Dict[str, Any]]) -> bool:
        """Apply artifact updates in the in-memory registry (if available).

        This tries to update artifacts via `handlers.registry_handler.update_artifact` if present.
        Falls back to no-op (returns True) when registry isn't available.
        """
        try:
            from handlers import registry_handler

            for u in updates:
                aid = u.get("artifact_id")
                ups = u.get("updates", {})
                if not aid:
                    continue
                try:
                    registry_handler.update_artifact(aid, **ups)
                except Exception:
                    # If registry update fails, log and continue
                    logger.exception(f"In-memory transact: failed to apply update for {aid}")
            return True
        except Exception:
            # No registry available — nothing to update in tests
            logger.info("No registry handler available for in-memory transact_update_artifacts; skipping")
            return True

    def get_transaction(self, txn_id: str) -> Optional[Dict[str, Any]]:
        return self.txns.get(txn_id)

    def conditional_set_status(self, txn_id: str, from_status: str, to_status: str) -> bool:
        t = self.txns.get(txn_id)
        if not t or t.get("status") != from_status:
            return False
        t["status"] = to_status
        return True

    def mark_committed(self, txn_id: str) -> bool:
        t = self.txns.get(txn_id)
        if not t:
            return False
        t["status"] = "committed"
        return True

    def abort_transaction(self, txn_id: str, reason: Optional[str] = None) -> bool:
        t = self.txns.get(txn_id)
        if not t:
            return False
        t["status"] = "aborted"
        if reason:
            t["abort_reason"] = reason
        return True


_store: Any = None


def _get_store() -> Any:
    global _store
    if _store is None:
        if "PYTEST_CURRENT_TEST" in os.environ or not DYNAMO_AVAILABLE:
            logger.info("Using in-memory transaction store")
            _store = _InMemoryTxnStore()
        else:
            try:
                _store = DynamoDB()  # type: ignore
            except Exception:
                logger.exception("Failed to initialize DynamoDB for transactions; falling back to in-memory store")
                _store = _InMemoryTxnStore()
    return _store


def _get_storage() -> Optional[Any]:
    if not S3_AVAILABLE:
        return None
    try:
        return S3Storage()  # type: ignore
    except Exception:
        logger.exception("Failed to initialize S3Storage")
        return None


def init_transaction(owner: Optional[str] = None, ttl_seconds: int = 3600) -> str:
    store = _get_store()
    return store.init_transaction(owner=owner, ttl_seconds=ttl_seconds)


def append_action(txn_id: str, action: Dict[str, Any]) -> bool:
    """Append an action to txn. If action requires staging (upload), caller should have staged and supplied staged_key."""
    store = _get_store()
    return store.append_transaction_action(txn_id, action)


def get_transaction(txn_id: str) -> Optional[Dict[str, Any]]:
    store = _get_store()
    return store.get_transaction(txn_id)


def abort_transaction(txn_id: str, reason: Optional[str] = None) -> bool:
    # Cleanup staged objects if S3 available
    storage = _get_storage()
    if storage:
        try:
            storage.abort_stage(txn_id)
        except Exception:
            logger.exception("Failed to cleanup staged objects during abort")
    store = _get_store()
    return store.abort_transaction(txn_id, reason=reason)


def execute_transaction(txn_id: str) -> Dict[str, Any]:
    """Execute a transaction: commit staged uploads to final keys, then atomically update DB artifacts.

    Actions supported:
      - {"type": "upload", "artifact_id": "123", "staged_key": "_staging/...", "final_key": "models/...", "metadata": {...}}
      - {"type": "update", "artifact_id": "123", "updates": {...}}
      - {"type": "delete", "artifact_id": "123", "final_key": "..."}
    """
    store = _get_store()

    # acquire execution lock: collecting -> executing
    if not store.conditional_set_status(txn_id, "collecting", "executing"):
        return {"ok": False, "reason": "could_not_acquire_execution_lock"}

    txn = store.get_transaction(txn_id)
    if not txn:
        store.abort_transaction(txn_id, reason="not_found")
        return {"ok": False, "reason": "transaction_not_found"}

    storage = _get_storage()
    artifact_updates: List[Dict[str, Any]] = []

    # First, commit all staged uploads to S3
    try:
        for action in txn.get("actions", []):
            t = action.get("type")
            if t == "upload":
                staged = action.get("staged_key")
                final = action.get("final_key")
                if not (staged and final):
                    raise RuntimeError("upload action missing keys")
                if not storage:
                    raise RuntimeError("S3 not available for commit")
                storage.commit_stage(txn_id, staged, final)
                # Prepare DB update to set url/metadata
                upd: Dict[str, Any] = {"artifact_id": action.get("artifact_id"), "updates": {}}
                if action.get("metadata"):
                    # store metadata as metadata_json string to match schema
                    upd["updates"]["metadata_json"] = json.dumps(action.get("metadata"))
                if final:
                    upd["updates"]["url"] = f"s3://{storage.bucket_name}/{final}"
                artifact_updates.append(upd)
            elif t == "update":
                artifact_updates.append({"artifact_id": action.get("artifact_id"), "updates": action.get("updates", {})})
            elif t == "delete":
                # delete just maps to setting url to unknown and optionally removing S3 object
                final = action.get("final_key")
                if final and storage:
                    try:
                        storage.delete_file(final)
                    except Exception:
                        logger.exception("Failed to delete file during transaction delete action")
                artifact_updates.append({"artifact_id": action.get("artifact_id"), "updates": {"url": "unknown"}})

        # Now perform atomic DB updates
        if artifact_updates:
            ok = store.transact_update_artifacts(artifact_updates)
            if not ok:
                # Attempt rollback of committed S3 objects by deleting final keys
                for action in txn.get("actions", []):
                    if action.get("type") == "upload":
                        final = action.get("final_key")
                        try:
                            if storage and final:
                                storage.delete_file(final)
                        except Exception:
                            logger.exception("Rollback: failed to delete final object")
                store.abort_transaction(txn_id, reason="db_transact_failed")
                return {"ok": False, "reason": "db_transact_failed"}

        # Mark committed
        store.mark_committed(txn_id)
        return {"ok": True}

    except Exception as e:
        logger.exception(f"Execution failed for txn {txn_id}: {e}")
        # Attempt cleanup
        try:
            if storage:
                storage.abort_stage(txn_id)
        except Exception:
            logger.exception("Failed to cleanup staged after execution failure")
        store.abort_transaction(txn_id, reason=str(e))
        return {"ok": False, "reason": str(e)}
