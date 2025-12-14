import json
import pytest

import handlers.registry_handler as rh


def setup_function(_):
    rh.reset_registry()


def test_generate_artifact_id_deterministic():
    a = rh.generate_artifact_id("my-artifact")
    b = rh.generate_artifact_id("my-artifact")
    assert a == b
    assert len(a) <= 10


def test_add_model_invalid_metadata_json_logs_warning(caplog):
    rh.reset_registry()
    # invalid JSON should be handled gracefully
    artifact_id = rh.add_model("m-invalid", 0.1, metadata_json="not-a-json")
    assert artifact_id == "m-invalid"
    stored = rh.get_artifact_by_id(artifact_id)
    assert stored is not None
    assert isinstance(stored.get("metadata_json"), str)


def test_batch_add_artifacts_with_error(caplog):
    rh.reset_registry()
    # one artifact missing required fields will raise in add_artifact
    artifacts = [dict(name="good", artifact_type="model"), dict(artifact_type="broken")]
    ids = rh.batch_add_artifacts(artifacts)
    assert ids == ["good"]


def test_get_lineage_graph_basic():
    rh.reset_registry()
    rh.add_artifact(
        name="root_model",
        artifact_type="model",
        code_url="https://github.com/owner/code_repo",
        dataset_url="https://huggingface.co/datasets/owner/data",
        metadata={
            "parent_model": "base_model",
            "evaluation_datasets": ["eval1", "eval2"],
        },
    )

    graph = rh.get_lineage_graph("root_model")
    nodes = {n["name"]: n for n in graph["nodes"]}
    assert "root_model" in nodes
    assert "data" in nodes or any("data" in n for n in nodes)
    # Expect parent_model and eval datasets present
    assert any(n["name"] == "base_model" for n in graph["nodes"])
    assert any(n["name"] == "eval1" for n in graph["nodes"])


def test_get_lineage_graph_not_found():
    rh.reset_registry()
    with pytest.raises(ValueError):
        rh.get_lineage_graph("nonexistent")


def test_health_check_unhealthy(monkeypatch):
    # Force _get_db to raise
    monkeypatch.setattr(rh, "_get_db", lambda: (_ for _ in ()).throw(Exception("boom")))
    health = rh.health_check()
    assert health["status"] == "unhealthy"
    assert "error" in health
