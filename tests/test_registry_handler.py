
# ==================================================
# BEGIN test_registry_handler.py
# ==================================================

# tests/test_registry_handler.py

import handlers.registry_handler as rh


def setup_function(_):
    """
    Ensures we always start with a clean in-memory registry.
    """
    rh.reset_registry()


def test_add_and_get_artifact():
    rh.reset_registry()

    artifact_id = rh.add_artifact(
        name="test_model",
        artifact_type="model",
        score=0.9,
        url="https://example.com",
        tags="nlp,test",
        code_url="https://github.com/test/repo",
        dataset_url="https://huggingface.co/datasets/test",
        metadata={"version": 1},
    )

    assert artifact_id == "test_model"

    stored = rh.get_artifact_by_id(artifact_id)
    assert stored is not None
    assert stored["name"] == "test_model"
    assert stored["artifact_type"] == "model"


def test_search_artifacts():
    rh.reset_registry()

    rh.add_artifact("alpha", "model", tags="embedding")
    rh.add_artifact("beta", "dataset", tags="vision")
    rh.add_artifact("alphabet", "model", tags="nlp")

    results = rh.search_artifacts("alp")
    names = sorted([r["name"] for r in results])

    assert names == ["alpha", "alphabet"]


def test_list_artifacts_and_filtering():
    rh.reset_registry()

    rh.add_artifact("m1", "model")
    rh.add_artifact("m2", "model")
    rh.add_artifact("d1", "dataset")

    all_items = rh.list_artifacts()
    models = rh.list_artifacts(artifact_type="model")

    assert len(all_items) == 3
    assert len(models) == 2
    assert {m["name"] for m in models} == {"m1", "m2"}


def test_update_artifact():
    rh.reset_registry()

    rh.add_artifact("u1", "model", score=1.0)
    assert rh.update_artifact("u1", score=2.0)

    updated = rh.get_artifact_by_id("u1")
    assert updated["score"] == 2.0


def test_delete_artifact():
    rh.reset_registry()

    rh.add_artifact("delme", "model")
    assert rh.delete_artifact("delme")
    assert rh.get_artifact_by_id("delme") is None


def test_registry_stats():
    rh.reset_registry()

    rh.add_artifact("m1", "model")
    rh.add_artifact("d1", "dataset")

    stats = rh.get_registry_stats()

    assert stats["model"] == 1
    assert stats["dataset"] == 1
    assert stats["total"] == 2


def test_list_models_legacy_format():
    rh.reset_registry()

    rh.add_model(
        name="legacy_model",
        score=0.5,
        tags="legacy",
        code_url="http://github.com/legacy",
        dataset_url="http://hf.co/dataset/legacy",
        metadata_json="{}",
    )

    models = rh.list_models()
    assert len(models) == 1
    m = models[0]

    assert m["name"] == "legacy_model"
    assert "created_at" in m
    assert "metadata_json" in m


def test_search_models_legacy():
    rh.reset_registry()

    rh.add_model("m1", 0.1, tags="nlp")
    rh.add_model("m2", 0.2, tags="vision")

    results = rh.search_models("nlp")
    assert len(results) == 1
    assert results[0]["name"] == "m1"


def test_batch_add_artifacts():
    rh.reset_registry()

    arts = [
        dict(name="a1", artifact_type="model"),
        dict(name="a2", artifact_type="dataset"),
    ]

    ids = rh.batch_add_artifacts(arts)

    assert ids == ["a1", "a2"]
    assert rh.get_artifact_by_id("a1") is not None
    assert rh.get_artifact_by_id("a2") is not None


def test_get_artifacts_by_type_count():
    rh.reset_registry()

    rh.add_artifact("m1", "model")
    rh.add_artifact("m2", "model")
    rh.add_artifact("d1", "dataset")

    type_counts = rh.get_artifacts_by_type_count()

    assert type_counts["model"] == 2
    assert type_counts["dataset"] == 1


def test_health_check():
    rh.reset_registry()
    health = rh.health_check()

    assert health["status"] == "healthy"
    assert "backend" in health
    assert "persistent" in health

# ==================================================
# BEGIN test_registry_handler_more.py
# ==================================================

import os



def test_generate_and_basic_crud(monkeypatch):
    # Ensure in-memory DB is used
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    rh.reset_registry()

    aid = rh.add_artifact(name="m1", artifact_type="model", score=0.5)
    assert aid == "m1"
    assert rh.artifact_exists("m1") is True

    obj = rh.get_artifact_by_id("m1")
    assert obj.get("name") == "m1"

    assert rh.update_artifact("m1", score=0.9) is True
    assert rh.get_artifact_by_id("m1").get("score") == 0.9

    assert rh.delete_artifact("m1") is True
    assert rh.artifact_exists("m1") is False


def test_list_search_and_stats():
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    rh.reset_registry()
    rh.add_artifact(name="a1", artifact_type="model", tags="t1")
    rh.add_artifact(name="b1", artifact_type="dataset", tags="t2")

    lst = rh.list_artifacts()
    assert len(lst) >= 2

    models = rh.list_models()
    assert isinstance(models, list)

    res = rh.search_artifacts("a1")
    assert any("a1" in r["name"] for r in res)

    stats = rh.get_registry_stats()
    assert stats.get("total") >= 2

    counts = rh.get_artifacts_by_type_count()
    assert "model" in counts


def test_add_model_and_get_by_name_and_batch_and_health():
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    rh.reset_registry()

    mid = rh.add_model(name="m2", score=0.7, tags="x")
    assert mid == "m2"

    assert rh.get_artifact_by_name("m2") is not None

    ids = rh.batch_add_artifacts([{"name": "c1", "artifact_type": "code"}, {"name": "c2", "artifact_type": "code"}])
    assert len(ids) == 2

    h = rh.health_check()
    assert h.get("status") == "healthy"


def test_get_lineage_graph():
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    rh.reset_registry()

    # Add a model with dataset and code and parent and evaluation datasets
    rh.add_artifact(name="mroot", artifact_type="model", metadata={"parent_model": "pm", "evaluation_datasets": ["ed1"]}, url="https://huggingface.co/mroot", code_url="https://github.com/owner/repo", dataset_url="https://huggingface.co/datasets/owner/ds")

    graph = rh.get_lineage_graph("mroot")
    assert "nodes" in graph and "edges" in graph

    # Not found should raise
    try:
        rh.get_lineage_graph("nope")
    except ValueError:
        pass


def test_get_db_dynamo_file_not_found(monkeypatch, capsys):
    # Simulate Dynamo available but constructor raises FileNotFoundError
    monkeypatch.setattr(rh, "DYNAMO_AVAILABLE", True)
    class FakeDynamo:
        def __init__(self):
            raise FileNotFoundError("missing config")

    monkeypatch.setattr(rh, "DynamoDB", FakeDynamo)
    monkeypatch.setattr(rh, "_db", None)

    db = rh._get_db()
    assert isinstance(db, rh._InMemoryDB)


def test_get_db_dynamo_exception(monkeypatch):
    monkeypatch.setattr(rh, "DYNAMO_AVAILABLE", True)

    class FakeDynamo2:
        def __init__(self):
            raise Exception("boom")

    monkeypatch.setattr(rh, "DynamoDB", FakeDynamo2)
    monkeypatch.setattr(rh, "_db", None)

    db = rh._get_db()
    assert isinstance(db, rh._InMemoryDB)


def test_get_db_when_dynamo_unavailable(monkeypatch, capsys):
    # Remove pytest marker to exercise alternate branch
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(rh, "DYNAMO_AVAILABLE", False)
    monkeypatch.setattr(rh, "_db", None)

    db = rh._get_db()
    assert isinstance(db, rh._InMemoryDB)
    captured = capsys.readouterr()
    assert "falling back to in-memory database" in captured.out or "WARNING" in captured.out

def test_init_registry_prints_warning(monkeypatch, capsys):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(rh, "DYNAMO_AVAILABLE", False)
    monkeypatch.setattr(rh, "_db", None)

    rh.init_registry()
    out = capsys.readouterr().out
    assert "REGISTRY USING IN-MEMORY DATABASE" in out or "WARNING" in out


def test_get_artifact_by_name_slow_path(monkeypatch):
    rh.reset_registry()
    rh.add_artifact("x", "model")

    db = rh._get_db()

    # Force direct lookup to return None, but list_artifacts still finds the item
    monkeypatch.setattr(db, "get_artifact_by_id", lambda _id, _t=None: None)
    res = rh.get_artifact_by_name("x")
    assert res is not None

# ==================================================
# BEGIN test_registry_handler_extra.py
# ==================================================

import json
import pytest



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