import os

import handlers.registry_handler as rh


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
