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
