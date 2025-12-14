import pytest
from fastapi.testclient import TestClient

import server


client = TestClient(server.app)


def test_rate_model_error_listing(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def boom():
        raise Exception("boom")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", boom)

    r = client.get("/artifact/model/1/rate")
    assert r.status_code == 500
    assert any("Error listing artifacts: boom" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_rate_model_invalid_id_format(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/notanumber/rate")
    assert r.status_code == 404
    assert any("Invalid ID format" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_rate_model_not_found_logs_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")

    # Return empty list so numeric lookup fails
    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/9999/rate")
    assert r.status_code == 404
    # Should log artifact not found messages
    assert any("Artifact '" in r.message for r in caplog.records if r.levelname == "ERROR")
