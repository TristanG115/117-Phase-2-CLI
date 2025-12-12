import asyncio
from unittest.mock import Mock

import pytest
import requests

import server
from fastapi import HTTPException


def test_startup_registry_init_failure(monkeypatch, caplog):
    # init_registry raises -> startup_event should log and re-raise
    monkeypatch.setattr("handlers.registry_handler.init_registry", lambda: (_ for _ in ()).throw(Exception("init fail")))
    caplog.clear()
    with pytest.raises(Exception):
        asyncio.run(server.startup_event())
    assert any("Failed to initialize registry" in rec.message for rec in caplog.records)


def test_rate_model_group_urls_error(monkeypatch, caplog):
    # group_urls_by_type raises -> should be caught and logged
    monkeypatch.setattr(server.model_evaluator.url_classifier, "group_urls_by_type", lambda *a, **k: (_ for _ in ()).throw(Exception("group fail")))
    monkeypatch.setattr(server.model_evaluator, "evaluate_urls", lambda *a, **k: [])
    monkeypatch.setattr("handlers.registry_handler.get_artifact_by_id", lambda *a, **k: None)
    caplog.clear()
    asyncio.run(server.rate_model_background(1, "n", "u"))
    assert any("URL classification error" in rec.message or "URL classification" in rec.message for rec in caplog.records)


def test_rate_model_evaluate_raises(monkeypatch, caplog):
    # evaluate_urls raises -> outer except should log Error rating model
    monkeypatch.setattr(server.model_evaluator.url_classifier, "group_urls_by_type", lambda *a, **k: {})
    monkeypatch.setattr(server.model_evaluator, "evaluate_urls", lambda *a, **k: (_ for _ in ()).throw(Exception("eval fail")))
    monkeypatch.setattr("handlers.registry_handler.get_artifact_by_id", lambda *a, **k: None)
    caplog.clear()
    asyncio.run(server.rate_model_background(2, "nm", "u"))
    assert any("Error rating model" in rec.message for rec in caplog.records)


def test_check_license_fetch_and_general_errors(monkeypatch, caplog):
    # RequestException path
    from requests.exceptions import RequestException

    def raise_req(*a, **k):
        raise RequestException("net")

    monkeypatch.setattr("requests.get", raise_req)
    caplog.clear()
    with pytest.raises(HTTPException):
        server._check_license_compatibility("https://github.com/owner/repo")
    assert any("Error fetching GitHub page" in rec.message for rec in caplog.records)

    # General exception path
    def raise_val(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr("requests.get", raise_val)
    caplog.clear()
    with pytest.raises(HTTPException):
        server._check_license_compatibility("https://github.com/owner/repo")
    assert any("Error checking license" in rec.message for rec in caplog.records)


def test_lineage_invalid_id_and_json_decode(monkeypatch, caplog):
    # Invalid ID format
    req = Mock(); req.headers = {}
    caplog.clear()
    with pytest.raises(HTTPException):
        server.get_artifact_lineage("notanint", req)
    assert any("Lineage request - Invalid ID format" in rec.message for rec in caplog.records)

    # JSON decode error path deeper in lineage
    # Prepare a matching artifact whose metadata_json is malformed
    bad = {"name": "x", "metadata_json": "{bad json"}
    monkeypatch.setattr("handlers.registry_handler.list_artifacts", lambda *a, **k: [bad])
    req2 = Mock(); req2.headers = {}
    caplog.clear()
    with pytest.raises(HTTPException):
        server.get_artifact_lineage(str(server.gen_id("x")), req2)
    assert any("JSON decode error" in rec.message or "Unexpected error for artifact" in rec.message for rec in caplog.records)


def test_log_audit_event_failure(monkeypatch, caplog):
    # Cause json.loads to raise by returning malformed metadata_json
    bad = {"name": "a", "metadata_json": "{bad"}
    monkeypatch.setattr("handlers.registry_handler.get_artifact_by_id", lambda *a, **k: bad)
    caplog.clear()
    server._log_audit_event(1, "a", "model", "CREATE")
    assert any("Failed to log audit event" in rec.message for rec in caplog.records)
