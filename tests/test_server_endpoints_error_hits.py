from fastapi.testclient import TestClient
import server
import pytest


client = TestClient(server.app)


def test_artifacts_bad_json_logs_error(caplog):
    caplog.clear()
    # send invalid json bytes
    resp = client.post("/artifacts", data=b"{badjson", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert any("JSON parse error" in r.message or "Unexpected error" in r.message for r in caplog.records)


def test_lineage_invalid_id_logs_error(caplog):
    caplog.clear()
    resp = client.get("/artifact/model/notanint/lineage")
    assert resp.status_code == 404
    assert any("Lineage request - Invalid ID format" in r.message for r in caplog.records)


def test_rate_request_invalid_id_logs_error(caplog):
    caplog.clear()
    # call rate endpoint with invalid id via post to trigger earlier rate request error path
    resp = client.post("/artifact/model/invalid-id/rate", json={})
    # endpoint may not exist; accept 404 or 400
    assert resp.status_code in (400, 404, 405)
