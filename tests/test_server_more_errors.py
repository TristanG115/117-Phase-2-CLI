from fastapi.testclient import TestClient
import server


client = TestClient(server.app)


def test_license_check_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def bad_get(*a, **k):
        raise server.requests.RequestException("boom")

    # Ensure artifact exists check passes
    monkeypatch.setattr(server, "_verify_artifact_exists", lambda *_a, **_k: True)

    monkeypatch.setattr(server, "requests", server.requests)
    monkeypatch.setattr(server.requests, "get", bad_get)

    r = client.post("/artifact/model/1/license-check", json={"github_url": "https://github.com/owner/repo"})
    assert r.status_code == 502
    assert any("Error fetching GitHub page" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_calculate_size_unexpected_error(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def bad_get(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr(server, "requests", server.requests)
    monkeypatch.setattr(server.requests, "get", bad_get)

    # Use a GitHub URL to exercise the GitHub branch which will now raise
    res = server._calculate_artifact_size_api("https://github.com/owner/repo", "model")
    assert res == 0.0
    assert any("GitHub API error" in r.message or "Unexpected error for" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_log_audit_event_failure(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "get_artifact_by_id", lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")))
    server._log_audit_event(1, "x", "model", "AUDIT")
    assert any("Failed to log audit event" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_lineage_artifact_not_found_and_invalid(monkeypatch, caplog):
    caplog.set_level("ERROR")

    # Not found (numeric but not present)
    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])
    r = client.get("/artifact/model/123/lineage")
    assert r.status_code == 404
    assert any("Lineage request - Artifact not found" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    # Invalid format
    r = client.get("/artifact/model/notanumber/lineage")
    assert r.status_code == 404
    assert any("Lineage request - Invalid ID format" in r.message for r in caplog.records if r.levelname == "ERROR")
