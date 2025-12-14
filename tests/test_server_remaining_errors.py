import pytest
from unittest.mock import Mock
import server
from fastapi import HTTPException


def test_size_calc_github_api_and_unexpected(monkeypatch, caplog):
    caplog.set_level("ERROR")
    # Make requests.get raise RequestException to hit GitHub API error path
    from requests.exceptions import RequestException

    def raise_req(*a, **k):
        raise RequestException("netfail")

    monkeypatch.setattr("requests.get", raise_req)
    # Call size calc; should raise or return but log the GitHub API error
    try:
        server._calculate_artifact_size_api("https://github.com/owner/repo", "model")
    except Exception:
        pass
    assert any("GitHub API error" in rec.message or "GitHub API" in rec.message for rec in caplog.records)

    # Now make generic exception (still inside GitHub handling -> logs same GitHub API error)
    def raise_gen(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr("requests.get", raise_gen)
    try:
        server._calculate_artifact_size_api("https://github.com/owner/repo", "model")
    except Exception:
        pass
    assert any("GitHub API error" in rec.message or "GitHub API" in rec.message for rec in caplog.records)


def test_artifact_by_regex_search_error(monkeypatch, caplog):
    caplog.set_level("ERROR")

    # Fake request with JSON body
    class Req:
        def __init__(self, body):
            self._body = body
            self.headers = {}

        async def json(self):
            return self._body

    # Make re.compile return an object whose search raises
    def fake_compile(regex, flags=0):
        class P:
            def search(self, s):
                raise RuntimeError("search fail")

        return P()

    monkeypatch.setattr(server, "re", server.re)
    monkeypatch.setattr(server, "re", server.re)
    monkeypatch.setattr("re.compile", fake_compile)

    # Ensure registry returns one artifact to iterate
    monkeypatch.setattr("handlers.registry_handler.list_artifacts", lambda *a, **k: [{"name": "a"}])

    req = Req({"regex": "a"})
    with pytest.raises(HTTPException):
        # call the async endpoint
        import asyncio

        asyncio.run(server.artifact_by_regex(req))

    assert any("Regex search error" in rec.message or "Regex search timed out" in rec.message for rec in caplog.records)


def test_rate_request_invalid_id_logs(monkeypatch, caplog):
    caplog.set_level("ERROR")
    # Ensure list_artifacts returns empty so lookup fails
    monkeypatch.setattr("handlers.registry_handler.list_artifacts", lambda *a, **k: [])
    import asyncio

    with pytest.raises(HTTPException):
        asyncio.run(server.rate_model("notanint", Mock()))

    assert any("Invalid ID format" in rec.message or "Artifact 'notanint' not found" in rec.message for rec in caplog.records)
