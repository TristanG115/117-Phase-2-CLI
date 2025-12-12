import requests

from lineage import LineageExtractor


def test_genai_timeout_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    le = LineageExtractor()
    # set an API key so _call_genai_api attempts the request
    le.api_key = "x"

    def bad_post(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr("lineage.requests.post", bad_post)

    res = le._call_genai_api("p", "content")
    assert res is None
    assert any("GenAI API call timed out" in r.message for r in caplog.records if r.levelname == "ERROR")
