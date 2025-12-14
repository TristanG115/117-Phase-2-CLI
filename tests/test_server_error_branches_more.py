import asyncio
import pytest

import server
from handlers import registry_handler


def test_startup_registry_init_failure(monkeypatch):
    def fail():
        raise RuntimeError("init fail")

    monkeypatch.setattr(registry_handler, "init_registry", fail)

    with pytest.raises(RuntimeError):
        # call the startup event directly
        asyncio.run(server.startup_event())


def test_rate_model_group_urls_error(caplog, monkeypatch):
    caplog.set_level("ERROR")

    # make url_classifier.group_urls_by_type raise
    class Bad:
        def group_urls_by_type(self, urls):
            raise Exception("boom")

    monkeypatch.setattr(server.model_evaluator, "url_classifier", Bad())

    # run the async background rating (just execute the path)
    asyncio.run(server.rate_model_background(12345, "name", "https://huggingface.co/model"))


def test_rate_model_evaluate_urls_error(caplog, monkeypatch):
    caplog.set_level("ERROR")

    def bad_eval(urls, artifact_id=None):
        raise Exception("eval boom")

    monkeypatch.setattr(server.model_evaluator, "evaluate_urls", bad_eval)

    asyncio.run(server.rate_model_background(123, "name", "https://huggingface.co/model"))
