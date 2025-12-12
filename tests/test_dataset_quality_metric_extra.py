import os

import pytest

from metrics.dataset_quality_metric import DatasetQualityMetric
from url_classifier import URLType


class FakeDataset:
    def __init__(self, url="https://huggingface.co/ds", api_data=None, quality=None):
        self.url = url
        self._api_data = api_data or {}
        self._quality = quality

    def get_huggingface_api_data(self):
        return self._api_data

    def get_quality_score(self):
        return self._quality


def test_no_dataset_returns_zero():
    m = DatasetQualityMetric()
    score, latency = m.calculate({},)
    assert score == 0.0


def test_llm_success(monkeypatch):
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "key")

    def fake_post(url, headers=None, json=None, timeout=None):
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "0.85"}}]}

        return R()

    monkeypatch.setattr("requests.post", fake_post)

    ds = FakeDataset()
    m = DatasetQualityMetric()
    score, latency = m.calculate({URLType.DATASET: [ds]})
    assert pytest.approx(score, rel=1e-3) == 0.85
    assert latency >= 0


def test_llm_non_numeric_fallback(monkeypatch):
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "key")

    def fake_post(url, headers=None, json=None, timeout=None):
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "not-a-number"}}]}

        return R()

    monkeypatch.setattr("requests.post", fake_post)

    ds = FakeDataset(api_data={})
    m = DatasetQualityMetric()
    score, _ = m.calculate({URLType.DATASET: [ds]})
    assert score >= 0.5


def test_llm_status_non_200(monkeypatch):
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "key")

    def fake_post(url, headers=None, json=None, timeout=None):
        class R:
            status_code = 500

            def json(self):
                return {}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    ds = FakeDataset()
    m = DatasetQualityMetric()
    score, _ = m.calculate({URLType.DATASET: [ds]})
    assert score >= 0.5


def test_llm_exception(monkeypatch):
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "key")

    def fake_post(url, headers=None, json=None, timeout=None):
        raise Exception("boom")

    monkeypatch.setattr("requests.post", fake_post)
    ds = FakeDataset()
    m = DatasetQualityMetric()
    score, _ = m.calculate({URLType.DATASET: [ds]})
    assert score >= 0.5


def test_heuristic_get_quality_score(monkeypatch):
    # dataset provides explicit get_quality_score
    ds = FakeDataset(quality=0.3)
    m = DatasetQualityMetric()
    score, _ = m.calculate({URLType.DATASET: [ds]})
    assert pytest.approx(score, rel=1e-3) == 0.5 + (0.3 * 0.5)


def test_heuristic_api_data_scoring():
    api_data = {
        "cardData": True,
        "downloads": 50,
        "tags": ["a", "b"],
        "siblings": ["f1"],
    }
    class BareDataset:
        def __init__(self, api_data):
            self.url = "https://huggingface.co/ds"
            self._api_data = api_data

        def get_huggingface_api_data(self):
            return self._api_data

    ds = BareDataset(api_data=api_data)
    m = DatasetQualityMetric()
    score, _ = m.calculate({URLType.DATASET: [ds]})
    # baseline 0.5 + doc 0.2 + downloads 0.1 + tags 0.1 + siblings 0.05 = 0.95
    assert pytest.approx(score, rel=1e-3) == 0.95
