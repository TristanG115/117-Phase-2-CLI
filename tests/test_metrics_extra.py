import os
from unittest.mock import Mock, patch

from metrics.dataset_quality_metric import DatasetQualityMetric
from metrics.size_score_metric import SizeScoreMetric
from url_classifier import URLType


def test_dataset_metric_no_resources():
    m = DatasetQualityMetric()
    score, latency = m.calculate({})
    assert score == 0.0


def test_evaluate_with_llm_success(monkeypatch):
    m = DatasetQualityMetric()
    dataset = Mock()
    dataset.url = "http://example"
    dataset.get_huggingface_api_data.return_value = {"description": "x", "tags": [], "downloads": 10}

    os.environ["GEN_AI_STUDIO_API_KEY"] = "key"

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "0.7"}}]}

    with patch("requests.post", return_value=mock_resp):
        score, _ = m.calculate({URLType.DATASET: [dataset]})
        assert 0.69 < score < 0.71

    del os.environ["GEN_AI_STUDIO_API_KEY"]


def test_evaluate_dataset_quality_fallback_with_get_quality_score():
    m = DatasetQualityMetric()
    dataset = Mock()
    dataset.get_quality_score.return_value = 0.4
    score, _ = m.calculate({URLType.DATASET: [dataset]})
    assert score > 0.5


def test_size_score_metric_various():
    m = SizeScoreMetric()

    # No model resources
    scores, _ = m.calculate({})
    assert scores == {}

    # Test with a tiny model
    tiny = Mock(); tiny.get_size_mb.return_value = 100  # ~0.1GB
    res, _ = m.calculate({URLType.MODEL: [tiny]})
    assert res["raspberry_pi"] == 1.0

    # Test with large model raising error
    bad = Mock(); bad.get_size_mb.side_effect = Exception("oops")
    res2 = m._calculate_hardware_compatibility(bad)
    assert isinstance(res2, dict)
