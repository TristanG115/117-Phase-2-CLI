from metrics.size_score_metric import SizeScoreMetric
from metrics.dataset_quality_metric import DatasetQualityMetric
from unittest.mock import Mock, patch
from url_classifier import URLType
import os


def test_size_score_thresholds():
    m = SizeScoreMetric()
    # create sizes in MB for different edge cases
    tiny = Mock(); tiny.get_size_mb.return_value = 400  # 0.39 GB
    small = Mock(); small.get_size_mb.return_value = 1024  # 1 GB
    medium = Mock(); medium.get_size_mb.return_value = 4096  # 4 GB
    large = Mock(); large.get_size_mb.return_value = 20000  # ~19.5 GB

    res_t, _ = m.calculate({URLType.MODEL: [tiny]})
    assert res_t["raspberry_pi"] == 1.0

    res_s, _ = m.calculate({URLType.MODEL: [small]})
    assert res_s["raspberry_pi"] < 1.0

    res_l, _ = m.calculate({URLType.MODEL: [large]})
    assert res_l["aws_server"] >= 0.85


def test_size_score_private_methods():
    m = SizeScoreMetric()
    # Raspberry Pi thresholds
    assert m._raspberry_pi_score(0.2) == 1.0
    assert m._raspberry_pi_score(0.75) == 0.9
    assert m._raspberry_pi_score(1.5) == 0.7
    assert m._raspberry_pi_score(2.5) == 0.4

    # Jetson
    assert m._jetson_nano_score(0.5) == 1.0
    assert m._jetson_nano_score(3.5) == 0.6

    # Desktop
    assert m._desktop_pc_score(1.0) == 1.0
    assert m._desktop_pc_score(10.0) == 0.8

    # AWS
    assert m._aws_server_score(5.0) == 1.0
    assert m._aws_server_score(30.0) == 0.95


def test_dataset_metric_llm_error_and_non200(monkeypatch):
    m = DatasetQualityMetric()
    dataset = Mock(); dataset.url = "http://x"
    dataset.get_huggingface_api_data.return_value = {}

    os.environ["GEN_AI_STUDIO_API_KEY"] = "key"

    # non-200 response
    r1 = Mock(); r1.status_code = 500
    with patch("requests.post", return_value=r1):
        score, _ = m.calculate({URLType.DATASET: [dataset]})
        # should fall back and return heuristic
        assert score >= 0.5

    # exception in requests.post
    def bad(*a, **k):
        raise Exception("boom")

    with patch("requests.post", side_effect=bad):
        score2, _ = m.calculate({URLType.DATASET: [dataset]})
        assert score2 >= 0.5

    del os.environ["GEN_AI_STUDIO_API_KEY"]


def test_evaluate_with_llm_non_numeric_and_missing_choices(monkeypatch):
    m = DatasetQualityMetric()
    dataset = Mock(); dataset.url = "http://x"
    dataset.get_huggingface_api_data.return_value = {}
    os.environ["GEN_AI_STUDIO_API_KEY"] = "key"

    # non-numeric content
    r1 = Mock(); r1.status_code = 200; r1.json.return_value = {"choices": [{"message": {"content": "not-a-number"}}]}
    with patch("requests.post", return_value=r1):
        assert m._evaluate_with_llm(dataset, "key") == 0.0

    # missing choices key
    r2 = Mock(); r2.status_code = 200; r2.json.return_value = {}
    with patch("requests.post", return_value=r2):
        assert m._evaluate_with_llm(dataset, "key") == 0.0

    del os.environ["GEN_AI_STUDIO_API_KEY"]
