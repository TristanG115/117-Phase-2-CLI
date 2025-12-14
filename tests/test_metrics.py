import unittest
import os
from unittest.mock import Mock
import pytest

from metrics import CodeQualityMetric, DatasetQualityMetric, LicenseMetric, SizeScoreMetric
from resource_handlers import BaseResourceHandler
from url_classifier import URLType
from metrics.size_score_metric import SizeScoreMetric
from metrics.dataset_quality_metric import DatasetQualityMetric
from unittest.mock import Mock, patch
from metrics.dataset_and_code_score_metric import DatasetAndCodeScoreMetric
from metrics.bus_factor_metric import BusFactorMetric

class TestMetrics(unittest.TestCase):
    """Test metric calculation functionality"""

    def setUp(self):
        self.mock_model = Mock()
        self.mock_dataset = Mock()
        self.mock_code = Mock()

        self.resources = {
            URLType.MODEL: [self.mock_model],
            URLType.DATASET: [self.mock_dataset],
            URLType.CODE: [self.mock_code],
        }

    def test_license_metric_required_types(self):
        """Test 11: LicenseMetric required URL types"""
        metric = LicenseMetric()
        required = metric.required_url_types()
        expected = [URLType.MODEL]
        self.assertEqual(required, expected)

    def test_size_score_metric_required_types(self):
        """Test 12: SizeScoreMetric required URL types"""
        metric = SizeScoreMetric()
        required = metric.required_url_types()
        expected = [URLType.MODEL]
        self.assertEqual(required, expected)

    def test_license_metric_calculation(self):
        """Test 13: LicenseMetric calculation"""
        self.mock_model.get_license_score.return_value = 0.8
        self.mock_dataset.get_license_score.return_value = 0.9
        self.mock_code.get_license_score.return_value = 0.7

        metric = LicenseMetric()
        score, latency = metric.calculate(self.resources)

        self.assertEqual(score, 0.8)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_size_score_metric_calculation(self):
        """Test 14: SizeScoreMetric calculation"""
        self.mock_model.get_size_mb.return_value = 500
        metric = SizeScoreMetric()
        score, latency = metric.calculate(self.resources)
        self.assertIsInstance(score, dict)
        self.assertIn("raspberry_pi", score)
        self.assertIn("desktop_pc", score)
        self.assertIsInstance(latency, int)

    def test_dataset_quality_metric_calculation(self):
        """Test 15: DatasetQualityMetric calculation"""
        self.mock_dataset.get_quality_score.return_value = 0.85
        metric = DatasetQualityMetric()
        resources = {URLType.DATASET: [self.mock_dataset]}
        score, latency = metric.calculate(resources)
        self.assertEqual(score, 0.85)
        self.assertIsInstance(latency, int)

    def test_code_quality_metric_calculation(self):
        """Test 16: CodeQualityMetric calculation"""
        self.mock_code.get_code_quality_score.return_value = 0.75
        metric = CodeQualityMetric()
        resources = {URLType.CODE: [self.mock_code]}
        score, latency = metric.calculate(resources)
        self.assertEqual(score, 0.75)
        self.assertIsInstance(latency, int)

    def test_metric_with_missing_resources(self):
        """Test 17: Metric calculation with missing resources"""
        metric = DatasetQualityMetric()
        empty_resources: dict[URLType, list[BaseResourceHandler]] = {}
        score, latency = metric.calculate(empty_resources)
        self.assertEqual(score, 0.0)
        self.assertIsInstance(latency, int)


<<<<<<< HEAD
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


    def test_dataset_and_code_score_metric_basic():
        m = DatasetAndCodeScoreMetric()
        dataset = Mock(); dataset.get_downloads.return_value = 2000; dataset.get_tags.return_value = [1,2,3]
        code = Mock(); code.has_tests.return_value = True; code.has_ci_cd.return_value = True; code.get_stars.return_value = 200

        score, _ = m.calculate({URLType.DATASET: [dataset], URLType.CODE: [code]})
        assert score > 0

        # Error branch in dataset
        bad_dataset = Mock()
        bad_dataset.get_downloads.side_effect = Exception("boom")
        score2, _ = m.calculate({URLType.DATASET: [bad_dataset]})
        assert score2 >= 0


    def test_bus_factor_metric_thresholds():
        m = BusFactorMetric()
        a = Mock(); a.get_contributor_count.return_value = 20
        b = Mock(); b.get_contributor_count.return_value = 1

        score, _ = m.calculate({URLType.MODEL: [a], URLType.DATASET: [a], URLType.CODE: [a]})
        assert score >= 0.9

        # Error getting contributor count logs error and falls back
        bad = Mock(); bad.get_contributor_count.side_effect = Exception("boom")
        score2, _ = m.calculate({URLType.MODEL: [bad]})
        assert score2 >= 0.5


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

    if __name__ == "__main__":
        unittest.main(verbosity=2)
=======
if __name__ == "__main__":
    unittest.main(verbosity=2)
>>>>>>> parent of 763fad3 (90% line coverage, need to consolidate tests and improve error message production)
