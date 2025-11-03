import unittest
from unittest.mock import Mock

from metrics import (CodeQualityMetric, DatasetQualityMetric, LicenseMetric,
                     SizeScoreMetric)
from url_classifier import URLType


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
        empty_resources = {}
        score, latency = metric.calculate(empty_resources)
        self.assertEqual(score, 0.0)
        self.assertIsInstance(latency, int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
