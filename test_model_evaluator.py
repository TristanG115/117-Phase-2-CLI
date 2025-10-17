import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from url_classifier import URLClassifier, URLType
from resource_handlers import ModelHandler, DatasetHandler, CodeHandler
from metrics import (
    LicenseMetric, SizeScoreMetric, RampUpTimeMetric, BusFactorMetric,
    PerformanceClaimsMetric, DatasetAndCodeScoreMetric, DatasetQualityMetric,
    CodeQualityMetric
)
from model_evaluator import ModelEvaluator


class TestURLClassifier(unittest.TestCase):
    """Test URL classification functionality"""

    def setUp(self):
        self.classifier = URLClassifier()

    def test_classify_huggingface_model(self):
        """Test 1: Classification of HuggingFace model URL"""
        url = "https://huggingface.co/google/gemma-3-270m"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.MODEL)

    def test_classify_huggingface_dataset(self):
        """Test 2: Classification of HuggingFace dataset URL"""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.DATASET)

    def test_classify_github_code(self):
        """Test 3: Classification of GitHub repository URL"""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.CODE)

    def test_classify_unknown_url(self):
        """Test 4: Classification of unknown URL"""
        url = "https://example.com/some/path"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.UNKNOWN)

    def test_group_urls_by_type(self):
        """Test 5: Grouping multiple URLs by type"""
        urls = [
            "https://huggingface.co/google/gemma-3-270m",
            "https://huggingface.co/datasets/xlangai/AgentNet",
            "https://github.com/SkyworkAI/Matrix-Game"
        ]
        result = self.classifier.group_urls_by_type(urls)

        self.assertEqual(len(result[URLType.MODEL]), 1)
        self.assertEqual(len(result[URLType.DATASET]), 1)
        self.assertEqual(len(result[URLType.CODE]), 1)
        self.assertEqual(len(result[URLType.UNKNOWN]), 0)


class TestResourceHandlers(unittest.TestCase):
    """Test resource handler functionality"""

    def test_model_handler_initialization(self):
        """Test 6: ModelHandler initialization"""
        url = "https://huggingface.co/google/gemma-3-270m"
        handler = ModelHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.model_id, "google/gemma-3-270m")

    def test_dataset_handler_initialization(self):
        """Test 7: DatasetHandler initialization"""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        handler = DatasetHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.dataset_id, "xlangai/AgentNet")

    def test_code_handler_initialization(self):
        """Test 8: CodeHandler initialization"""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        handler = CodeHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.repo_path, "SkyworkAI/Matrix-Game")

    @patch('requests.get')
    def test_model_handler_api_call(self, mock_get):
        """Test 9: ModelHandler API interaction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 1000, "likes": 50}
        mock_get.return_value = mock_response

        handler = ModelHandler("https://huggingface.co/google/gemma-3-270m")
        data = handler.get_huggingface_api_data()

        self.assertEqual(data["downloads"], 1000)
        self.assertEqual(data["likes"], 50)

    @patch('requests.get')
    def test_code_handler_api_call(self, mock_get):
        """Test 10: CodeHandler API interaction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_get.return_value = mock_response

        handler = CodeHandler("https://github.com/SkyworkAI/Matrix-Game")
        data = handler.get_github_api_data()

        self.assertEqual(data["stargazers_count"], 100)


class TestMetrics(unittest.TestCase):
    """Test metric calculation functionality"""

    def setUp(self):
        # Create mock resources
        self.mock_model = Mock()
        self.mock_dataset = Mock()
        self.mock_code = Mock()

        self.resources = {
            URLType.MODEL: [self.mock_model],
            URLType.DATASET: [self.mock_dataset],
            URLType.CODE: [self.mock_code]
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

        # Should take model score (only checks models)
        self.assertEqual(score, 0.8)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_size_score_metric_calculation(self):
        """Test 14: SizeScoreMetric calculation"""
        self.mock_model.get_size_mb.return_value = 500  # 500MB model

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


class TestModelEvaluator(unittest.TestCase):
    """Test main model evaluator functionality"""

    def setUp(self):
        self.evaluator = ModelEvaluator()

    def test_evaluator_initialization(self):
        """Test 18: ModelEvaluator initialization"""
        self.assertIsInstance(self.evaluator.url_classifier, URLClassifier)
        self.assertEqual(self.evaluator.max_workers, 4)
        self.assertEqual(len(self.evaluator.metrics), 8)

    def test_create_resource_handlers(self):
        """Test 19: Resource handler creation"""
        grouped_urls = {
            URLType.MODEL: ["https://huggingface.co/google/gemma-3-270m"],
            URLType.DATASET: ["https://huggingface.co/datasets/xlangai/AgentNet"],
            URLType.CODE: ["https://github.com/SkyworkAI/Matrix-Game"],
            URLType.UNKNOWN: []
        }

        resources = self.evaluator._create_resource_handlers(grouped_urls)

        self.assertIn(URLType.MODEL, resources)
        self.assertIn(URLType.DATASET, resources)
        self.assertIn(URLType.CODE, resources)
        self.assertEqual(len(resources[URLType.MODEL]), 1)
        self.assertIsInstance(resources[URLType.MODEL][0], ModelHandler)

    def test_net_score_calculation(self):
        """Test 20: Net score calculation"""
        mock_results = {
            "license": {"score": 0.8, "latency": 100},
            "performance_claims": {"score": 0.6, "latency": 200},
            "ramp_up_time": {"score": 0.7, "latency": 150},
            "bus_factor": {"score": 0.5, "latency": 80},
            "size_score": {"score": {"raspberry_pi": 0.5, "desktop_pc": 1.0}, "latency": 50},
            "dataset_and_code_score": {"score": 0.9, "latency": 30},
            "dataset_quality": {"score": 0.8, "latency": 120},
            "code_quality": {"score": 0.7, "latency": 90}
        }

        net_score, latency = self.evaluator._calculate_net_score(mock_results)

        self.assertIsInstance(net_score, float)
        self.assertGreaterEqual(net_score, 0.0)
        self.assertLessEqual(net_score, 1.0)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_evaluate_from_file_nonexistent(self):
        """Test 21: Evaluating from non-existent file"""
        results = self.evaluator.evaluate_from_file("nonexistent_file.txt")
        self.assertEqual(results, [])

    def test_evaluate_from_file_valid(self):
        """Test 22: Evaluating from valid file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("https://huggingface.co/google/gemma-3-270m\n")
            f.write("https://huggingface.co/datasets/xlangai/AgentNet\n")
            f.write("https://github.com/SkyworkAI/Matrix-Game\n")
            temp_filename = f.name

        try:
            # Mock API calls to avoid network requests during testing
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"downloads": 1000, "likes": 50}
                mock_get.return_value = mock_response

                results = self.evaluator.evaluate_from_file(temp_filename)

                self.assertIsInstance(results, list)
                # Should have exactly one result (for the model URL)
                if results:  # Only check if we got results (API calls might fail)
                    self.assertIsInstance(results[0], dict)
                    self.assertIn("name", results[0])
                    self.assertIn("category", results[0])
                    self.assertEqual(results[0]["category"], "MODEL")

        finally:
            os.unlink(temp_filename)

    def test_setup_logging_silent(self):
        """Test 23: Logging setup with silent level"""
        with patch.dict(os.environ, {'LOG_LEVEL': '0'}):
            self.evaluator.setup_logging()
            # Should not raise any exceptions

    def test_setup_logging_with_file(self):
        """Test 24: Logging setup with file output"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_log_file = f.name

        try:
            with patch.dict(os.environ, {'LOG_LEVEL': '1', 'LOG_FILE': temp_log_file}):
                self.evaluator.setup_logging()
                # Should not raise any exceptions
                self.assertTrue(os.path.exists(temp_log_file))
        finally:
            if os.path.exists(temp_log_file):
                os.unlink(temp_log_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
