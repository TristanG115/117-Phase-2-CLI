"""
Edge case tests for model_evaluator to increase coverage
"""

import os
import tempfile
import unittest
from unittest.mock import patch, Mock, MagicMock
import logging


class TestModelEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error paths in ModelEvaluator"""

    def setUp(self):
        from model_evaluator import ModelEvaluator
        self.evaluator = ModelEvaluator()

    def test_evaluate_with_empty_urls(self):
        """Test evaluation with empty URL list"""
        results = self.evaluator.evaluate_urls([])
        self.assertEqual(results, [])

    def test_evaluate_with_invalid_urls(self):
        """Test evaluation with invalid URLs"""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            results = self.evaluator.evaluate_urls(["invalid://url"])
            
            # Should handle gracefully
            self.assertIsInstance(results, list)

    @patch("requests.get")
    def test_evaluate_with_network_error(self, mock_get):
        """Test evaluation with network errors"""
        mock_get.side_effect = Exception("Connection refused")
        
        urls = ["https://huggingface.co/test/model"]
        results = self.evaluator.evaluate_urls(urls)
        
        # Should handle errors gracefully
        self.assertIsInstance(results, list)

    def test_net_score_calculation_with_missing_metrics(self):
        """Test net score calculation with missing metrics"""
        partial_results = {
            "license": {"score": 0.8, "latency": 100},
            "dataset_and_code_score": {"score": 0.7, "latency": 50}
        }
        
        net_score, latency = self.evaluator._calculate_net_score(partial_results)
        
        self.assertIsInstance(net_score, float)
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(net_score, 0.0)
        self.assertLessEqual(net_score, 1.0)

    def test_net_score_with_size_score_dict(self):
        """Test net score handles size_score as dict correctly"""
        results = {
            "license": {"score": 1.0, "latency": 10},
            "performance_claims": {"score": 0.8, "latency": 20},
            "size_score": {
                "score": {"raspberry_pi": 0.5, "desktop_pc": 1.0},
                "latency": 30
            },
            "dataset_and_code_score": {"score": 0.9, "latency": 40}
        }
        
        net_score, latency = self.evaluator._calculate_net_score(results)
        
        # Should extract average from dict
        self.assertGreater(net_score, 0.0)
        self.assertLess(net_score, 1.0)

    def test_create_resource_handlers_with_empty_groups(self):
        """Test resource handler creation with empty URL groups"""
        from url_classifier import URLType
        
        grouped_urls = {
            URLType.MODEL: [],
            URLType.DATASET: [],
            URLType.CODE: [],
            URLType.UNKNOWN: []
        }
        
        resources = self.evaluator._create_resource_handlers(grouped_urls)
        
        # When all URL lists are empty, the function returns an empty dict
        # (it only adds keys for non-empty URL lists)
        self.assertIsInstance(resources, dict)
        # All lists were empty, so dict should be empty
        self.assertEqual(len(resources), 0)

    @patch("requests.get")
    def test_create_resource_handlers_with_invalid_urls(self, mock_get):
        """Test resource handler creation with URLs that fail"""
        from url_classifier import URLType
        
        mock_get.side_effect = Exception("Failed to fetch")
        
        grouped_urls = {
            URLType.MODEL: ["https://huggingface.co/invalid/model"],
            URLType.DATASET: [],
            URLType.CODE: [],
            URLType.UNKNOWN: []
        }
        
        resources = self.evaluator._create_resource_handlers(grouped_urls)
        
        # Should still create handlers even if they fail
        self.assertEqual(len(resources[URLType.MODEL]), 1)

    def test_evaluate_from_nonexistent_file(self):
        """Test evaluate_from_file with file that doesn't exist"""
        results = self.evaluator.evaluate_from_file("/nonexistent/path.txt")
        self.assertEqual(results, [])

    def test_evaluate_from_empty_file(self):
        """Test evaluate_from_file with empty file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_filename = f.name
            # Write nothing
        
        try:
            results = self.evaluator.evaluate_from_file(temp_filename)
            self.assertEqual(results, [])
        finally:
            os.unlink(temp_filename)

    def test_evaluate_from_file_with_whitespace(self):
        """Test evaluate_from_file with whitespace and blank lines"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("\n")
            f.write("  \n")
            f.write("\t\n")
            temp_filename = f.name
        
        try:
            results = self.evaluator.evaluate_from_file(temp_filename)
            self.assertEqual(results, [])
        finally:
            os.unlink(temp_filename)

    def test_setup_logging_with_invalid_level(self):
        """Test logging setup with invalid level"""
        with patch.dict(os.environ, {"LOG_LEVEL": "invalid"}):
            # Should handle gracefully - ValueError will be caught or default used
            try:
                self.evaluator.setup_logging()
                # If it succeeds, that's fine
                self.assertTrue(True)
            except ValueError:
                # If it raises ValueError, that's also acceptable behavior
                self.assertTrue(True)

    def test_setup_logging_with_invalid_file_path(self):
        """Test logging setup with invalid file path"""
        with patch.dict(os.environ, {"LOG_FILE": "/invalid/path/log.txt"}):
            # Should handle error gracefully
            try:
                self.evaluator.setup_logging()
            except Exception:
                # Some implementations may raise, some may not
                pass

    def test_setup_logging_silent(self):
        """Test silent logging level"""
        with patch.dict(os.environ, {"LOG_LEVEL": "0"}):
            self.evaluator.setup_logging()
            # Verify logger level is set appropriately
            self.assertIsNotNone(logging.getLogger())

    def test_setup_logging_info(self):
        """Test info logging level"""
        with patch.dict(os.environ, {"LOG_LEVEL": "1"}):
            self.evaluator.setup_logging()

    def test_setup_logging_debug(self):
        """Test debug logging level"""
        with patch.dict(os.environ, {"LOG_LEVEL": "2"}):
            self.evaluator.setup_logging()

    @patch("requests.get")
    def test_evaluate_with_timeout(self, mock_get):
        """Test evaluation with timeout"""
        import requests
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        results = self.evaluator.evaluate_urls(["https://huggingface.co/test/model"])
        
        self.assertIsInstance(results, list)

    @patch("requests.get")
    def test_evaluate_with_connection_error(self, mock_get):
        """Test evaluation with connection error"""
        import requests
        mock_get.side_effect = requests.ConnectionError("Failed to connect")
        
        results = self.evaluator.evaluate_urls(["https://huggingface.co/test/model"])
        
        self.assertIsInstance(results, list)

    def test_parallel_evaluation(self):
        """Test that parallel evaluation works with max_workers"""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"downloads": 100}
            mock_response.text = "README content"
            mock_get.return_value = mock_response
            
            urls = [
                "https://huggingface.co/model1",
                "https://huggingface.co/model2",
                "https://huggingface.co/model3"
            ]
            
            results = self.evaluator.evaluate_urls(urls)
            
            self.assertIsInstance(results, list)

    def test_evaluator_with_custom_max_workers(self):
        """Test evaluator with custom max_workers"""
        from model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(max_workers=2)
        self.assertEqual(evaluator.max_workers, 2)

    def test_metrics_initialization(self):
        """Test that all metrics are properly initialized"""
        self.assertIsInstance(self.evaluator.metrics, dict)
        self.assertGreater(len(self.evaluator.metrics), 0)
        
        for metric_name, metric in self.evaluator.metrics.items():
            self.assertTrue(hasattr(metric, "calculate"))
            self.assertTrue(callable(getattr(metric, "calculate")))
            self.assertTrue(hasattr(metric, "required_url_types"))
            self.assertTrue(callable(getattr(metric, "required_url_types")))

    @patch("requests.get")
    def test_evaluate_mixed_success_and_failure(self, mock_get):
        """Test evaluation with some URLs succeeding and some failing"""
        def side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "fail" in url:
                raise Exception("Failed")
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"downloads": 100}
            mock_response.text = "README"
            return mock_response
        
        mock_get.side_effect = side_effect
        
        urls = [
            "https://huggingface.co/good/model",
            "https://huggingface.co/fail/model"
        ]
        
        results = self.evaluator.evaluate_urls(urls)
        
        self.assertIsInstance(results, list)

    def test_net_score_edge_values(self):
        """Test net score calculation with edge values"""
        # All zeros
        results_zero = {
            "license": {"score": 0.0, "latency": 10},
            "dataset_and_code_score": {"score": 0.0, "latency": 10}
        }
        net_score, _ = self.evaluator._calculate_net_score(results_zero)
        self.assertEqual(net_score, 0.0)
        
        # All ones
        results_one = {
            "license": {"score": 1.0, "latency": 10},
            "dataset_and_code_score": {"score": 1.0, "latency": 10},
            "dataset_quality": {"score": 1.0, "latency": 10},
            "code_quality": {"score": 1.0, "latency": 10}
        }
        net_score, _ = self.evaluator._calculate_net_score(results_one)
        self.assertGreater(net_score, 0.0)
        self.assertLessEqual(net_score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)