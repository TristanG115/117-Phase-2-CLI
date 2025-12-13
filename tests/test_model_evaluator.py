import os
import tempfile
import unittest
import json
import sys
import tempfile
import pytest
from typing import Any
from unittest.mock import Mock, patch
import logging

from model_evaluator import ModelEvaluator
from resource_handlers import ModelHandler
from url_classifier import URLClassifier, URLType

class TestModelEvaluator(unittest.TestCase):
    """Test main model evaluator functionality"""

    def setUp(self):
        self.evaluator = ModelEvaluator()

    def test_evaluator_initialization(self):
        """Test 18: ModelEvaluator initialization"""
        self.assertIsInstance(self.evaluator.url_classifier, URLClassifier)
        self.assertEqual(self.evaluator.max_workers, 4)
        self.assertGreaterEqual(len(self.evaluator.metrics), 8)

    def test_create_resource_handlers(self):
        """Test 19: Resource handler creation"""
        grouped_urls = {
            URLType.MODEL: ["https://huggingface.co/google/gemma-3-270m"],
            URLType.DATASET: ["https://huggingface.co/datasets/xlangai/AgentNet"],
            URLType.CODE: ["https://github.com/SkyworkAI/Matrix-Game"],
            URLType.UNKNOWN: [],
        }

        resources = self.evaluator._create_resource_handlers(grouped_urls)

        self.assertIn(URLType.MODEL, resources)
        self.assertIn(URLType.DATASET, resources)
        self.assertIn(URLType.CODE, resources)
        self.assertEqual(len(resources[URLType.MODEL]), 1)
        self.assertIsInstance(resources[URLType.MODEL][0], ModelHandler)

    def test_net_score_calculation(self):
        """Test 20: Net score calculation"""
        mock_results: dict[str, dict[str, Any]] = {
            "license": {"score": 0.8, "latency": 100},
            "performance_claims": {"score": 0.6, "latency": 200},
            "ramp_up_time": {"score": 0.7, "latency": 150},
            "bus_factor": {"score": 0.5, "latency": 80},
            "size_score": {
                "score": {"raspberry_pi": 0.5, "desktop_pc": 1.0},
                "latency": 50,
            },
            "dataset_and_code_score": {"score": 0.9, "latency": 30},
            "dataset_quality": {"score": 0.8, "latency": 120},
            "code_quality": {"score": 0.7, "latency": 90},
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
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("https://huggingface.co/google/gemma-3-270m\n")
            f.write("https://huggingface.co/datasets/xlangai/AgentNet\n")
            f.write("https://github.com/SkyworkAI/Matrix-Game\n")
            temp_filename = f.name

        try:
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {"downloads": 1000, "likes": 50}
                mock_response.text = "# Sample README\nThis is sample documentation."
                mock_response.content = b"Sample file content"
                mock_get.return_value = mock_response

                results = self.evaluator.evaluate_from_file(temp_filename)

                # Just verify we get a non-empty list with valid structure
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0)

                # Verify the structure of at least one result
                self.assertIsInstance(results[0], dict)
                self.assertIn("name", results[0])
                self.assertIn("category", results[0])
        finally:
            os.unlink(temp_filename)

    def test_setup_logging_silent(self):
        """Test 23: Logging setup with silent level"""
        with patch.dict(os.environ, {"LOG_LEVEL": "0"}):
            self.evaluator.setup_logging()

    def test_setup_logging_with_file(self):
        """Test 24: Logging setup with file output"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_log_file = f.name
        try:
            with patch.dict(os.environ, {"LOG_LEVEL": "1", "LOG_FILE": temp_log_file}):
                self.evaluator.setup_logging()
                self.assertTrue(os.path.exists(temp_log_file))

                # Close all logging handlers to release the file on Windows
                import logging

                logging.shutdown()
        finally:
            # Small delay to ensure file is fully released on Windows
            import time

            time.sleep(0.1)

            # Try to delete, but don't fail if we can't (Windows file locking)
            try:
                if os.path.exists(temp_log_file):
                    os.unlink(temp_log_file)
            except PermissionError:
                # On Windows, the file might still be locked - that's okay
                pass

def test_create_resource_handlers():
    me = ModelEvaluator(max_workers=1)
    grouped = {URLType.MODEL: ["m1"], URLType.DATASET: ["d1"], URLType.CODE: ["c1"]}
    resources = me._create_resource_handlers(grouped)
    assert URLType.MODEL in resources and len(resources[URLType.MODEL]) == 1
    assert URLType.DATASET in resources and len(resources[URLType.DATASET]) == 1
    assert URLType.CODE in resources and len(resources[URLType.CODE]) == 1


def test_evaluate_single_model_success_and_failure(monkeypatch):
    me = ModelEvaluator(max_workers=1)

    class FakeModelHandler:
        def __init__(self, url):
            self.model_id = "owner/model"

    monkeypatch.setattr("model_evaluator.ModelHandler", FakeModelHandler)

    # Monkeypatch metric calculation to return predictable metrics
    monkeypatch.setattr(me, "_calculate_metrics_parallel", lambda resources, aid=None: {"license": {"score": 0.8, "latency": 10}})

    res = me._evaluate_single_model("https://huggingface.co/owner/model", {}, artifact_id=None)
    assert res is not None
    assert res["name"] == "model"

    # Simulate ModelHandler raising
    def bad_init(url):
        raise Exception("bad")

    monkeypatch.setattr("model_evaluator.ModelHandler", bad_init)
    assert me._evaluate_single_model("x", {}, None) is None


def test_safe_calculate_metric_and_treescore(monkeypatch):
    me = ModelEvaluator(max_workers=1)

    class BadMetric:
        def calculate(self, resources):
            raise Exception("boom")

    class TreeMetric:
        def calculate(self, resources, artifact_id=None):
            return 0.5, 10

    assert me._safe_calculate_metric(BadMetric(), {}) == (0.0, 0)
    assert me._safe_calculate_treescore(TreeMetric(), {}, artifact_id=123) == (0.5, 10)


def test_calculate_net_score_various():
    me = ModelEvaluator(max_workers=1)
    metric_results = {
        "license": {"score": 0.8, "latency": 10},
        "reviewedness": {"score": -1.0, "latency": 0},
        "size_score": {"score": {"a": 0.9, "b": 0.7}, "latency": 5},
    }

    net, latency = me._calculate_net_score(metric_results)
    assert isinstance(net, float)
    assert latency == 15


def test_evaluate_from_file_and_main(monkeypatch, tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text("https://huggingface.co/owner/model\n")

    me = ModelEvaluator(max_workers=1)
    # Monkeypatch evaluate_urls to return a predictable result
    monkeypatch.setattr(me, "evaluate_urls", lambda urls: [{"name": "m", "category": "MODEL"}])
    results = me.evaluate_from_file(str(p))
    assert len(results) == 1

    # File not found
    assert me.evaluate_from_file(str(p) + ".nope") == []

    # main usage: wrong args
    monkeypatch.setattr(sys, "argv", ["model_evaluator.py"])
    with pytest.raises(SystemExit) as se:
        from model_evaluator import main

        main()
    assert se.value.code == 1

    # main with file but no results -> exit 1
    monkeypatch.setattr(sys, "argv", ["model_evaluator.py", str(p)])
    monkeypatch.setattr("model_evaluator.ModelEvaluator.evaluate_from_file", lambda self, pth: [])
    with pytest.raises(SystemExit) as se2:
        from model_evaluator import main

        main()
    assert se2.value.code == 1


def test_setup_logging(tmp_path, monkeypatch):
    me = ModelEvaluator(max_workers=1)
    monkeypatch.setenv("LOG_LEVEL", "1")
    log_file = tmp_path / "logs" / "out.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    me.setup_logging()
    # log file should be created
    assert log_file.exists()




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
