import os
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

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
        self.assertEqual(len(self.evaluator.metrics), 8)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
