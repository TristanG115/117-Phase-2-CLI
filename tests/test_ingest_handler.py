"""
Additional tests for ingest_handler to increase coverage
"""

import json
import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os


class TestIngestHandlerCoverage(unittest.TestCase):
    """Additional coverage tests for ingest_handler"""

    @patch("handlers.ingest_handler.HfApi")
    @patch("handlers.ingest_handler.RepoCard")
    def test_extract_metadata_with_card_data(self, mock_card, mock_api):
        """Test extracting metadata with cardData"""
        from handlers.ingest_handler import _extract_metadata
        
        mock_info = MagicMock()
        mock_info.cardData = {"license": "mit", "datasets": ["test-dataset"]}
        
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = mock_info
        
        mock_card.load.return_value.content = "# Model Card\nThis is a test model"
        
        metadata, card_text = _extract_metadata("test/model", mock_api_instance)
        
        self.assertIsInstance(metadata, dict)
        self.assertIsInstance(card_text, str)
        self.assertIn("license", metadata)

    @patch("handlers.ingest_handler.HfApi")
    def test_extract_metadata_failure(self, mock_api):
        """Test metadata extraction failure"""
        from handlers.ingest_handler import _extract_metadata
        
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.side_effect = Exception("API Error")
        
        metadata, card_text = _extract_metadata("test/model", mock_api_instance)
        
        self.assertEqual(metadata, {})
        self.assertEqual(card_text, "")

    def test_extract_github_url_patterns(self):
        """Test various GitHub URL patterns"""
        from handlers.ingest_handler import _extract_github_url
        
        # Test with https URL
        text1 = "Code at https://github.com/user/repo"
        url1 = _extract_github_url(text1)
        self.assertEqual(url1, "https://github.com/user/repo")
        
        # Test without https
        text2 = "See github.com/user/repo for details"
        url2 = _extract_github_url(text2)
        self.assertEqual(url2, "https://github.com/user/repo")
        
        # Test with .git extension
        text3 = "Clone from https://github.com/user/repo.git"
        url3 = _extract_github_url(text3)
        self.assertEqual(url3, "https://github.com/user/repo")
        
        # Test not found
        text4 = "No repository here"
        url4 = _extract_github_url(text4)
        self.assertEqual(url4, "unknown")

    def test_extract_dataset_url_direct(self):
        """Test direct dataset URL extraction"""
        from handlers.ingest_handler import _extract_dataset_url_direct
        
        # Test full URL
        text1 = "Dataset: https://huggingface.co/datasets/org/dataset"
        url1 = _extract_dataset_url_direct(text1)
        self.assertEqual(url1, "https://huggingface.co/datasets/org/dataset")
        
        # Test without https
        text2 = "Using huggingface.co/datasets/org/dataset"
        url2 = _extract_dataset_url_direct(text2)
        self.assertEqual(url2, "https://huggingface.co/datasets/org/dataset")
        
        # Test not found
        text3 = "No dataset URL"
        url3 = _extract_dataset_url_direct(text3)
        self.assertIsNone(url3)

    def test_extract_dataset_from_metadata(self):
        """Test dataset extraction from metadata"""
        from handlers.ingest_handler import _extract_dataset_from_metadata
        
        # Test with list
        metadata1 = {"datasets": ["squad", "glue"]}
        result1 = _extract_dataset_from_metadata(metadata1)
        self.assertEqual(result1, "squad")
        
        # Test with string
        metadata2 = {"datasets": "squad"}
        result2 = _extract_dataset_from_metadata(metadata2)
        self.assertEqual(result2, "squad")
        
        # Test not found
        metadata3 = {}
        result3 = _extract_dataset_from_metadata(metadata3)
        self.assertIsNone(result3)

    def test_extract_dataset_from_yaml(self):
        """Test dataset extraction from YAML"""
        from handlers.ingest_handler import _extract_dataset_from_yaml
        
        text = """
---
datasets:
  - squad
  - glue
---
"""
        result = _extract_dataset_from_yaml(text)
        self.assertEqual(result, "squad")
        
        # Test not found
        text2 = "No YAML here"
        result2 = _extract_dataset_from_yaml(text2)
        self.assertIsNone(result2)

    def test_extract_dataset_from_prose(self):
        """Test dataset extraction from prose"""
        from handlers.ingest_handler import _extract_dataset_from_prose
        
        # Test "trained on" pattern
        text1 = "This model was trained on the SQUAD dataset"
        result1 = _extract_dataset_from_prose(text1)
        self.assertEqual(result1, "SQUAD")
        
        # Test "using" pattern
        text2 = "We trained using GLUE dataset"
        result2 = _extract_dataset_from_prose(text2)
        self.assertEqual(result2, "GLUE")
        
        # Test "dataset:" pattern
        text3 = "dataset: squad-v2"
        result3 = _extract_dataset_from_prose(text3)
        self.assertEqual(result3, "squad-v2")
        
        # Test not found
        text4 = "No dataset mentioned"
        result4 = _extract_dataset_from_prose(text4)
        self.assertIsNone(result4)

    def test_construct_dataset_url(self):
        """Test dataset URL construction"""
        from handlers.ingest_handler import _construct_dataset_url
        
        url1 = _construct_dataset_url("squad")
        self.assertEqual(url1, "https://huggingface.co/datasets/squad")
        
        url2 = _construct_dataset_url("  SQUAD  ")
        self.assertEqual(url2, "https://huggingface.co/datasets/squad")
        
        url3 = _construct_dataset_url("org/dataset")
        self.assertEqual(url3, "https://huggingface.co/datasets/org/dataset")

    @patch("requests.head")
    def test_validate_url_success(self, mock_head):
        """Test URL validation success"""
        from handlers.ingest_handler import validate_url
        
        mock_head.return_value.status_code = 200
        
        result = validate_url("https://github.com/test/repo", "github")
        self.assertTrue(result)

    @patch("requests.head")
    def test_validate_url_failure(self, mock_head):
        """Test URL validation failure"""
        from handlers.ingest_handler import validate_url
        
        mock_head.return_value.status_code = 404
        
        result = validate_url("https://github.com/test/nonexistent", "github")
        self.assertFalse(result)

    @patch("requests.head")
    def test_validate_url_exception(self, mock_head):
        """Test URL validation with exception"""
        from handlers.ingest_handler import validate_url
        
        mock_head.side_effect = Exception("Network error")
        
        result = validate_url("https://github.com/test/repo", "github")
        self.assertFalse(result)

    def test_validate_url_unknown(self):
        """Test validation of 'unknown' URL"""
        from handlers.ingest_handler import validate_url
        
        result = validate_url("unknown", "github")
        self.assertFalse(result)

    def test_ensure_required_metrics(self):
        """Test ensuring required metrics are present"""
        from handlers.ingest_handler import _ensure_required_metrics
        
        result = {"net_score": 0.8}
        _ensure_required_metrics(result)
        
        self.assertEqual(result["dataset_and_code_score"], -1)
        self.assertEqual(result["dataset_quality"], -1)
        self.assertEqual(result["code_quality"], -1)
        self.assertEqual(result["reproducibility"], -1)
        self.assertEqual(result["reviewedness"], -1)
        self.assertEqual(result["tree_score"], -1)

    def test_check_threshold_pass(self):
        """Test threshold check passing"""
        from handlers.ingest_handler import _check_threshold
        
        result = {
            "net_score": 0.9,
            "dataset_and_code_score": 0.8,
            "dataset_quality": 0.7,
            "code_quality": 0.6,
            "license": 1.0
        }
        
        passes = _check_threshold(result, 0.5)
        self.assertTrue(passes)

    def test_check_threshold_fail(self):
        """Test threshold check failing"""
        from handlers.ingest_handler import _check_threshold
        
        result = {
            "net_score": 0.9,
            "dataset_and_code_score": 0.3,  # Below threshold
            "dataset_quality": 0.7,
            "code_quality": 0.6,
        }
        
        passes = _check_threshold(result, 0.5)
        self.assertFalse(passes)

    def test_check_threshold_with_minus_one(self):
        """Test threshold check ignores -1 values"""
        from handlers.ingest_handler import _check_threshold
        
        result = {
            "net_score": 0.9,
            "dataset_and_code_score": 0.8,
            "dataset_quality": -1,  # Should be ignored
            "code_quality": 0.6,
        }
        
        passes = _check_threshold(result, 0.5)
        self.assertTrue(passes)

    @patch("handlers.ingest_handler.HfApi")
    @patch("requests.get")
    def test_calculate_artifact_size_huggingface(self, mock_get, mock_api):
        """Test artifact size calculation for HuggingFace"""
        from handlers.ingest_handler import _calculate_artifact_size
        
        mock_info = MagicMock()
        mock_file1 = MagicMock()
        mock_file1.size = 1024 * 1024 * 100  # 100 MB
        mock_file2 = MagicMock()
        mock_file2.size = 1024 * 1024 * 50  # 50 MB
        mock_info.siblings = [mock_file1, mock_file2]
        
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = mock_info
        mock_api.return_value = mock_api_instance
        
        size = _calculate_artifact_size("https://huggingface.co/test/model", "model")
        self.assertGreater(size, 0)

    @patch("requests.get")
    def test_calculate_artifact_size_github(self, mock_get):
        """Test artifact size calculation for GitHub"""
        from handlers.ingest_handler import _calculate_artifact_size
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"size": 10240}  # KB
        mock_get.return_value = mock_response
        
        size = _calculate_artifact_size("https://github.com/test/repo", "code")
        self.assertGreater(size, 0)

    def test_calculate_artifact_size_unknown(self):
        """Test artifact size with unknown URL"""
        from handlers.ingest_handler import _calculate_artifact_size
        
        size = _calculate_artifact_size("unknown", "model")
        self.assertEqual(size, 0.0)

    @patch("handlers.ingest_handler.registry_handler")
    @patch("handlers.ingest_handler.infer_links_from_hf")
    @patch("handlers.ingest_handler.score_model_with_evaluator")
    def test_ingest_dataset(self, mock_score, mock_infer, mock_registry):
        """Test dataset ingestion"""
        from handlers.ingest_handler import ingest_dataset
        
        mock_registry._db = MagicMock()
        mock_registry.add_artifact.return_value = "dataset-123"
        
        result = ingest_dataset("https://huggingface.co/datasets/test/data")
        
        self.assertEqual(result["status"], "success")
        self.assertIn("artifact_id", result)
        self.assertIn("dataset", result)

    @patch("handlers.ingest_handler.registry_handler")
    @patch("handlers.ingest_handler._calculate_artifact_size")
    def test_ingest_code(self, mock_size, mock_registry):
        """Test code repository ingestion"""
        from handlers.ingest_handler import ingest_code
        
        mock_size.return_value = 50.0
        mock_registry._db = MagicMock()
        mock_registry.add_artifact.return_value = "code-123"
        
        result = ingest_code("https://github.com/test/repo")
        
        self.assertEqual(result["status"], "success")
        self.assertIn("artifact_id", result)
        self.assertIn("code", result)

    @patch("handlers.ingest_handler.ingest_model")
    def test_batch_ingest(self, mock_ingest):
        """Test batch ingestion"""
        from handlers.ingest_handler import batch_ingest
        
        mock_ingest.return_value = {"status": "success", "model": "test"}
        
        urls = [
            "https://huggingface.co/model1",
            "https://huggingface.co/model2"
        ]
        
        results = batch_ingest(urls)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_ingest.call_count, 2)

    @patch("handlers.ingest_handler.ingest_model")
    def test_batch_ingest_with_error(self, mock_ingest):
        """Test batch ingestion with error"""
        from handlers.ingest_handler import batch_ingest
        
        mock_ingest.side_effect = [
            {"status": "success"},
            Exception("Failed")
        ]
        
        urls = [
            "https://huggingface.co/model1",
            "https://huggingface.co/model2"
        ]
        
        results = batch_ingest(urls)
        
        self.assertEqual(len(results), 2)
        self.assertIn("error", results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)