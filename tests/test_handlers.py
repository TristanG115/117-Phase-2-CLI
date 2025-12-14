import json
import os
import unittest
from unittest.mock import Mock, patch

from handlers import ingest_handler, registry_handler
from resource_handlers import CodeHandler, DatasetHandler, ModelHandler


class TestResourceHandlers(unittest.TestCase):
    """Test resource handler functionality"""

    def test_model_handler_initialization(self):
        """Test 6: ModelHandler initialization"""
        url = "https://huggingface.co/google/gemma-3-270m"
        handler = ModelHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.model_id, "google/gemma-3-270m")

<<<<<<< HEAD
=======
    @patch("handlers.model_handler.requests.get")
    def test_model_handler_api_call_and_cache(self, mock_get):
        """ModelHandler.get_huggingface_api_data hits API once and then uses cache."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 123, "likes": 10}
        mock_get.return_value = mock_response

        handler = ModelHandler("https://huggingface.co/google/gemma-3-270m")

        data1 = handler.get_hf_model_info()
        data2 = handler.get_hf_model_info()

        self.assertEqual(data1["downloads"], 123)
        self.assertEqual(data2["likes"], 10)
        # Should only call the API once due to caching
        mock_get.assert_called_once()

    # -------------------------
    # DatasetHandler
    # -------------------------
>>>>>>> parent of 763fad3 (90% line coverage, need to consolidate tests and improve error message production)
    def test_dataset_handler_initialization(self):
        """Test 7: DatasetHandler initialization"""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        handler = DatasetHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.dataset_id, "xlangai/AgentNet")

<<<<<<< HEAD
=======
    @patch("handlers.dataset_handler.requests.get")
    def test_dataset_handler_api_call(self, mock_get):
        """DatasetHandler.get_hf_dataset_info returns parsed JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 999, "tags": ["nlp", "agents"]}
        mock_get.return_value = mock_response

        handler = DatasetHandler("https://huggingface.co/datasets/xlangai/AgentNet")
        info = handler.get_hf_dataset_info()

        self.assertEqual(info["downloads"], 999)
        self.assertIn("nlp", info["tags"])

    # -------------------------
    # CodeHandler
    # -------------------------
>>>>>>> parent of 763fad3 (90% line coverage, need to consolidate tests and improve error message production)
    def test_code_handler_initialization(self):
        """Test 8: CodeHandler initialization"""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        handler = CodeHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.repo_path, "SkyworkAI/Matrix-Game")

    @patch("requests.get")
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

    @patch("requests.get")
    def test_code_handler_api_call(self, mock_get):
        """Test 10: CodeHandler API interaction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_get.return_value = mock_response

        handler = CodeHandler("https://github.com/SkyworkAI/Matrix-Game")
        data = handler.get_github_api_data()

        self.assertEqual(data["stargazers_count"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRegistryAndIngest(unittest.TestCase):
    """Tests for registry database logic and model ingestion"""

    def setUp(self):
        # Ensure a clean registry
        if os.path.exists("registry.db"):
            os.remove("registry.db")
        registry_handler.init_registry()

    def tearDown(self):
        if os.path.exists("registry.db"):
            os.remove("registry.db")

    def test_registry_add_and_list(self):
        """Test: Add, list, and reset registry"""
        registry_handler.add_model(
            name="test-model",
            score=0.9,
            tags="test",
            code_url="https://github.com/example/repo",
            dataset_url="https://huggingface.co/datasets/example/data",
            metadata_json=json.dumps({"key": "value"}),
        )
        models = registry_handler.list_models()
        self.assertTrue(len(models) == 1)
        self.assertEqual(models[0]["name"], "test-model")

        # Reset should clear registry
        registry_handler.reset_registry()
        models = registry_handler.list_models()
        self.assertEqual(models, [])

    @patch("handlers.registry_handler.add_model")
    @patch("handlers.ingest_handler.snapshot_download")
    @patch("handlers.ingest_handler.subprocess.check_output")
    @patch("handlers.ingest_handler.infer_links_from_hf")
    def test_ingest_model_success(self, mock_infer, mock_subproc, mock_download, mock_add_model):
        """Test: Successful model ingestion"""
        mock_download.return_value = None
        mock_infer.return_value = (
            "https://github.com/example/repo",
            "https://huggingface.co/datasets/example/data",
        )

        # Mock scoring output
        fake_score = {
            "name": "bert-base-uncased",
            "category": "MODEL",
            "net_score": 0.87,
            "dataset_and_code_score": 1.0,
            "dataset_quality": 1.0,
            "code_quality": 0.9,
            "license": 1.0,
        }
        mock_subproc.return_value = json.dumps(fake_score)

        result = ingest_handler.ingest_model("https://huggingface.co/google/bert-base-uncased")
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        mock_add_model.assert_called_once()

    @patch("handlers.ingest_handler.snapshot_download")
    @patch("handlers.ingest_handler.subprocess.check_output")
    @patch("handlers.ingest_handler.infer_links_from_hf")
    def test_ingest_model_fails_threshold(self, mock_infer, mock_subproc, mock_download):
        """Test: Ingestion fails if score too low"""
        mock_download.return_value = None
        mock_infer.return_value = ("unknown", "unknown")

        fake_score = {
            "name": "bert-base-uncased",
            "category": "MODEL",
            "net_score": 0.3,
            "dataset_and_code_score": 0.2,
            "dataset_quality": 0.1,
            "code_quality": 0.1,
            "license": 1.0,
        }
        mock_subproc.return_value = json.dumps(fake_score)

        result = ingest_handler.ingest_model("https://huggingface.co/google/bert-base-uncased")
        self.assertIn("error", result)
        self.assertIn("Model did not meet threshold criteria.", result["error"])
<<<<<<< HEAD
=======

        # Ensure nothing was written into registry on failure
        models = registry_handler.list_models()
        self.assertEqual(models, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
>>>>>>> parent of 763fad3 (90% line coverage, need to consolidate tests and improve error message production)
