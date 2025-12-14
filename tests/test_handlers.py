import json
import os
import unittest
from unittest.mock import Mock, patch

from handlers import ingest_handler, registry_handler
from resource_handlers import CodeHandler, DatasetHandler, ModelHandler


class TestResourceHandlers(unittest.TestCase):
    """Tests for the three resource handlers (model, dataset, code)."""

    # -------------------------
    # ModelHandler
    # -------------------------
    def test_model_handler_initialization(self):
        """ModelHandler correctly parses model_id from URL."""
        url = "https://huggingface.co/google/gemma-3-270m"
        handler = ModelHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.model_id, "google/gemma-3-270m")

    @patch("handlers.model_handler.requests.get")
    def test_model_handler_api_call_and_cache(self, mock_get):
        """ModelHandler.get_huggingface_api_data hits API once and then uses cache."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 123, "likes": 10}
        mock_get.return_value = mock_response

        handler = ModelHandler("https://huggingface.co/google/gemma-3-270m")

        data1 = handler.get_huggingface_api_data()
        data2 = handler.get_huggingface_api_data()

        self.assertEqual(data1["downloads"], 123)
        self.assertEqual(data2["likes"], 10)
        # Should only call the API once due to caching
        mock_get.assert_called_once()

    # -------------------------
    # DatasetHandler - FIXED
    # -------------------------
    def test_dataset_handler_initialization(self):
        """DatasetHandler correctly parses dataset_id from URL."""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        handler = DatasetHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.dataset_id, "xlangai/AgentNet")

    @patch("handlers.dataset_handler.requests.get")
    def test_dataset_handler_api_call(self, mock_get):
        """DatasetHandler.get_huggingface_api_data returns parsed JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 999, "tags": ["nlp", "agents"]}
        mock_get.return_value = mock_response

        handler = DatasetHandler("https://huggingface.co/datasets/xlangai/AgentNet")
        # Call the correct method - get_huggingface_api_data() with NO arguments
        info = handler.get_huggingface_api_data()

        self.assertEqual(info["downloads"], 999)
        self.assertIn("nlp", info["tags"])

    # -------------------------
    # CodeHandler
    # -------------------------
    def test_code_handler_initialization(self):
        """CodeHandler correctly parses repo_path from URL."""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        handler = CodeHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.repo_path, "SkyworkAI/Matrix-Game")

    @patch("handlers.code_handler.requests.get")
    def test_code_handler_api_call(self, mock_get):
        """CodeHandler.get_github_api_data returns parsed JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_get.return_value = mock_response

        handler = CodeHandler("https://github.com/SkyworkAI/Matrix-Game")
        data = handler.get_github_api_data()

        self.assertEqual(data["stargazers_count"], 100)

    def test_code_handler_has_ci_cd_from_repo_tree(self):
        """CodeHandler.has_ci_cd detects CI/CD configs in the tree."""
        handler = CodeHandler("https://github.com/org/repo")

        # Fake repo tree containing a GitHub Actions workflow file
        fake_tree = [
            {"path": "src/main.py"},
            {"path": ".github/workflows/ci.yml"},
        ]

        # Monkeypatch get_repo_tree so we don't hit the network
        handler.get_repo_tree = lambda: fake_tree

        self.assertTrue(handler.has_ci_cd())

    def test_code_handler_has_linting_config_from_repo_tree(self):
        """CodeHandler._has_linting_config detects lint config files."""
        handler = CodeHandler("https://github.com/org/repo")

        fake_tree = [
            {"path": "setup.cfg"},
            {"path": "src/app.py"},
        ]
        handler.get_repo_tree = lambda: fake_tree

        self.assertTrue(handler.has_linting_config())


class TestRegistryAndIngest(unittest.TestCase):
    """
    Tests for registry database logic and model ingestion.

    IMPORTANT: These tests force the registry to use the in-memory
    backend so we never talk to real DynamoDB or AWS.
    """

    def setUp(self):
        # Force the registry module to use its in-memory DB implementation.
        # This avoids any AWS/DynamoDB credential issues.
        registry_handler._db = registry_handler._InMemoryDB()

    def tearDown(self):
        # Reset back to a fresh in-memory DB between tests
        registry_handler._db = registry_handler._InMemoryDB()

    def test_generate_artifact_id_is_deterministic(self):
        """generate_artifact_id returns the same ID for the same name."""
        from handlers.registry_handler import generate_artifact_id

        name = "my-model"
        id1 = generate_artifact_id(name)
        id2 = generate_artifact_id(name)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.isdigit())
        self.assertEqual(len(id1), 10)

    def test_registry_add_list_search_and_reset(self):
        """Add, list, search, and reset using the in-memory registry."""
        registry_handler.add_model(
            name="test-model",
            score=0.9,
            tags="test,tag",
            code_url="https://github.com/example/repo",
            dataset_url="https://huggingface.co/datasets/example/data",
            metadata_json=json.dumps({"key": "value"}),
        )

        models = registry_handler.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "test-model")

        # Search by name
        search_results = registry_handler.search_models("test-model")
        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0]["name"], "test-model")

        # Search by tag
        tag_results = registry_handler.search_models("tag")
        self.assertEqual(len(tag_results), 1)

        # Get by exact name
        artifact = registry_handler.get_artifact_by_name("test-model")
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact["name"], "test-model")

        # Reset should clear registry
        registry_handler.reset_registry()
        self.assertEqual(registry_handler.list_models(), [])

    @patch("handlers.ingest_handler.snapshot_download")
    @patch("handlers.ingest_handler.subprocess.check_output")
    @patch("handlers.ingest_handler.infer_links_from_hf")
    def test_ingest_model_success(self, mock_infer, mock_subproc, mock_download):
        """Successful ingestion when net_score meets threshold."""
        # Prevent any real download; we don't care about this path here
        mock_download.return_value = None

        # Pretend we inferred code and dataset URLs
        mock_infer.return_value = (
            "https://github.com/example/repo",
            "https://huggingface.co/datasets/example/data",
        )

        # Fake scoring output from the CLI evaluator
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

        # Use in-memory DB
        registry_handler._db = registry_handler._InMemoryDB()

        result = ingest_handler.ingest_model("https://huggingface.co/google/bert-base-uncased")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "success")

        # Confirm it was inserted into the in-memory registry
        models = registry_handler.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "bert-base-uncased")

    @patch("handlers.ingest_handler.snapshot_download")
    @patch("handlers.ingest_handler.subprocess.check_output")
    @patch("handlers.ingest_handler.infer_links_from_hf")
    def test_ingest_model_fails_threshold(self, mock_infer, mock_subproc, mock_download):
        """Ingestion fails if net_score is below threshold."""
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

        registry_handler._db = registry_handler._InMemoryDB()

        result = ingest_handler.ingest_model("https://huggingface.co/google/bert-base-uncased")
        self.assertIn("error", result)
        self.assertIn("Model did not meet threshold criteria.", result["error"])

        # Ensure nothing was written into registry on failure
        models = registry_handler.list_models()
        self.assertEqual(models, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)