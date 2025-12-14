
# ==================================================
# BEGIN test_dataset_handler_combined.py
# ==================================================

import sys
from unittest.mock import Mock, patch

import pytest

import handlers.dataset_handler as dh


class FakeResp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text

    def json(self):
        return self._data


def make_api_response(**kwargs):
    r = Mock()
    r.status_code = 200
    r.json.return_value = kwargs
    return r


def test_extract_dataset_id_variants():
    d1 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert d1.dataset_id == "owner/ds"

    d2 = dh.DatasetHandler("https://huggingface.co/datasets/owner")
    assert d2.dataset_id == "owner"

    d3 = dh.DatasetHandler("https://huggingface.co/notdatasets/owner")
    assert d3.dataset_id == ""


def test_get_hf_api_data_list_response(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # Return list with matching id
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"id": "owner/ds", "downloads": 5}]))
    data = ds.get_huggingface_api_data()
    assert data.get("downloads") == 5

    # Return list with dict but no id match -> should pick first dict
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"foo": 1}, {"bar": 2}]))
    data2 = ds.get_huggingface_api_data()
    assert isinstance(data2, dict)


def test_get_readme_content_and_error(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    # Successful readme
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="hello"))
    assert ds.get_readme_content() == "hello"

    # Exception path
    def raise_req(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr("requests.get", raise_req)
    ds2 = dh.DatasetHandler("https://huggingface.co/datasets/x/y")
    assert ds2.get_readme_content() == ""


def test_has_evaluation_dataset():
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    ds._cache_set("hf_api_data", {"tags": ["Evaluation", "other"]})
    assert ds.has_evaluation_dataset() is True


def test_quality_and_docs_and_license_and_contributors(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    api = {
        "cardData": {"dataset_info": True, "license": "MIT"},
        "description": "x" * 250,
        "downloads": 2000,
        "tags": ["a", "b", "license:Apache-2.0"],
        "siblings": [{}, {}],
        "license": None,
    }
    ds._cache_set("hf_api_data", api)
    # readme short
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="readme"))

    q = ds.get_quality_score()
    assert q > 0.5

    d = ds.get_documentation_score()
    assert d > 0.0

    lic = ds.get_license_score()
    assert lic > 0.0

    contrib = ds.get_contributor_count()
    assert contrib >= 2

    assert ds.get_tags() == api["tags"]
    assert ds.get_downloads() == api["downloads"]
    assert ds.get_description() == api["description"]
    assert ds.get_siblings() == api["siblings"]


def test_quality_component_scoring_and_bounds():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    api_resp = make_api_response(
        cardData={"dataset_info": {"rows": 10}},
        description="x" * 120,
        downloads=15000,
        tags=[1, 2, 3, 4, 5, 6],
        siblings=[{}, {}, {}, {}, {}, {}],
    )
    readme_resp = Mock(); readme_resp.status_code = 200; readme_resp.text = "x" * 1200

    with patch("requests.get", side_effect=[api_resp, readme_resp]):
        q = h.get_quality_score()
        assert q > 0.5

    # Description length scoring
    api_resp2 = make_api_response(description="x" * 60)
    with patch("requests.get", return_value=api_resp2):
        assert h._quality_description_score(api_resp2.json.return_value) == 0.05

    # Downloads thresholds
    assert h._quality_downloads_score({"downloads": 20000}) == 0.2
    assert h._quality_downloads_score({"downloads": 2000}) == 0.15
    assert h._quality_downloads_score({"downloads": 200}) == 0.1
    assert h._quality_downloads_score({"downloads": 20}) == 0.05

    # Tags & siblings
    assert h._quality_tags_score({"tags": [1, 2, 3, 4, 5, 6]}) == 0.15
    assert h._quality_tags_score({"tags": [1, 2]}) == 0.1
    assert h._quality_siblings_score({"siblings": [{}, {}]}) == 0.1


def test_documentation_score_readme_thresholds():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # Small readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 50):
        assert h._doc_readme_score("x" * 50) == 0.0

    # Medium readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 600):
        assert h._doc_readme_score("x" * 600) == 0.2

    # Large readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 1500):
        assert h._doc_readme_score("x" * 1500) == 0.3


def test_license_and_contributor_helpers():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # License via tag
    api = make_api_response(tags=["license:mit"])
    with patch("requests.get", return_value=api):
        assert h.get_license_score() == 1.0

    # No license found (use fresh handler to avoid cache)
    h2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    api2 = make_api_response()
    with patch("requests.get", return_value=api2):
        assert h2.get_license_score() == 0.0

    # Contributor thresholds
    assert h.get_contributor_count() == 1
    assert h.get_contributor_count() == 1


def test_readme_fetch_exception_handling():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    def bad_get(*a, **k):
        raise Exception("boom")

    with patch("requests.get", side_effect=bad_get):
        assert h.get_readme_content() == ""


def test_cached_readme_and_api_cache():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    h._readme_content = "cached"
    # requests.get should not be called
    with patch("requests.get", side_effect=Exception("should not call")):
        assert h.get_readme_content() == "cached"

    h2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds2")
    h2._cache_set("hf_api_data", {"downloads": 42})
    assert h2.get_huggingface_api_data()["downloads"] == 42


def test_documentation_card_and_tag_scores():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert h._doc_card_data_score({"cardData": {"x": 1}}) == 0.2
    assert h._doc_description_score({"description": "x" * 250}) == 0.2
    assert h._doc_tags_score({"tags": [1]}) == 0.05
    assert h._doc_structured_info_score({"cardData": {"dataset_info": {"rows": 1}}}) == 0.15

# ==================================================
# BEGIN test_handlers.py
# ==================================================

import json
import os
import unittest

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

# ==================================================
# BEGIN test_handlers_comprehensive.py
# ==================================================

"""
Comprehensive tests for handler modules to improve coverage
"""
from unittest.mock import Mock, patch, MagicMock
from handlers.code_handler import CodeHandler
from handlers.dataset_handler import DatasetHandler
from handlers.model_handler import ModelHandler


class TestCodeHandlerComprehensive:
    """Comprehensive tests for CodeHandler"""

    @pytest.fixture
    def mock_github_api(self):
        """Mock GitHub API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def code_handler(self):
        """Create a CodeHandler instance"""
        return CodeHandler("https://github.com/owner/repo")

    def test_extract_repo_path(self, code_handler):
        """Test extracting repo path from URL"""
        assert code_handler.repo_path == "owner/repo"

    def test_get_github_api_data_success(self, code_handler, mock_github_api):
        """Test successful GitHub API data fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stargazers_count": 100,
            "forks_count": 20,
            "has_readme": True,
            "updated_at": "2024-01-01"
        }
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data["stargazers_count"] == 100
        assert data["forks_count"] == 20

    def test_get_github_api_data_401(self, code_handler, mock_github_api):
        """Test GitHub API 401 authentication failure"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_404(self, code_handler, mock_github_api):
        """Test GitHub API 404 not found"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_other_error(self, code_handler, mock_github_api):
        """Test GitHub API other error codes"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_exception(self, code_handler, mock_github_api):
        """Test GitHub API exception handling"""
        mock_github_api.side_effect = Exception("Network error")

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_repo_tree_success(self, code_handler, mock_github_api):
        """Test successful repo tree fetch"""
        # Mock API data call
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {"default_branch": "main"}

        # Mock tree call
        mock_tree_response = Mock()
        mock_tree_response.status_code = 200
        mock_tree_response.json.return_value = {
            "tree": [
                {"path": "test/test_file.py", "type": "blob"},
                {"path": ".github/workflows/ci.yml", "type": "blob"}
            ]
        }

        mock_github_api.side_effect = [mock_api_response, mock_tree_response]

        tree = code_handler.get_repo_tree()
        assert len(tree) == 2
        assert tree[0]["path"] == "test/test_file.py"

    def test_get_repo_tree_failure(self, code_handler, mock_github_api):
        """Test repo tree fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        tree = code_handler.get_repo_tree()
        assert tree == []

    def test_get_repo_tree_exception(self, code_handler, mock_github_api):
        """Test repo tree exception handling"""
        mock_github_api.side_effect = Exception("Network error")

        tree = code_handler.get_repo_tree()
        assert tree == []

    def test_has_tests_true(self, code_handler):
        """Test detecting test files"""
        code_handler._repo_tree = [
            {"path": "tests/test_main.py"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_tests() is True

    def test_has_tests_false(self, code_handler):
        """Test no test files"""
        code_handler._repo_tree = [
            {"path": "src/main.py"},
            {"path": "README.md"}
        ]
        assert code_handler.has_tests() is False

    def test_has_ci_cd_true(self, code_handler):
        """Test detecting CI/CD config"""
        code_handler._repo_tree = [
            {"path": ".github/workflows/ci.yml"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_ci_cd() is True

    def test_has_ci_cd_travis(self, code_handler):
        """Test detecting Travis CI"""
        code_handler._repo_tree = [
            {"path": ".travis.yml"}
        ]
        assert code_handler.has_ci_cd() is True

    def test_has_ci_cd_false(self, code_handler):
        """Test no CI/CD config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_ci_cd() is False

    def test_has_linting_config_true(self, code_handler):
        """Test detecting linting config"""
        code_handler._repo_tree = [
            {"path": ".flake8"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_linting_config() is True

    def test_has_linting_config_false(self, code_handler):
        """Test no linting config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_linting_config() is False

    def test_get_python_file_count(self, code_handler):
        """Test counting Python files"""
        code_handler._repo_tree = [
            {"path": "src/main.py"},
            {"path": "tests/test_main.py"},
            {"path": "README.md"}
        ]
        assert code_handler.get_python_file_count() == 2

    def test_has_evaluation_code_true(self, code_handler):
        """Test detecting evaluation code"""
        code_handler._repo_tree = [
            {"path": "evaluation/eval.py"}
        ]
        assert code_handler.has_evaluation_code() is True

    def test_has_evaluation_code_false(self, code_handler):
        """Test no evaluation code"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_evaluation_code() is False

    def test_has_formatting_check_true(self, code_handler):
        """Test detecting formatting config"""
        code_handler._repo_tree = [
            {"path": "pyproject.toml"}
        ]
        assert code_handler.has_formatting_check() is True

    def test_has_formatting_check_false(self, code_handler):
        """Test no formatting config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_formatting_check() is False

    def test_has_coverage_config_true(self, code_handler):
        """Test detecting coverage config"""
        code_handler._repo_tree = [
            {"path": ".coveragerc"}
        ]
        assert code_handler.has_coverage_config() is True

    def test_has_coverage_config_false(self, code_handler):
        """Test no coverage config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_coverage_config() is False

    def test_get_code_quality_score(self, code_handler, mock_github_api):
        """Test code quality scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_readme": True,
            "stargazers_count": 150,
            "updated_at": "2024-06-01"
        }
        mock_github_api.return_value = mock_response

        code_handler._repo_tree = [
            {"path": "tests/test_main.py"},
            {"path": ".github/workflows/ci.yml"},
            {"path": ".flake8"}
        ]

        score = code_handler.get_code_quality_score()
        assert 0.0 <= score <= 1.0

    def test_get_license_score(self, code_handler, mock_github_api):
        """Test license scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": {"spdx_id": "MIT"}
        }
        mock_github_api.return_value = mock_response

        score = code_handler.get_license_score()
        assert score > 0.0

    def test_get_license_score_no_license(self, code_handler, mock_github_api):
        """Test no license found"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_github_api.return_value = mock_response

        score = code_handler.get_license_score()
        assert score == 0.0

    def test_get_documentation_score(self, code_handler, mock_github_api):
        """Test documentation scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_readme": True,
            "description": "A great project with detailed documentation",
            "has_wiki": True,
            "homepage": "https://example.com"
        }
        mock_github_api.return_value = mock_response

        code_handler._repo_tree = [
            {"path": "docs/guide.md"}
        ]

        score = code_handler.get_documentation_score()
        assert 0.0 <= score <= 1.0

    def test_get_contributor_count_success(self, code_handler, mock_github_api):
        """Test getting contributor count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"login": "user1"}, {"login": "user2"}]
        mock_github_api.return_value = mock_response

        count = code_handler.get_contributor_count()
        assert count == 2

    def test_get_contributor_count_fallback(self, code_handler, mock_github_api):
        """Test contributor count fallback"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        # Set cached API data for fallback
        code_handler._cache_set("github_api_data", {"stargazers_count": 150})

        count = code_handler.get_contributor_count()
        assert count > 0

    def test_get_stars(self, code_handler, mock_github_api):
        """Test getting star count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_github_api.return_value = mock_response

        stars = code_handler.get_stars()
        assert stars == 100

    def test_get_forks(self, code_handler, mock_github_api):
        """Test getting fork count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"forks_count": 20}
        mock_github_api.return_value = mock_response

        forks = code_handler.get_forks()
        assert forks == 20


class TestDatasetHandlerComprehensive:
    """Comprehensive tests for DatasetHandler"""

    @pytest.fixture
    def mock_hf_api(self):
        """Mock HuggingFace API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def dataset_handler(self):
        """Create a DatasetHandler instance"""
        return DatasetHandler("https://huggingface.co/datasets/owner/dataset")

    def test_extract_dataset_id(self, dataset_handler):
        """Test extracting dataset ID"""
        assert dataset_handler.dataset_id == "owner/dataset"

    def test_get_huggingface_api_data_success(self, dataset_handler, mock_hf_api):
        """Test successful API data fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 1000,
            "tags": ["nlp", "text-classification"]
        }
        mock_hf_api.return_value = mock_response

        data = dataset_handler.get_huggingface_api_data()
        assert data["downloads"] == 1000

    def test_get_huggingface_api_data_failure(self, dataset_handler, mock_hf_api):
        """Test API data fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_hf_api.return_value = mock_response

        data = dataset_handler.get_huggingface_api_data()
        assert data == {}

    def test_get_readme_content_success(self, dataset_handler, mock_hf_api):
        """Test README fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# Dataset README\n\nThis is a test dataset."
        mock_hf_api.return_value = mock_response

        readme = dataset_handler.get_readme_content()
        assert "Dataset README" in readme

    def test_get_readme_content_failure(self, dataset_handler, mock_hf_api):
        """Test README fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_hf_api.return_value = mock_response

        readme = dataset_handler.get_readme_content()
        assert readme == ""

    def test_has_evaluation_dataset_true(self, dataset_handler, mock_hf_api):
        """Test identifying evaluation dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["evaluation", "benchmark"]
        }
        mock_hf_api.return_value = mock_response

        assert dataset_handler.has_evaluation_dataset() is True

    def test_has_evaluation_dataset_false(self, dataset_handler, mock_hf_api):
        """Test non-evaluation dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["nlp", "text"]
        }
        mock_hf_api.return_value = mock_response

        assert dataset_handler.has_evaluation_dataset() is False

    def test_get_quality_score(self, dataset_handler, mock_hf_api):
        """Test quality scoring"""
        # Mock API call
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"dataset_info": {"splits": []}},
            "description": "A comprehensive dataset for testing",
            "downloads": 5000,
            "tags": ["nlp", "text", "classification"],
            "siblings": [{"rfilename": "data.csv"}]
        }

        # Mock README call
        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "# Dataset\n" + "x" * 1000

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = dataset_handler.get_quality_score()
        assert 0.0 <= score <= 1.0

    def test_get_documentation_score(self, dataset_handler, mock_hf_api):
        """Test documentation scoring"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"dataset_info": {}},
            "description": "A well-documented dataset with examples",
            "tags": ["nlp", "text", "classification", "benchmark"]
        }

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "# Dataset\n" + "x" * 1500

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = dataset_handler.get_documentation_score()
        assert 0.0 <= score <= 1.0

    def test_get_license_score(self, dataset_handler, mock_hf_api):
        """Test license scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": "mit"
        }
        mock_hf_api.return_value = mock_response

        score = dataset_handler.get_license_score()
        assert score > 0.0

    def test_get_contributor_count(self, dataset_handler, mock_hf_api):
        """Test contributor count estimation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 15000
        }
        mock_hf_api.return_value = mock_response

        count = dataset_handler.get_contributor_count()
        assert count > 0

    def test_get_tags(self, dataset_handler, mock_hf_api):
        """Test getting tags"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["nlp", "text"]
        }
        mock_hf_api.return_value = mock_response

        tags = dataset_handler.get_tags()
        assert len(tags) == 2

    def test_get_downloads(self, dataset_handler, mock_hf_api):
        """Test getting download count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 5000
        }
        mock_hf_api.return_value = mock_response

        downloads = dataset_handler.get_downloads()
        assert downloads == 5000


class TestModelHandlerComprehensive:
    """Comprehensive tests for ModelHandler"""

    @pytest.fixture
    def mock_hf_api(self):
        """Mock HuggingFace API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def model_handler(self):
        """Create a ModelHandler instance"""
        return ModelHandler("https://huggingface.co/owner/model")

    def test_extract_model_id(self, model_handler):
        """Test extracting model ID"""
        assert model_handler.model_id == "owner/model"

    def test_get_huggingface_api_data_dict(self, model_handler, mock_hf_api):
        """Test API data as dict"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 10000,
            "likes": 100
        }
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 10000

    def test_get_huggingface_api_data_list_match(self, model_handler, mock_hf_api):
        """Test API data as list with exact match"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "other/model", "downloads": 500},
            {"id": "owner/model", "downloads": 10000}
        ]
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 10000

    def test_get_huggingface_api_data_list_no_match(self, model_handler, mock_hf_api):
        """Test API data as list without exact match"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "other/model", "downloads": 500}
        ]
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 500

    def test_extract_github_urls(self, model_handler, mock_hf_api):
        """Test extracting GitHub URLs from README"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        # Model
        Code: https://github.com/owner/repo
        Training: https://github.com/owner/training-code
        """
        mock_hf_api.return_value = mock_response

        urls = model_handler.extract_github_urls_from_readme()
        assert len(urls) >= 1

    def test_extract_dataset_urls(self, model_handler, mock_hf_api):
        """Test extracting dataset URLs from README"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        # Model
        Trained on: https://huggingface.co/datasets/owner/dataset
        """
        mock_hf_api.return_value = mock_response

        urls = model_handler.extract_dataset_urls_from_readme()
        assert len(urls) >= 1

    def test_get_size_mb_from_safetensors(self, model_handler, mock_hf_api):
        """Test size calculation from safetensors"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "safetensors": {"total": 500000000}  # 500MB
        }
        mock_hf_api.return_value = mock_response

        size = model_handler.get_size_mb()
        assert size > 400  # ~477 MB

    def test_get_size_mb_from_files(self, model_handler, mock_hf_api):
        """Test size calculation from file list"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {}

        mock_files_response = Mock()
        mock_files_response.status_code = 200
        mock_files_response.json.return_value = [
            {"size": 100000000},  # 100MB
            {"size": 50000000}    # 50MB
        ]

        mock_hf_api.side_effect = [mock_api_response, mock_files_response]

        size = model_handler.get_size_mb()
        assert size > 100

    def test_has_performance_benchmarks_true(self, model_handler, mock_hf_api):
        """Test detecting performance benchmarks"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"model-index": [{"results": []}]}
        }
        mock_hf_api.return_value = mock_api_response

        assert model_handler.has_performance_benchmarks() is True

    def test_has_performance_benchmarks_in_readme(self, model_handler, mock_hf_api):
        """Test detecting benchmarks in README"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {}

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "Performance: 95% accuracy on benchmark"

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        assert model_handler.has_performance_benchmarks() is True

    def test_get_license_score_from_api(self, model_handler, mock_hf_api):
        """Test license from API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": "apache-2.0"
        }
        mock_hf_api.return_value = mock_response

        score = model_handler.get_license_score()
        assert score > 0.0

    def test_get_documentation_score(self, model_handler, mock_hf_api):
        """Test documentation scoring"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"model-index": []}
        }

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = """
        # Model
        ## Usage
        Example code here
        ## Training
        Training details
        ## Installation
        pip install requirements
        ## Citation
        BibTeX here
        """ + "x" * 1000

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = model_handler.get_documentation_score()
        assert score > 0.5

    def test_get_contributor_count(self, model_handler, mock_hf_api):
        """Test contributor count estimation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 150000,
            "likes": 600
        }
        mock_hf_api.return_value = mock_response

        count = model_handler.get_contributor_count()
        assert count >= 10

    def test_get_tags(self, model_handler, mock_hf_api):
        """Test getting tags"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["transformers", "bert"]
        }
        mock_hf_api.return_value = mock_response

        tags = model_handler.get_tags()
        assert len(tags) == 2

    def test_get_hf_model_info_success(self, model_handler, mock_hf_api):
        """Test get_hf_model_info success"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "data"}
        mock_hf_api.return_value = mock_response

        info = model_handler.get_hf_model_info()
        assert info["model"] == "data"

    def test_get_hf_model_info_raises(self, model_handler, mock_hf_api):
        """Test get_hf_model_info raises exception"""
        mock_hf_api.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            model_handler.get_hf_model_info()

# ==================================================
# BEGIN test_handlers_mocks.py
# ==================================================

"""
Comprehensive tests for handler modules to improve coverage
"""


class TestCodeHandlerComprehensive:
    """Comprehensive tests for CodeHandler"""

    @pytest.fixture
    def mock_github_api(self):
        """Mock GitHub API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def code_handler(self):
        """Create a CodeHandler instance"""
        return CodeHandler("https://github.com/owner/repo")

    def test_extract_repo_path(self, code_handler):
        """Test extracting repo path from URL"""
        assert code_handler.repo_path == "owner/repo"

    def test_get_github_api_data_success(self, code_handler, mock_github_api):
        """Test successful GitHub API data fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stargazers_count": 100,
            "forks_count": 20,
            "has_readme": True,
            "updated_at": "2024-01-01"
        }
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data["stargazers_count"] == 100
        assert data["forks_count"] == 20

    def test_get_github_api_data_401(self, code_handler, mock_github_api):
        """Test GitHub API 401 authentication failure"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_404(self, code_handler, mock_github_api):
        """Test GitHub API 404 not found"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_other_error(self, code_handler, mock_github_api):
        """Test GitHub API other error codes"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_github_api.return_value = mock_response

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_github_api_data_exception(self, code_handler, mock_github_api):
        """Test GitHub API exception handling"""
        mock_github_api.side_effect = Exception("Network error")

        data = code_handler.get_github_api_data()
        assert data == {}

    def test_get_repo_tree_success(self, code_handler, mock_github_api):
        """Test successful repo tree fetch"""
        # Mock API data call
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {"default_branch": "main"}

        # Mock tree call
        mock_tree_response = Mock()
        mock_tree_response.status_code = 200
        mock_tree_response.json.return_value = {
            "tree": [
                {"path": "test/test_file.py", "type": "blob"},
                {"path": ".github/workflows/ci.yml", "type": "blob"}
            ]
        }

        mock_github_api.side_effect = [mock_api_response, mock_tree_response]

        tree = code_handler.get_repo_tree()
        assert len(tree) == 2
        assert tree[0]["path"] == "test/test_file.py"

    def test_get_repo_tree_failure(self, code_handler, mock_github_api):
        """Test repo tree fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        tree = code_handler.get_repo_tree()
        assert tree == []

    def test_get_repo_tree_exception(self, code_handler, mock_github_api):
        """Test repo tree exception handling"""
        mock_github_api.side_effect = Exception("Network error")

        tree = code_handler.get_repo_tree()
        assert tree == []

    def test_has_tests_true(self, code_handler):
        """Test detecting test files"""
        code_handler._repo_tree = [
            {"path": "tests/test_main.py"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_tests() is True

    def test_has_tests_false(self, code_handler):
        """Test no test files"""
        code_handler._repo_tree = [
            {"path": "src/main.py"},
            {"path": "README.md"}
        ]
        assert code_handler.has_tests() is False

    def test_has_ci_cd_true(self, code_handler):
        """Test detecting CI/CD config"""
        code_handler._repo_tree = [
            {"path": ".github/workflows/ci.yml"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_ci_cd() is True

    def test_has_ci_cd_travis(self, code_handler):
        """Test detecting Travis CI"""
        code_handler._repo_tree = [
            {"path": ".travis.yml"}
        ]
        assert code_handler.has_ci_cd() is True

    def test_has_ci_cd_false(self, code_handler):
        """Test no CI/CD config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_ci_cd() is False

    def test_has_linting_config_true(self, code_handler):
        """Test detecting linting config"""
        code_handler._repo_tree = [
            {"path": ".flake8"},
            {"path": "src/main.py"}
        ]
        assert code_handler.has_linting_config() is True

    def test_has_linting_config_false(self, code_handler):
        """Test no linting config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_linting_config() is False

    def test_get_python_file_count(self, code_handler):
        """Test counting Python files"""
        code_handler._repo_tree = [
            {"path": "src/main.py"},
            {"path": "tests/test_main.py"},
            {"path": "README.md"}
        ]
        assert code_handler.get_python_file_count() == 2

    def test_has_evaluation_code_true(self, code_handler):
        """Test detecting evaluation code"""
        code_handler._repo_tree = [
            {"path": "evaluation/eval.py"}
        ]
        assert code_handler.has_evaluation_code() is True

    def test_has_evaluation_code_false(self, code_handler):
        """Test no evaluation code"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_evaluation_code() is False

    def test_has_formatting_check_true(self, code_handler):
        """Test detecting formatting config"""
        code_handler._repo_tree = [
            {"path": "pyproject.toml"}
        ]
        assert code_handler.has_formatting_check() is True

    def test_has_formatting_check_false(self, code_handler):
        """Test no formatting config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_formatting_check() is False

    def test_has_coverage_config_true(self, code_handler):
        """Test detecting coverage config"""
        code_handler._repo_tree = [
            {"path": ".coveragerc"}
        ]
        assert code_handler.has_coverage_config() is True

    def test_has_coverage_config_false(self, code_handler):
        """Test no coverage config"""
        code_handler._repo_tree = [
            {"path": "src/main.py"}
        ]
        assert code_handler.has_coverage_config() is False

    def test_get_code_quality_score(self, code_handler, mock_github_api):
        """Test code quality scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_readme": True,
            "stargazers_count": 150,
            "updated_at": "2024-06-01"
        }
        mock_github_api.return_value = mock_response

        code_handler._repo_tree = [
            {"path": "tests/test_main.py"},
            {"path": ".github/workflows/ci.yml"},
            {"path": ".flake8"}
        ]

        score = code_handler.get_code_quality_score()
        assert 0.0 <= score <= 1.0

    def test_get_license_score(self, code_handler, mock_github_api):
        """Test license scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": {"spdx_id": "MIT"}
        }
        mock_github_api.return_value = mock_response

        score = code_handler.get_license_score()
        assert score > 0.0

    def test_get_license_score_no_license(self, code_handler, mock_github_api):
        """Test no license found"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_github_api.return_value = mock_response

        score = code_handler.get_license_score()
        assert score == 0.0

    def test_get_documentation_score(self, code_handler, mock_github_api):
        """Test documentation scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_readme": True,
            "description": "A great project with detailed documentation",
            "has_wiki": True,
            "homepage": "https://example.com"
        }
        mock_github_api.return_value = mock_response

        code_handler._repo_tree = [
            {"path": "docs/guide.md"}
        ]

        score = code_handler.get_documentation_score()
        assert 0.0 <= score <= 1.0

    def test_get_contributor_count_success(self, code_handler, mock_github_api):
        """Test getting contributor count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"login": "user1"}, {"login": "user2"}]
        mock_github_api.return_value = mock_response

        count = code_handler.get_contributor_count()
        assert count == 2

    def test_get_contributor_count_fallback(self, code_handler, mock_github_api):
        """Test contributor count fallback"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_github_api.return_value = mock_response

        # Set cached API data for fallback
        code_handler._cache_set("github_api_data", {"stargazers_count": 150})

        count = code_handler.get_contributor_count()
        assert count > 0

    def test_get_stars(self, code_handler, mock_github_api):
        """Test getting star count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_github_api.return_value = mock_response

        stars = code_handler.get_stars()
        assert stars == 100

    def test_get_forks(self, code_handler, mock_github_api):
        """Test getting fork count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"forks_count": 20}
        mock_github_api.return_value = mock_response

        forks = code_handler.get_forks()
        assert forks == 20


class TestDatasetHandlerComprehensive:
    """Comprehensive tests for DatasetHandler"""

    @pytest.fixture
    def mock_hf_api(self):
        """Mock HuggingFace API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def dataset_handler(self):
        """Create a DatasetHandler instance"""
        return DatasetHandler("https://huggingface.co/datasets/owner/dataset")

    def test_extract_dataset_id(self, dataset_handler):
        """Test extracting dataset ID"""
        assert dataset_handler.dataset_id == "owner/dataset"

    def test_get_huggingface_api_data_success(self, dataset_handler, mock_hf_api):
        """Test successful API data fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 1000,
            "tags": ["nlp", "text-classification"]
        }
        mock_hf_api.return_value = mock_response

        data = dataset_handler.get_huggingface_api_data()
        assert data["downloads"] == 1000

    def test_get_huggingface_api_data_failure(self, dataset_handler, mock_hf_api):
        """Test API data fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_hf_api.return_value = mock_response

        data = dataset_handler.get_huggingface_api_data()
        assert data == {}

    def test_get_readme_content_success(self, dataset_handler, mock_hf_api):
        """Test README fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# Dataset README\n\nThis is a test dataset."
        mock_hf_api.return_value = mock_response

        readme = dataset_handler.get_readme_content()
        assert "Dataset README" in readme

    def test_get_readme_content_failure(self, dataset_handler, mock_hf_api):
        """Test README fetch failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_hf_api.return_value = mock_response

        readme = dataset_handler.get_readme_content()
        assert readme == ""

    def test_has_evaluation_dataset_true(self, dataset_handler, mock_hf_api):
        """Test identifying evaluation dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["evaluation", "benchmark"]
        }
        mock_hf_api.return_value = mock_response

        assert dataset_handler.has_evaluation_dataset() is True

    def test_has_evaluation_dataset_false(self, dataset_handler, mock_hf_api):
        """Test non-evaluation dataset"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["nlp", "text"]
        }
        mock_hf_api.return_value = mock_response

        assert dataset_handler.has_evaluation_dataset() is False

    def test_get_quality_score(self, dataset_handler, mock_hf_api):
        """Test quality scoring"""
        # Mock API call
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"dataset_info": {"splits": []}},
            "description": "A comprehensive dataset for testing",
            "downloads": 5000,
            "tags": ["nlp", "text", "classification"],
            "siblings": [{"rfilename": "data.csv"}]
        }

        # Mock README call
        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "# Dataset\n" + "x" * 1000

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = dataset_handler.get_quality_score()
        assert 0.0 <= score <= 1.0

    def test_get_documentation_score(self, dataset_handler, mock_hf_api):
        """Test documentation scoring"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"dataset_info": {}},
            "description": "A well-documented dataset with examples",
            "tags": ["nlp", "text", "classification", "benchmark"]
        }

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "# Dataset\n" + "x" * 1500

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = dataset_handler.get_documentation_score()
        assert 0.0 <= score <= 1.0

    def test_get_license_score(self, dataset_handler, mock_hf_api):
        """Test license scoring"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": "mit"
        }
        mock_hf_api.return_value = mock_response

        score = dataset_handler.get_license_score()
        assert score > 0.0

    def test_get_contributor_count(self, dataset_handler, mock_hf_api):
        """Test contributor count estimation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 15000
        }
        mock_hf_api.return_value = mock_response

        count = dataset_handler.get_contributor_count()
        assert count > 0

    def test_get_tags(self, dataset_handler, mock_hf_api):
        """Test getting tags"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["nlp", "text"]
        }
        mock_hf_api.return_value = mock_response

        tags = dataset_handler.get_tags()
        assert len(tags) == 2

    def test_get_downloads(self, dataset_handler, mock_hf_api):
        """Test getting download count"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 5000
        }
        mock_hf_api.return_value = mock_response

        downloads = dataset_handler.get_downloads()
        assert downloads == 5000


class TestModelHandlerComprehensive:
    """Comprehensive tests for ModelHandler"""

    @pytest.fixture
    def mock_hf_api(self):
        """Mock HuggingFace API responses"""
        with patch('requests.get') as mock_get:
            yield mock_get

    @pytest.fixture
    def model_handler(self):
        """Create a ModelHandler instance"""
        return ModelHandler("https://huggingface.co/owner/model")

    def test_extract_model_id(self, model_handler):
        """Test extracting model ID"""
        assert model_handler.model_id == "owner/model"

    def test_get_huggingface_api_data_dict(self, model_handler, mock_hf_api):
        """Test API data as dict"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 10000,
            "likes": 100
        }
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 10000

    def test_get_huggingface_api_data_list_match(self, model_handler, mock_hf_api):
        """Test API data as list with exact match"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "other/model", "downloads": 500},
            {"id": "owner/model", "downloads": 10000}
        ]
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 10000

    def test_get_huggingface_api_data_list_no_match(self, model_handler, mock_hf_api):
        """Test API data as list without exact match"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "other/model", "downloads": 500}
        ]
        mock_hf_api.return_value = mock_response

        data = model_handler.get_huggingface_api_data()
        assert data["downloads"] == 500

    def test_extract_github_urls(self, model_handler, mock_hf_api):
        """Test extracting GitHub URLs from README"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        # Model
        Code: https://github.com/owner/repo
        Training: https://github.com/owner/training-code
        """
        mock_hf_api.return_value = mock_response

        urls = model_handler.extract_github_urls_from_readme()
        assert len(urls) >= 1

    def test_extract_dataset_urls(self, model_handler, mock_hf_api):
        """Test extracting dataset URLs from README"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        # Model
        Trained on: https://huggingface.co/datasets/owner/dataset
        """
        mock_hf_api.return_value = mock_response

        urls = model_handler.extract_dataset_urls_from_readme()
        assert len(urls) >= 1

    def test_get_size_mb_from_safetensors(self, model_handler, mock_hf_api):
        """Test size calculation from safetensors"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "safetensors": {"total": 500000000}  # 500MB
        }
        mock_hf_api.return_value = mock_response

        size = model_handler.get_size_mb()
        assert size > 400  # ~477 MB

    def test_get_size_mb_from_files(self, model_handler, mock_hf_api):
        """Test size calculation from file list"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {}

        mock_files_response = Mock()
        mock_files_response.status_code = 200
        mock_files_response.json.return_value = [
            {"size": 100000000},  # 100MB
            {"size": 50000000}    # 50MB
        ]

        mock_hf_api.side_effect = [mock_api_response, mock_files_response]

        size = model_handler.get_size_mb()
        assert size > 100

    def test_has_performance_benchmarks_true(self, model_handler, mock_hf_api):
        """Test detecting performance benchmarks"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"model-index": [{"results": []}]}
        }
        mock_hf_api.return_value = mock_api_response

        assert model_handler.has_performance_benchmarks() is True

    def test_has_performance_benchmarks_in_readme(self, model_handler, mock_hf_api):
        """Test detecting benchmarks in README"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {}

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "Performance: 95% accuracy on benchmark"

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        assert model_handler.has_performance_benchmarks() is True

    def test_get_license_score_from_api(self, model_handler, mock_hf_api):
        """Test license from API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "license": "apache-2.0"
        }
        mock_hf_api.return_value = mock_response

        score = model_handler.get_license_score()
        assert score > 0.0

    def test_get_documentation_score(self, model_handler, mock_hf_api):
        """Test documentation scoring"""
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "cardData": {"model-index": []}
        }

        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = """
        # Model
        ## Usage
        Example code here
        ## Training
        Training details
        ## Installation
        pip install requirements
        ## Citation
        BibTeX here
        """ + "x" * 1000

        mock_hf_api.side_effect = [mock_api_response, mock_readme_response]

        score = model_handler.get_documentation_score()
        assert score > 0.5

    def test_get_contributor_count(self, model_handler, mock_hf_api):
        """Test contributor count estimation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "downloads": 150000,
            "likes": 600
        }
        mock_hf_api.return_value = mock_response

        count = model_handler.get_contributor_count()
        assert count >= 10

    def test_get_tags(self, model_handler, mock_hf_api):
        """Test getting tags"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tags": ["transformers", "bert"]
        }
        mock_hf_api.return_value = mock_response

        tags = model_handler.get_tags()
        assert len(tags) == 2

    def test_get_hf_model_info_success(self, model_handler, mock_hf_api):
        """Test get_hf_model_info success"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "data"}
        mock_hf_api.return_value = mock_response

        info = model_handler.get_hf_model_info()
        assert info["model"] == "data"

    def test_get_hf_model_info_raises(self, model_handler, mock_hf_api):
        """Test get_hf_model_info raises exception"""
        mock_hf_api.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            model_handler.get_hf_model_info()

# ==================================================
# BEGIN test_ingest_handler.py
# ==================================================

"""
Additional tests for ingest_handler to increase coverage
"""

from unittest.mock import patch, MagicMock, Mock
import tempfile


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
        
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.side_effect = Exception("API Error")
        
        metadata, card_text = ih._extract_metadata("test/model", mock_api_instance)
        
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
        
        mock_head.return_value.status_code = 404
        
        result = ih.validate_url("https://github.com/test/nonexistent", "github")
        self.assertFalse(result)

    @patch("requests.head")
    def test_validate_url_exception(self, mock_head):
        """Test URL validation with exception"""
        
        mock_head.side_effect = Exception("Network error")
        
        result = ih.validate_url("https://github.com/test/repo", "github")
        self.assertFalse(result)

    def test_validate_url_unknown(self):
        """Test validation of 'unknown' URL"""
        
        result = ih.validate_url("unknown", "github")
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
        
        result = {
            "net_score": 0.9,
            "dataset_and_code_score": 0.3,  # Below threshold
            "dataset_quality": 0.7,
            "code_quality": 0.6,
        }
        
        passes = ih._check_threshold(result, 0.5)
        self.assertFalse(passes)

    def test_check_threshold_with_minus_one(self):
        """Test threshold check ignores -1 values"""
        
        result = {
            "net_score": 0.9,
            "dataset_and_code_score": 0.8,
            "dataset_quality": -1,  # Should be ignored
            "code_quality": 0.6,
        }
        
        passes = ih._check_threshold(result, 0.5)
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
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"size": 10240}  # KB
        mock_get.return_value = mock_response
        
        size = ih._calculate_artifact_size("https://github.com/test/repo", "code")
        self.assertGreater(size, 0)

    def test_calculate_artifact_size_unknown(self):
        """Test artifact size with unknown URL"""
        
        size = ih._calculate_artifact_size("unknown", "model")
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
        
        mock_ingest.side_effect = [
            {"status": "success"},
            Exception("Failed")
        ]
        
        urls = [
            "https://huggingface.co/model1",
            "https://huggingface.co/model2"
        ]
        
        results = ih.batch_ingest(urls)
        
        self.assertEqual(len(results), 2)
        self.assertIn("error", results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ==================================================
# BEGIN test_ingest_handler_more.py
# ==================================================



import handlers.ingest_handler as ih


def test_extract_card_data_and_load_card_content(monkeypatch):
    # dict card data
    md, txt = ih._extract_card_data({"a": 1, "b": 2})
    assert md.get("a") == 1

    class Obj:
        def to_dict(self):
            return {"x": "y"}

    md2, txt2 = ih._extract_card_data(Obj())
    assert md2.get("x") == "y"

    # load card content success and fail
    class Card:
        content = "hello"

    monkeypatch.setattr("handlers.ingest_handler.RepoCard.load", lambda mid: Card())
    assert ih._load_card_content("owner/model") == "hello"

    monkeypatch.setattr("handlers.ingest_handler.RepoCard.load", lambda mid: (_ for _ in ()).throw(Exception("nope")))
    assert ih._load_card_content("owner/model") == ""


def test_extract_metadata_and_url_extraction(monkeypatch):
    class API:
        def model_info(self, mid):
            class Info:
                cardData = {"datasets": ["owner/ds"]}

            return Info()

    monkeypatch.setattr("handlers.ingest_handler.HfApi", lambda: API())
    md, card = ih._extract_metadata("owner/model", API())
    assert isinstance(md, dict)

    # github extraction
    assert ih._extract_github_url("see https://github.com/owner/repo here") == "https://github.com/owner/repo"
    assert ih._extract_github_url("github.com/owner/repo.git") == "https://github.com/owner/repo"

    # dataset extraction from text and metadata
    assert ih._extract_dataset_url_direct("see https://huggingface.co/datasets/owner/ds") == "https://huggingface.co/datasets/owner/ds"
    assert ih._extract_dataset_from_metadata({"datasets": ["owner/ds"]}) == "owner/ds"
    assert ih._extract_dataset_from_yaml("datasets:\n- owner/ds") == "owner/ds"
    assert ih._extract_dataset_from_prose("trained on owner/ds dataset") == "owner/ds"
    assert ih._construct_dataset_url("Owner/Ds") == "https://huggingface.co/datasets/owner/ds"


def test_extract_dataset_url_strategy():
    # direct
    assert ih._extract_dataset_url({}, "see https://huggingface.co/datasets/owner/ds") == "https://huggingface.co/datasets/owner/ds"
    # metadata
    assert ih._extract_dataset_url({"datasets": ["owner/ds"]}, "") == "https://huggingface.co/datasets/owner/ds"
    # yaml
    assert ih._extract_dataset_url({}, "datasets:\n- owner/ds") == "https://huggingface.co/datasets/owner/ds"
    # prose
    assert ih._extract_dataset_url({}, "trained on owner/ds dataset") == "https://huggingface.co/datasets/owner/ds"
    # none
    assert ih._extract_dataset_url({}, "nothing here") == "unknown"


def test_validate_url(monkeypatch):
    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 200})())
    assert ih.validate_url("https://example.com", "github") is True

    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 404})())
    assert ih.validate_url("https://example.com", "github") is False

    monkeypatch.setattr("requests.head", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert ih.validate_url("https://example.com", "github") is False


def test_ensure_and_check_threshold():
    res = {}
    ih._ensure_required_metrics(res)
    assert res.get("dataset_and_code_score") == -1

    good = {"a": 1.0, "b": 0.9}
    assert ih._check_threshold({"a": 1.0, "b": 0.8}, 0.5) is True
    assert ih._check_threshold({"a": 0.4, "b": 0.8}, 0.5) is False


def test_calculate_artifact_size_hf_and_github(monkeypatch):
    # huggingface with siblings
    class Info:
        siblings = [type("F", (), {"size": 5 * 1024 * 1024}), type("F", (), {"size": None}), type("F", (), {"size": 2 * 1024 * 1024})]

    class API:
        def model_info(self, repo_id):
            return Info()

        def dataset_info(self, repo_id):
            return Info()

    monkeypatch.setattr("handlers.ingest_handler.HfApi", lambda: API())
    size = ih._calculate_artifact_size("https://huggingface.co/owner/model", "model")
    assert size > 0

    # github path
    class R:
        status_code = 200

        def json(self):
            return {"size": 2048}

    import requests as _req

    monkeypatch.setattr("requests.get", lambda *a, **k: R())
    size2 = ih._calculate_artifact_size("https://github.com/owner/repo", "code")
    assert size2 > 0


def test_ingest_model_and_dataset_and_code(monkeypatch, tmp_path):
    # snapshot download fail
    monkeypatch.setattr("handlers.ingest_handler.snapshot_download", lambda **k: (_ for _ in ()).throw(Exception("nope")))
    res = ih.ingest_model("https://huggingface.co/owner/model", download=True)
    assert "error" in res

    # successful ingest_model flow (mock evaluator and registry)
    monkeypatch.setattr("handlers.ingest_handler.infer_links_from_hf", lambda u, show_card=False: ("https://github.com/owner/repo", "https://huggingface.co/datasets/owner/ds"))
    monkeypatch.setattr("handlers.ingest_handler._extract_metadata", lambda mid, api: ({}, ""))
    monkeypatch.setattr("handlers.ingest_handler.score_model_with_evaluator", lambda a, b, c: {"net_score": 0.9})
    monkeypatch.setattr("handlers.ingest_handler.registry_handler", type("R", (), {"add_model": lambda **k: 123}))

    res2 = ih.ingest_model("https://huggingface.co/owner/model", download=False, validate_urls=False)
    assert res2.get("status") == "success"

    # ingest_dataset and ingest_code
    monkeypatch.setattr("handlers.ingest_handler.registry_handler", type("R", (), {"add_artifact": lambda **k: 321}))
    dres = ih.ingest_dataset("https://huggingface.co/datasets/owner/ds")
    assert dres.get("status") == "success"

    cres = ih.ingest_code("https://github.com/owner/repo")
    assert cres.get("status") == "success"


def test_batch_ingest(monkeypatch):
    monkeypatch.setattr("handlers.ingest_handler.ingest_model", lambda url, min_score=0.5, download=False, validate_urls=False: {"status": "ok", "url": url})
    results = ih.batch_ingest(["a", "b"])
    assert len(results) == 2


def test_extract_card_data_variants():
    # dict input
    meta, text = ih._extract_card_data({"a": 1, "b": 2})
    assert isinstance(meta, dict)
    assert "1" in text or "2" in text

    class Box:
        def to_dict(self):
            return {"x": "y"}

    meta2, text2 = ih._extract_card_data(Box())
    assert meta2.get("x") == "y"


def test_extract_github_and_dataset_urls():
    # github
    assert ih._extract_github_url("See https://github.com/owner/repo for code") == "https://github.com/owner/repo"
    assert ih._extract_github_url("owner/repo") == "unknown"

    # dataset direct
    assert ih._extract_dataset_url_direct("... https://huggingface.co/datasets/owner/ds ...") == "https://huggingface.co/datasets/owner/ds"
    # yaml extraction
    txt = "datasets:\n  - owner/ds"
    assert ih._extract_dataset_from_yaml(txt) == "owner/ds"
    # prose extraction
    assert ih._extract_dataset_from_prose("Trained on the owner/ds dataset") == "owner/ds"


def test_construct_dataset_url_and_extract():
    url = ih._construct_dataset_url(" Owner/DS ")
    assert url.endswith("owner/ds")

    # If nothing found, returns unknown
    assert ih._extract_dataset_url({}, "no data here") == "unknown"


def test_validate_url_and_head(monkeypatch):
    class R: pass
    r = R(); r.status_code = 200
    import sys
    monkeypatch.setitem(sys.modules, "requests", Mock(head=lambda *a, **k: r))
    assert ih.validate_url("https://example.com", "github") is True

    r2 = R(); r2.status_code = 404
    monkeypatch.setitem(sys.modules, "requests", Mock(head=lambda *a, **k: r2))
    assert ih.validate_url("https://example.com", "github") is False


def test_score_model_with_evaluator_fallback(monkeypatch, tmp_path):
    # Force subprocess to raise so fallback is used
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: json.dumps([{"net_score": 0.9}]))

    # Use real fallback path by making subprocess succeed
    res = ih.score_model_with_evaluator("http://code", "http://data", "http://model")
    assert isinstance(res, dict)


def test_ensure_required_and_threshold():
    r = {"dataset_quality": 0.6, "code_quality": 0.6}
    ih._ensure_required_metrics(r)
    # required keys exist
    assert "dataset_and_code_score" in r

    assert ih._check_threshold({"a": 0.6, "b": 0.7}, 0.5) is True
    assert ih._check_threshold({"a": 0.4, "b": 0.7}, 0.5) is False


def test_calculate_artifact_size_github_and_hf(monkeypatch):
    # GitHub branch
    class Resp: pass
    r = Resp(); r.status_code = 200; r.json = lambda: {"size": 2048}
    monkeypatch.setitem(sys.modules, "requests", Mock(get=lambda *a, **k: r))
    size = ih._calculate_artifact_size("https://github.com/owner/repo", "code")
    assert size > 0

    # HF dataset branch - simulate api with siblings sizes
    class Info: pass
    f = Info()
    class File: pass
    file = File(); file.size = 1024 * 1024  # 1MB
    f.siblings = [file]

    monkeypatch.setattr(ih, "HfApi", lambda *a, **k: Mock(dataset_info=lambda repo: f, model_info=lambda repo: f))
    size2 = ih._calculate_artifact_size("https://huggingface.co/datasets/owner/ds", "dataset")
    assert size2 > 0


def test_ingest_dataset_and_code_and_batch(monkeypatch):
    # ingest_dataset and ingest_code should add artifacts
    res = ih.ingest_dataset("https://huggingface.co/datasets/owner/ds")
    assert res["status"] == "success"

    res2 = ih.ingest_code("https://github.com/owner/repo")
    assert res2["status"] == "success"

    # batch_ingest handles error gracefully
    out = ih.batch_ingest(["https://huggingface.co/models/owner/model", "bad://url"])
    assert isinstance(out, list)

# ==================================================
# BEGIN test_model_handler_more.py
# ==================================================



import handlers.model_handler as mh


class FakeResp:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("bad")


def test_extract_model_id_and_readme_and_urls(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    assert m.model_id == "owner/model"

    # readme extraction
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="See https://github.com/owner/repo and https://huggingface.co/datasets/owner/ds"))
    assert "https://github.com/owner/repo" in m.extract_github_urls_from_readme()
    assert "https://huggingface.co/datasets/owner/ds" in m.extract_dataset_urls_from_readme()


def test_get_hf_api_data_list_and_errors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # list with exact match
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"id": "owner/model", "downloads": 1}]))
    d = m.get_huggingface_api_data()
    assert d.get("downloads") == 1

    # empty list -> returns {}
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, []))
    assert m2.get_huggingface_api_data() == {}

    # unexpected type
    m3 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, "not-a-dict"))
    assert m3.get_huggingface_api_data() == {}


def test_get_model_files_and_resolve_size_and_get_size_mb(monkeypatch, tmp_path):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # files listing
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"path": "a", "size": 1024}, {"path": "b", "size": 2048}]))
    files = m.get_model_files()
    assert isinstance(files, list)

    # HEAD probe finds content-length
    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 200, "headers": {"Content-Length": "3145728"}})())
    sz = m._resolve_file_size_via_head("pytorch_model.bin")
    assert sz > 0

    # get_size_mb with safetensors
    m._cache_set("hf_api_data", {"safetensors": {"total": 10 * 1024 * 1024}})
    assert m.get_size_mb() >= 10


def test_has_performance_benchmarks_and_docs(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    m._cache_set("hf_api_data", {"cardData": {"model-index": True}})
    assert m.has_performance_benchmarks() is True

    # readme keywords
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="This benchmark achieved high accuracy and has usage examples"))
    assert m2.has_performance_benchmarks() is True


def test_license_and_doc_and_contributors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # license via api
    m._cache_set("hf_api_data", {"license": "MIT"})
    assert m.get_license_score() > 0

    # license via README YAML
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="---\nlicense: GPL-3.0\n---\n"))
    assert m2.get_license_score() >= 0

    # documentation score
    m3 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="Usage example: how to use this model for training and evaluation"))
    assert m3.get_documentation_score() > 0

    # contributors heuristic
    m4 = mh.ModelHandler("https://huggingface.co/owner/model")
    m4._cache_set("hf_api_data", {"downloads": 20000, "likes": 0})
    assert m4.get_contributor_count() >= 10


def test_get_hf_model_info_errors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # success
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, {"a": 1}))
    assert isinstance(m.get_hf_model_info(), dict)

    # failure should raise (use a fresh handler to avoid cache)
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(Exception("nope")))
    with pytest.raises(Exception):
        m2.get_hf_model_info()


def test_get_model_files_success_and_failure():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    mock_resp = Mock(); mock_resp.status_code = 200; mock_resp.json.return_value = [{"size": 1024}]
    with patch("handlers.model_handler.requests.get", return_value=mock_resp):
        files = m.get_model_files()
        assert isinstance(files, list)

    # Exception path
    def bad_get(*a, **k):
        raise Exception("boom")

    m_new = mh.ModelHandler("https://huggingface.co/owner/model-new")
    with patch("handlers.model_handler.requests.get", side_effect=bad_get):
        assert m_new.get_model_files() == []


def test_resolve_file_size_and_get_size_mb_branches(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # safetensors path
    monkeypatch.setattr(m, "get_huggingface_api_data", lambda: {"safetensors": {"total": 2 * 1024 * 1024}})
    assert m.get_size_mb() >= 2.0

    # tree sizes path
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    monkeypatch.setattr(m2, "get_huggingface_api_data", lambda: {})
    monkeypatch.setattr(m2, "get_model_files", lambda: [{"size": 1024 * 1024}, {"size": "2048"}])
    assert m2.get_size_mb() > 0

    # siblings rsize path
    m3 = mh.ModelHandler("https://huggingface.co/owner/model3")
    monkeypatch.setattr(m3, "get_huggingface_api_data", lambda: {"siblings": [{"rsize": 1024}, {"size": 1024}]})
    assert m3.get_size_mb() >= 0

    # HEAD probe path
    m4 = mh.ModelHandler("https://huggingface.co/owner/model4")
    monkeypatch.setattr(m4, "get_huggingface_api_data", lambda: {})
    monkeypatch.setattr(m4, "get_model_files", lambda: [])
    monkeypatch.setattr(m4, "_resolve_file_size_via_head", lambda fname: 1024 * 1024)
    assert m4.get_size_mb() > 0


def test_has_benchmarks_and_license_parsing():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # cardData model-index
    monkeypatch1 = patch("handlers.model_handler.ModelHandler.get_huggingface_api_data", return_value={"cardData": {"model-index": [1]}})
    with monkeypatch1:
        assert m.has_performance_benchmarks() is True

    # README benchmark keywords
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    monkeypatch2 = patch("handlers.model_handler.ModelHandler.get_readme_content", return_value="This shows benchmark results and accuracy")
    with monkeypatch2:
        assert m2.has_performance_benchmarks() is True

    # License in YAML frontmatter
    m3 = mh.ModelHandler("https://huggingface.co/owner/model3")
    monkeypatch3 = patch("handlers.model_handler.ModelHandler.get_readme_content", return_value="---\nlicense: mit\n---\n rest")
    with monkeypatch3:
        assert m3.get_license_score() == 1.0


def test_get_huggingface_api_data_list_errors():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # empty list
    resp = Mock(); resp.status_code = 200; resp.json.return_value = []
    with patch("handlers.model_handler.requests.get", return_value=resp):
        assert m.get_huggingface_api_data() == {}

    # list with non-dict first item
    resp2 = Mock(); resp2.status_code = 200; resp2.json.return_value = ["nope", 123]
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    with patch("handlers.model_handler.requests.get", return_value=resp2):
        assert m2.get_huggingface_api_data() == {}