import json
import os
import unittest
import pytest
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
from unittest.mock import Mock, patch
import handlers.model_handler as mh


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

"""
Comprehensive tests for handler modules to improve coverage
"""
import pytest
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


"""
Comprehensive tests for handler modules to improve coverage
"""
import pytest
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



import pytest

import handlers.dataset_handler as dh


class FakeResp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text

    def json(self):
        return self._data


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
from unittest.mock import Mock, patch

import handlers.dataset_handler as dh


def make_api_response(**kwargs):
    r = Mock()
    r.status_code = 200
    r.json.return_value = kwargs
    return r


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

from unittest.mock import Mock, patch

import handlers.dataset_handler as dh


def test_extract_dataset_id_variants():
    d1 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert d1.dataset_id == "owner/ds"

    d2 = dh.DatasetHandler("https://huggingface.co/datasets/owner")
    assert d2.dataset_id == "owner"

    d3 = dh.DatasetHandler("https://huggingface.co/owner/ds")
    assert d3.dataset_id == ""


def test_get_huggingface_api_data_list_response():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_response = Mock()
    mock_response.status_code = 200
    # Return a list where one item matches expected id
    mock_response.json.return_value = [
        {"id": "other/thing"},
        {"id": "owner/ds", "downloads": 5000},
    ]

    with patch("requests.get", return_value=mock_response):
        data = handler.get_huggingface_api_data()
        assert data.get("downloads") == 5000


def test_has_evaluation_dataset_true_false():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tags": ["Evaluation", "nlp"]}
    with patch("requests.get", return_value=mock_response):
        assert handler.has_evaluation_dataset() is True

    mock_response2 = Mock()
    mock_response2.status_code = 200
    mock_response2.json.return_value = {"tags": ["nlp"]}
    handler2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    with patch("requests.get", return_value=mock_response2):
        assert handler2.has_evaluation_dataset() is False


def test_quality_score_and_components():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_api_response = Mock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = {
        "cardData": {"dataset_info": {"rows": 100}},
        "description": "x" * 250,
        "downloads": 2000,
        "tags": ["a", "b", "c"],
        "siblings": [{}, {}],
    }

    mock_readme = Mock()
    mock_readme.status_code = 200
    mock_readme.text = "x" * 600

    with patch("requests.get", side_effect=[mock_api_response, mock_readme]):
        qscore = handler.get_quality_score()
        assert qscore > 0.5


def test_get_license_score_variants():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # license in root
    r1 = Mock(); r1.status_code = 200; r1.json.return_value = {"license": "mit"}
    with patch("requests.get", return_value=r1):
        assert handler.get_license_score() > 0

    # license in cardData
    r2 = Mock(); r2.status_code = 200; r2.json.return_value = {"cardData": {"license": "apache-2.0"}}
    with patch("requests.get", return_value=r2):
        assert handler.get_license_score() > 0

    # license in tags
    r3 = Mock(); r3.status_code = 200; r3.json.return_value = {"tags": ["license:bsd-3-clause"]}
    with patch("requests.get", return_value=r3):
        assert handler.get_license_score() > 0


def test_contributor_count_thresholds():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    r = Mock(); r.status_code = 200; r.json.return_value = {"downloads": 150000}
    with patch("requests.get", return_value=r):
        assert handler.get_contributor_count() >= 10


def test_get_hf_dataset_info_stub():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    out = handler.get_hf_dataset_info("some_url")
    assert out["status"] == "ok"

if __name__ == "__main__":
    unittest.main(verbosity=2)