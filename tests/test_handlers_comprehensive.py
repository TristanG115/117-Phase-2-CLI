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