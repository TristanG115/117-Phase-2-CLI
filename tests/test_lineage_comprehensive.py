"""
Comprehensive tests for lineage.py to improve coverage
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from lineage import LineageExtractor, get_lineage_extractor


class TestLineageExtractorComprehensive:
    """Comprehensive tests for LineageExtractor"""

    @pytest.fixture
    def extractor_with_api(self):
        """Create extractor with API key"""
        with patch.dict('os.environ', {'GEN_AI_STUDIO_API_KEY': 'test-key'}):
            return LineageExtractor()

    @pytest.fixture
    def extractor_without_api(self):
        """Create extractor without API key"""
        with patch.dict('os.environ', {}, clear=True):
            return LineageExtractor()

    def test_init_with_api_key(self, extractor_with_api):
        """Test initialization with API key"""
        assert extractor_with_api.api_available is True
        assert extractor_with_api.api_key == 'test-key'

    def test_init_without_api_key(self, extractor_without_api):
        """Test initialization without API key"""
        assert extractor_without_api.api_available is False

    @patch('requests.post')
    def test_call_genai_api_success(self, mock_post, extractor_with_api):
        """Test successful GenAI API call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "training_datasets": ["squad"],
                        "code_repositories": ["https://github.com/owner/repo"],
                        "parent_models": ["bert-base"],
                        "evaluation_datasets": ["glue"]
                    })
                }
            }]
        }
        mock_post.return_value = mock_response

        result = extractor_with_api._call_genai_api("test prompt", "test readme")
        assert result is not None
        assert "training_datasets" in result

    @patch('requests.post')
    def test_call_genai_api_with_markdown(self, mock_post, extractor_with_api):
        """Test API response with markdown code blocks"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '```json\n{"training_datasets": ["squad"]}\n```'
                }
            }]
        }
        mock_post.return_value = mock_response

        result = extractor_with_api._call_genai_api("test prompt", "test readme")
        assert result is not None
        assert "training_datasets" in result

    @patch('requests.post')
    def test_call_genai_api_invalid_json(self, mock_post, extractor_with_api):
        """Test API response with invalid JSON"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON"
                }
            }]
        }
        mock_post.return_value = mock_response

        result = extractor_with_api._call_genai_api("test prompt", "test readme")
        assert result is None

    @patch('requests.post')
    def test_call_genai_api_non_200(self, mock_post, extractor_with_api):
        """Test API non-200 response"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = extractor_with_api._call_genai_api("test prompt", "test readme")
        assert result is None

    @patch('requests.post')
    def test_call_genai_api_timeout(self, mock_post, extractor_with_api):
        """Test API timeout"""
        mock_post.side_effect = Exception("Timeout")

        result = extractor_with_api._call_genai_api("test prompt", "test readme")
        assert result is None

    def test_call_genai_api_no_key(self, extractor_without_api):
        """Test API call without key"""
        result = extractor_without_api._call_genai_api("test prompt", "test readme")
        assert result is None

    @patch('requests.post')
    def test_extract_lineage_with_api(self, mock_post, extractor_with_api):
        """Test lineage extraction with API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "training_datasets": ["https://huggingface.co/datasets/squad"],
                        "code_repositories": ["https://github.com/owner/repo"],
                        "parent_models": ["bert-base-uncased"],
                        "evaluation_datasets": ["glue"]
                    })
                }
            }]
        }
        mock_post.return_value = mock_response

        readme = "This model was trained on SQuAD dataset"
        result = extractor_with_api.extract_lineage(readme, "test-model")

        assert "datasets" in result
        assert "code_repos" in result
        assert "parent_models" in result
        assert "evaluation_datasets" in result

    def test_extract_lineage_short_readme(self, extractor_with_api):
        """Test with too-short README"""
        result = extractor_with_api.extract_lineage("short", "test-model")
        assert result["datasets"] == []

    def test_extract_lineage_truncation(self, extractor_with_api):
        """Test README truncation"""
        long_readme = "x" * 15000
        with patch.object(extractor_with_api, '_call_genai_api') as mock_call:
            mock_call.return_value = {
                "training_datasets": [],
                "code_repositories": [],
                "parent_models": [],
                "evaluation_datasets": []
            }
            result = extractor_with_api.extract_lineage(long_readme, "test-model")
            # Verify truncation happened
            call_args = mock_call.call_args[0]
            assert len(call_args[1]) < 15000

    def test_fallback_extraction_hf_datasets(self, extractor_without_api):
        """Test fallback extraction of HF datasets"""
        readme = """
        This model uses https://huggingface.co/datasets/squad for training.
        """
        result = extractor_without_api._fallback_extraction(readme)
        assert any("squad" in ds for ds in result["datasets"])

    def test_fallback_extraction_github_repos(self, extractor_without_api):
        """Test fallback extraction of GitHub repos"""
        readme = """
        Code available at https://github.com/owner/repo
        """
        result = extractor_without_api._fallback_extraction(readme)
        assert any("owner/repo" in repo for repo in result["code_repos"])

    def test_fallback_extraction_yaml_datasets(self, extractor_without_api):
        """Test fallback extraction from YAML"""
        readme = """
        ---
        datasets:
          - squad
          - glue
        ---
        """
        result = extractor_without_api._fallback_extraction(readme)
        assert len(result["datasets"]) > 0

    def test_fallback_extraction_parent_models(self, extractor_without_api):
        """Test fallback extraction of parent models"""
        readme = """
        This model is fine-tuned from bert-base-uncased.
        It is based on roberta-large.
        """
        result = extractor_without_api._fallback_extraction(readme)
        assert len(result["parent_models"]) > 0

    def test_fallback_extraction_evaluation_datasets(self, extractor_without_api):
        """Test fallback extraction of eval datasets"""
        readme = """
        The model was evaluated on GLUE benchmark.
        """
        result = extractor_without_api._fallback_extraction(readme)
        assert len(result["evaluation_datasets"]) > 0

    def test_fallback_extraction_empty_readme(self, extractor_without_api):
        """Test fallback with empty README"""
        result = extractor_without_api._fallback_extraction("")
        assert result["datasets"] == []
        assert result["code_repos"] == []

    def test_fallback_extraction_deduplication(self, extractor_without_api):
        """Test that fallback deduplicates entries"""
        readme = """
        https://github.com/owner/repo
        https://github.com/owner/repo
        """
        result = extractor_without_api._fallback_extraction(readme)
        # Should only have one repo
        assert len(result["code_repos"]) == 1

    def test_normalize_urls_datasets(self, extractor_without_api):
        """Test URL normalization for datasets"""
        lineage = {
            "datasets": [
                "https://huggingface.co/datasets/squad",
                "glue"
            ],
            "code_repos": [],
            "parent_models": [],
            "evaluation_datasets": []
        }
        result = extractor_without_api.normalize_urls(lineage)
        assert all("https://" in ds for ds in result["datasets"])

    def test_normalize_urls_code_repos(self, extractor_without_api):
        """Test URL normalization for code repos"""
        lineage = {
            "datasets": [],
            "code_repos": [
                "https://github.com/owner/repo.git",
                "owner/repo2"
            ],
            "parent_models": [],
            "evaluation_datasets": []
        }
        result = extractor_without_api.normalize_urls(lineage)
        assert all("https://github.com/" in repo for repo in result["code_repos"])
        assert not any(repo.endswith(".git") for repo in result["code_repos"])

    def test_normalize_urls_trailing_slashes(self, extractor_without_api):
        """Test removal of trailing slashes"""
        lineage = {
            "datasets": [],
            "code_repos": ["https://github.com/owner/repo/"],
            "parent_models": [],
            "evaluation_datasets": []
        }
        result = extractor_without_api.normalize_urls(lineage)
        assert not result["code_repos"][0].endswith("/")

    def test_extract_from_model_metadata_with_lineage(self, extractor_with_api):
        """Test extract_from_model_metadata with lineage"""
        with patch.object(extractor_with_api, 'extract_lineage') as mock_extract:
            mock_extract.return_value = {
                "datasets": ["https://huggingface.co/datasets/squad"],
                "code_repos": ["https://github.com/owner/repo"],
                "parent_models": ["bert-base"],
                "evaluation_datasets": ["glue"]
            }

            readme = "Test README"
            metadata = {}
            code_url, dataset_url, parents, evals = extractor_with_api.extract_from_model_metadata(
                readme, metadata, "test-model"
            )

            assert code_url == "https://github.com/owner/repo"
            assert dataset_url == "https://huggingface.co/datasets/squad"
            assert len(parents) == 1
            assert len(evals) == 1

    def test_extract_from_model_metadata_no_lineage(self, extractor_with_api):
        """Test extract_from_model_metadata without lineage"""
        with patch.object(extractor_with_api, 'extract_lineage') as mock_extract:
            mock_extract.return_value = {
                "datasets": [],
                "code_repos": [],
                "parent_models": [],
                "evaluation_datasets": []
            }

            readme = "Test README"
            metadata = {}
            code_url, dataset_url, parents, evals = extractor_with_api.extract_from_model_metadata(
                readme, metadata, "test-model"
            )

            assert code_url == "unknown"
            assert dataset_url == "unknown"

    def test_extract_from_model_metadata_with_metadata_dataset(self, extractor_with_api):
        """Test extraction with dataset in metadata"""
        with patch.object(extractor_with_api, 'extract_lineage') as mock_extract:
            mock_extract.return_value = {
                "datasets": [],
                "code_repos": [],
                "parent_models": [],
                "evaluation_datasets": []
            }

            readme = "Test README"
            metadata = {"datasets": ["squad"]}
            code_url, dataset_url, parents, evals = extractor_with_api.extract_from_model_metadata(
                readme, metadata, "test-model"
            )

            assert "squad" in dataset_url

    def test_extract_from_model_metadata_with_string_dataset(self, extractor_with_api):
        """Test extraction with string dataset in metadata"""
        with patch.object(extractor_with_api, 'extract_lineage') as mock_extract:
            mock_extract.return_value = {
                "datasets": [],
                "code_repos": [],
                "parent_models": [],
                "evaluation_datasets": []
            }

            readme = "Test README"
            metadata = {"datasets": "squad"}
            code_url, dataset_url, parents, evals = extractor_with_api.extract_from_model_metadata(
                readme, metadata, "test-model"
            )

            assert "squad" in dataset_url

    def test_get_lineage_extractor_singleton(self):
        """Test singleton pattern for extractor"""
        extractor1 = get_lineage_extractor()
        extractor2 = get_lineage_extractor()
        assert extractor1 is extractor2

    @patch('requests.post')
    def test_extract_lineage_fallback_on_api_failure(self, mock_post, extractor_with_api):
        """Test fallback when API fails"""
        mock_post.side_effect = Exception("API Error")

        readme = """
        Training data: https://huggingface.co/datasets/squad
        Code: https://github.com/owner/repo
        """
        result = extractor_with_api.extract_lineage(readme, "test-model")

        # Should fall back to regex extraction
        assert len(result["datasets"]) > 0 or len(result["code_repos"]) > 0

    def test_normalize_urls_preserves_parent_models(self, extractor_without_api):
        """Test that normalization preserves parent model names"""
        lineage = {
            "datasets": [],
            "code_repos": [],
            "parent_models": ["bert-base-uncased", "roberta-large"],
            "evaluation_datasets": ["glue", "squad"]
        }
        result = extractor_without_api.normalize_urls(lineage)
        assert result["parent_models"] == ["bert-base-uncased", "roberta-large"]
        assert result["evaluation_datasets"] == ["glue", "squad"]