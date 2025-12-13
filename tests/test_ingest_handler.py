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


import json

import pytest

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
from unittest.mock import Mock, patch
import handlers.ingest_handler as ih
import json


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
    import sys
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


if __name__ == "__main__":
    unittest.main(verbosity=2)